"""
train_fifo_depth.py

Treina um modelo two-stage para prever FIFO depth antes da síntese HLS:

  Stage 1 — Classificador (XGBoostClassifier)
    Input : features de upstream/downstream layers
    Output: is_constrained (depth > 2 → 1, depth == 2 → 0)

  Stage 2 — Regressor (XGBoostRegressor)
    Input : mesmas features, apenas nas FIFOs constrained
    Output: log(depth)  → predito e depois exp()

Modelos salvos em results/trained_models/:
  StreamingFIFO_depth_classifier.pkl   ← Stage 1
  StreamingFIFO_depth_regressor.pkl    ← Stage 2
  StreamingFIFO_depth_cv_results.csv   ← métricas CV de cada stage
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

INPUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results", "fifo_depth", "exhaustive_fifo_depths.csv"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results", "trained_models"
)

PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots", "StreamingFIFO_depth")

N_SPLITS = 10

# Colunas que não são features
NON_FEATURE_COLS = [
    "model_id", "session", "timestamp", "run_name",
    "fifo_name", "depth", "is_constrained",
    "up_out_fifo_depths", "run_number", "is_baseline", "fifo_pos"
]

# Colunas categóricas que fazem sentido para one-hot
CATEGORICAL_COLS = [
    "up_layer_type", "up_impl_type", "up_op_type",
    "dn_layer_type", "dn_impl_type", "dn_op_type",
    "fifo_impl_style",
]

XGB_CLS_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

XGB_REG_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

# =============================================================================


def prepare_features(df):
    """
    Faz one-hot encoding das colunas categóricas e retorna X numérico.
    Retorna (X_df, feature_names).
    """
    drop_cols = [c for c in NON_FEATURE_COLS if c in df.columns]

    # Colunas que são strings mas não estão nas categóricas listadas
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    cat_to_encode = [c for c in CATEGORICAL_COLS if c in df.columns]
    other_str = [c for c in obj_cols if c not in cat_to_encode and c not in drop_cols]

    feat_df = df.drop(columns=drop_cols + other_str, errors="ignore")

    # One-hot das categóricas
    for col in cat_to_encode:
        if col in feat_df.columns:
            dummies = pd.get_dummies(feat_df[col], prefix=col, prefix_sep="_", dtype=int)
            feat_df = pd.concat([feat_df.drop(columns=[col]), dummies], axis=1)

    # Converte tudo para numérico
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return feat_df.values.astype(np.float32), list(feat_df.columns)


def plot_feature_importance_bar(importances, feature_names, title, out_path, top_n=20):
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    ax.barh([feature_names[i] for i in idx], importances[idx], color="steelblue")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_pred_vs_actual_depth(y_true, y_pred, title, out_path):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.2)
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ax.set_xlabel("Actual depth", fontsize=11)
    ax.set_ylabel("Predicted depth", fontsize=11)
    ax.set_title(f"R² = {r2:.4f}  |  MAE = {mae:.1f}", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(residuals, bins=40, color="coral", edgecolor="white", alpha=0.85)
    ax2.axvline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("Residual (Actual − Predicted)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title(f"Residuals  (mean={residuals.mean():.1f}, std={residuals.std():.1f})", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# =============================================================================
# STAGE 1 — CLASSIFICADOR
# =============================================================================

def train_classifier(df, X, y_cls, feature_names):
    print("\n" + "="*60)
    print("STAGE 1 — Classificador: is_constrained (depth > 2)")
    print(f"  Positivos (depth>2): {y_cls.sum()} ({y_cls.mean()*100:.1f}%)")
    print(f"  Negativos (depth=2): {(1-y_cls).sum()}")
    print("="*60)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_records = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y_cls)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y_cls[tr_idx], y_cls[val_idx]

        clf = XGBClassifier(**XGB_CLS_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, zero_division=0)
        fold_records.append({"fold": fold_idx+1, "accuracy": acc, "f1": f1})
        print(f"  Fold {fold_idx+1:2d}: Accuracy={acc:.4f}  F1={f1:.4f}")

    df_cv = pd.DataFrame(fold_records)
    print(f"\n  Média CV: Accuracy={df_cv['accuracy'].mean():.4f} ± {df_cv['accuracy'].std():.4f}")
    print(f"            F1      ={df_cv['f1'].mean():.4f} ± {df_cv['f1'].std():.4f}")

    # Treino final
    print(f"\n  Treinando classificador final em {len(X)} amostras...")
    final_clf = XGBClassifier(**XGB_CLS_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_clf.fit(X, y_cls)

    # Feature importance
    imp = final_clf.feature_importances_
    plot_feature_importance_bar(
        imp, feature_names,
        "Stage 1 — Classifier Feature Importance",
        os.path.join(PLOTS_DIR, "classifier_feature_importance.png")
    )

    return final_clf, df_cv


# =============================================================================
# STAGE 2 — REGRESSOR (em log-space)
# =============================================================================

def train_regressor(df_constrained, X_constrained, y_depth, feature_names):
    print("\n" + "="*60)
    print(f"STAGE 2 — Regressor: depth (apenas FIFOs constrained, n={len(X_constrained)})")
    print(f"  depth range: {y_depth.min():.0f} → {y_depth.max():.0f}")
    print("="*60)

    # Regressão em log(depth) para estabilizar a escala
    y_log = np.log1p(y_depth)

    kf = KFold(n_splits=min(N_SPLITS, len(X_constrained)//2), shuffle=True, random_state=42)
    fold_records = []

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_constrained)):
        X_tr, X_val = X_constrained[tr_idx], X_constrained[val_idx]
        y_tr, y_val = y_log[tr_idx], y_log[val_idx]

        reg = XGBRegressor(**XGB_REG_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(X_tr, y_tr)

        y_pred_log = reg.predict(X_val)
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_val)

        r2  = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        fold_records.append({"fold": fold_idx+1, "R2": r2, "MAE": mae, "RMSE": rmse})
        print(f"  Fold {fold_idx+1:2d}: R²={r2:.4f}  MAE={mae:.1f}  RMSE={rmse:.1f}")

    df_cv = pd.DataFrame(fold_records)
    print(f"\n  Média CV: R²={df_cv['R2'].mean():.4f} ± {df_cv['R2'].std():.4f}")
    print(f"            MAE={df_cv['MAE'].mean():.1f} ± {df_cv['MAE'].std():.1f}")

    # Treino final
    print(f"\n  Treinando regressor final em {len(X_constrained)} amostras...")
    final_reg = XGBRegressor(**XGB_REG_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_reg.fit(X_constrained, y_log)

    # Gráfico Actual vs Predicted (escala real)
    y_pred_final = np.expm1(final_reg.predict(X_constrained))
    y_true_final = np.expm1(y_log)
    plot_pred_vs_actual_depth(
        y_true_final, y_pred_final,
        "Stage 2 — Regressor: Actual vs Predicted depth",
        os.path.join(PLOTS_DIR, "regressor_actual_vs_pred.png")
    )

    # Feature importance
    plot_feature_importance_bar(
        final_reg.feature_importances_, feature_names,
        "Stage 2 — Regressor Feature Importance",
        os.path.join(PLOTS_DIR, "regressor_feature_importance.png")
    )

    return final_reg, df_cv


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if not os.path.exists(INPUT_CSV):
        print(f"[!] CSV não encontrado: {INPUT_CSV}")
        print("    Execute get_exhaustive_fifo_depths.py primeiro.")
        return

    print("=" * 60)
    print("HARA — Two-Stage FIFO Depth Prediction")
    print(f"Fonte: {INPUT_CSV}")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    print(f"\nTotal de FIFOs: {len(df)}")
    print(f"Modelos: {df['model_id'].unique()}")

    # Prepara features (one-hot + numérico)
    X, feature_names = prepare_features(df)
    y_cls   = df["is_constrained"].values.astype(int)
    y_depth = df["depth"].values.astype(np.float32)

    # ---- STAGE 1 ----
    clf_model, cv_cls = train_classifier(df, X, y_cls, feature_names)

    # ---- STAGE 2 ---- (apenas das FIFOs constrained)
    mask_constrained = (y_cls == 1)
    X_constrained = X[mask_constrained]
    y_depth_constrained = y_depth[mask_constrained]

    if X_constrained.shape[0] < 20:
        print("\n[!] FIFOs constrained insuficientes para treinar regressor.")
        reg_model, cv_reg = None, pd.DataFrame()
    else:
        reg_model, cv_reg = train_regressor(
            df[mask_constrained], X_constrained, y_depth_constrained, feature_names
        )

    # ---- Salva modelos ----
    clf_path = os.path.join(OUTPUT_DIR, "StreamingFIFO_depth_classifier.pkl")
    with open(clf_path, "wb") as f:
        pickle.dump({"model": clf_model, "feature_names": feature_names}, f)
    print(f"\n[✓] Classificador salvo: {clf_path}")

    if reg_model is not None:
        reg_path = os.path.join(OUTPUT_DIR, "StreamingFIFO_depth_regressor.pkl")
        with open(reg_path, "wb") as f:
            pickle.dump({"model": reg_model, "feature_names": feature_names}, f)
        print(f"[✓] Regressor salvo:     {reg_path}")

    # ---- Salva CV results ----
    cv_all = pd.concat([
        cv_cls.assign(stage="classifier"),
        cv_reg.assign(stage="regressor") if not cv_reg.empty else pd.DataFrame()
    ], ignore_index=True)
    cv_path = os.path.join(OUTPUT_DIR, "StreamingFIFO_depth_cv_results.csv")
    cv_all.to_csv(cv_path, index=False)
    print(f"[✓] CV results salvo: {cv_path}")

    # ---- Tabela de feature_names para uso em inferência ----
    feat_path = os.path.join(OUTPUT_DIR, "StreamingFIFO_depth_feature_names.json")
    with open(feat_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"[✓] Feature names salvo: {feat_path}")

    print("\n✅ Pipeline two-stage concluído.")
    print("   Para prever depth de uma nova FIFO:")
    print("   1. Monta vetor com mesmas features (order = feature_names.json)")
    print("   2. clf.predict(X) → is_constrained")
    print("   3. Se constrained: np.expm1(reg.predict(X)) → depth")


import json  # necessário para salvar feature_names

if __name__ == "__main__":
    main()
