"""
train_fifo_depth_specialists.py
Treina um modelo XGBoost especialista de FIFO depth para cada topologia.
Salva: StreamingFIFO_depth_{TOPOLOGY}_model.pkl

Melhorias v2:
  - Log-transform no target (log1p/expm1): elimina dominância dos outliers no MSE
  - Stage 1: classificador binário depth==2 vs depth>2 (rule-based + LightGBM fallback)
  - Features de interação explícitas: log(cycle_ratio), chain*cycles, spatial*cycle_ratio
  - Feature op_pair: interação entre produtor_op e consumidor_op
  - Sample weights proporcionais a log1p(depth): foco crescente em casos difíceis
  - Métricas em escala log para avaliação honesta
"""

import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
import argparse

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "results", "fifo_depth", "fifo_backpressure_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "results", "trained_models")

NUMERIC_FEATURES = [
    # Tensor
    "dataType_bits", "tensor_volume", "tensor_spatial", "tensor_C",
    # Immediate producer/consumer
    "produtor_PE", "produtor_cycles", "p_throughput", "p_transfers",
    "consumidor_SIMD", "consumidor_cycles", "c_throughput", "c_transfers",
    "cycle_ratio", "parallelism_mismatch",
    "theoretical_accumulation", "theoretical_fifo_depth",
    # Computational (look-through DWC/bridges)
    "comp_produtor_PE", "comp_produtor_SIMD", "comp_produtor_cycles", "comp_p_throughput",
    "comp_consumidor_PE", "comp_consumidor_SIMD", "comp_consumidor_cycles", "comp_c_throughput",
    "comp_cycle_ratio", "comp_theo_accumulation", "comp_theo_depth", "comp_parallelism_mismatch",
    # Spatial attrs from compute nodes
    "p_IFMDim", "p_OFMDim", "p_KernelDim", "p_IFMChannels", "p_OFMChannels",
    "c_IFMDim", "c_IFMChannels", "window_volume",
    "p_IFMDim_sq", "p_OFMDim_sq", "window_area",
    # CIG startup features (buffer needed before first output)
    "cig_warmup_rows", "cig_startup_vol",
    # Derived
    "drain_time", "fill_time", "channel_per_spatial",
    # NOTE: chain_length intentionally excluded — it's determined BY depth (data leakage)
]

CATEGORICAL_FEATURES = [
    "produtor_op", "consumidor_op", "op_pair",
    "comp_produtor_op", "comp_consumidor_op"
]

TARGET = "real_depth"

TOPOLOGY_MAP = {
    #"MNIST_1W1A":  "MNIST_TFC",
    "SAT6_T2W2":   "SAT6_T2",
    #"CIFAR10_1W1A": "CIFAR10_CNV",
    #"CIFAR10_2W2A": "CIFAR10_CNV",
}


def get_topology(session_name):
    for prefix, topo in TOPOLOGY_MAP.items():
        if session_name.startswith(prefix):
            return topo
    return "UNKNOWN"


# =============================================================================
# STAGE 1 — Classificador binário depth==2
# Regra determinística cobre ~97% dos casos com cycle_ratio>2.
# O XGBClassifier captura o restante.
# =============================================================================

def is_depth2_rule(df: pd.DataFrame) -> np.ndarray:
    """
    Regra determinística derivada da análise exploratória:
      cycle_ratio > 2  →  97%+ são depth==2
      speed_ratio < 1  (consumidor mais rápido) →  81.6% são depth==2
    Retorna array booleano: True = certamente depth==2, False = incerto.
    """
    cr = df["cycle_ratio"].fillna(0).values
    # speed_ratio = p_throughput / c_throughput
    p_thr = df["p_throughput"].fillna(0).values
    c_thr = df["c_throughput"].fillna(1e-12).values
    speed_ratio = p_thr / np.clip(c_thr, 1e-12, None)

    # Regra conservadora: só marca como depth==2 quando o sinal é muito forte
    certain_min = (cr > 10) & (speed_ratio < 1.0)
    return certain_min


def build_stage1_features(df: pd.DataFrame, numeric_feats: list, cat_feats: list) -> np.ndarray:
    """Monta matrix de features para o classificador stage 1."""
    df_f = df[numeric_feats].copy()
    for cat_col in cat_feats:
        if cat_col in df.columns:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, dtype=int)
            df_f = pd.concat([df_f, dummies], axis=1)
    return df_f.values, list(df_f.columns)


# =============================================================================
# FEATURES ENGENHEIRADAS v2
# =============================================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- interação op_pair ---
    if "produtor_op" in df.columns and "consumidor_op" in df.columns:
        df["op_pair"] = df["produtor_op"].astype(str) + "→" + df["consumidor_op"].astype(str)

    # --- log-transforms de features de entrada ---
    for col in ["cycle_ratio", "comp_cycle_ratio", "tensor_volume",
                "theoretical_fifo_depth", "comp_theo_depth",
                "consumidor_cycles", "comp_consumidor_cycles",
                "cig_startup_vol", "drain_time"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].fillna(0).clip(lower=0))

    if "tensor_spatial" in df.columns and "cycle_ratio" in df.columns:
        df["spatial_x_cycle"] = np.log1p(
            df["tensor_spatial"].fillna(0) * df["cycle_ratio"].fillna(0).clip(lower=0)
        )

    if "tensor_spatial" in df.columns:
        df["is_spatial_large"] = (df["tensor_spatial"] > 256).astype(int)

    # --- speed_ratio ---
    if "p_throughput" in df.columns and "c_throughput" in df.columns:
        df["speed_ratio"] = (
            df["p_throughput"].fillna(0) /
            df["c_throughput"].fillna(1e-12).clip(lower=1e-12)
        )
        df["log_speed_ratio"] = np.log1p(df["speed_ratio"].clip(lower=0))

    # Interação pesada para CIGs
    if "window_volume" in df.columns and "consumidor_cycles" in df.columns:
        # Janela grande + MVAU lenta = Explosão de Depth
        df["burst_x_cycles"] = np.log1p(df["window_volume"].fillna(0) * df["consumidor_cycles"].fillna(0))

    return df


def get_extended_numeric_features(base_feats: list, df: pd.DataFrame) -> list:
    """Retorna base_feats + features engenheiradas v2 que existem no df."""
    extra = [
        "log_cycle_ratio", "log_comp_cycle_ratio", "log_tensor_volume",
        "log_theoretical_fifo_depth", "log_comp_theo_depth",
        "log_consumidor_cycles", "log_comp_consumidor_cycles",
        "log_cig_startup_vol", "log_drain_time",
        "spatial_x_cycle", "is_spatial_large",
        "speed_ratio", "log_speed_ratio",
        "burst_x_cycles",
        # chain_length and derived features (log_chain_length, chain_x_cycles) intentionally excluded
    ]
    all_feats = list(base_feats)
    for f in extra:
        if f in df.columns and f not in all_feats:
            all_feats.append(f)
    return [f for f in all_feats if f in df.columns]


# =============================================================================
# TREINAMENTO PRINCIPAL
# =============================================================================

def train_specialist(df_topo: pd.DataFrame, topology_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f" Treinando especialista: {topology_name} ({len(df_topo)} amostras)")
    print(f"{'='*60}")

    # --- pré-processamento ---
    df_topo = df_topo.dropna(subset=[TARGET])
    df_topo = df_topo.replace([np.inf, -np.inf], 0)
    df_topo = df_topo.fillna(0)

    # features engenheiradas v2
    df_topo = add_engineered_features(df_topo)

    numeric_feats = get_extended_numeric_features(NUMERIC_FEATURES, df_topo)

    # --- target em escala log ---
    y_raw = df_topo[TARGET].values.astype(float)
    y_log = np.log1p(y_raw)   # treino/val em log; predição revertida com expm1

    print(f"  Depth range: [{y_raw.min():.0f}, {y_raw.max():.0f}] (median: {np.median(y_raw):.0f})")
    print(f"  Log-depth range: [{y_log.min():.2f}, {y_log.max():.2f}] (median: {np.median(y_log):.2f})")

    # --- one-hot encoding ---
    df_features = df_topo[numeric_feats].copy()
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in df_topo.columns:
            dummies = pd.get_dummies(df_topo[cat_col], prefix=cat_col, dtype=int)
            df_features = pd.concat([df_features, dummies], axis=1)

    X = df_features.values
    feature_names = list(df_features.columns)

    # --- split estratificado por faixa de depth ---
    depth_bins = pd.cut(y_raw, bins=[0, 2, 32, 256, 1024, 8192, np.inf],
                        labels=False, include_lowest=True)
    X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = train_test_split(
        X, y_log, y_raw, test_size=0.15, random_state=42, stratify=depth_bins
    )
    print(f"  Treino: {len(X_train)} | Teste: {len(X_test)}")

    # -----------------------------------------------------------------------
    # STAGE 1 — Classificador binário depth==2
    # -----------------------------------------------------------------------
    print(f"\n  [Stage 1] Treinando classificador binário depth==2 ...")

    y_train_is2 = (y_train_raw == 2).astype(int)
    pct_is2 = y_train_is2.mean() * 100
    print(f"  Classe depth==2 no treino: {pct_is2:.1f}%")

    scale_pos = (y_train_is2 == 0).sum() / max(1, (y_train_is2 == 1).sum())
    clf_stage1 = XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42, n_jobs=-1, eval_metric="logloss",
    )
    clf_stage1.fit(X_train, y_train_is2)

    # Threshold ajustado: preferimos recall alto para depth==2
    # (melhor prever depth>2 em poucos casos errados do que errar grandes depths)
    stage1_proba_test = clf_stage1.predict_proba(X_test)[:, 1]
    stage1_pred_test  = (stage1_proba_test >= 0.55).astype(int)

    # F1 do classificador no teste
    f1_s1 = f1_score((y_test_raw == 2).astype(int), stage1_pred_test)
    print(f"  Stage 1 F1 (depth==2): {f1_s1:.4f}")

    # -----------------------------------------------------------------------
    # STAGE 2 — Regressão em log-scale só para depth>2
    # -----------------------------------------------------------------------
    print(f"\n  [Stage 2] Treinando regressor log-scale (depth > 2) ...")

    # Treina o regressor apenas em amostras reais depth>2
    mask_gt2_train = y_train_raw > 2
    X_reg  = X_train[mask_gt2_train]
    y_reg  = y_train_log[mask_gt2_train]
    y_reg_raw = y_train_raw[mask_gt2_train]

    print(f"  Amostras para regressão: {mask_gt2_train.sum()} / {len(y_train_raw)}")

    # Sample weights: proporcionais a log1p(depth) — foco nos casos difíceis
    sample_weights_reg = np.log1p(y_reg_raw)
    sample_weights_reg = sample_weights_reg / sample_weights_reg.mean()

    # K-Fold CV no regressor
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2_log, fold_r2_raw = [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_reg)):
        Xkf_tr, Xkf_val   = X_reg[tr_idx], X_reg[val_idx]
        ykf_tr, ykf_val   = y_reg[tr_idx], y_reg[val_idx]
        wkf_tr            = sample_weights_reg[tr_idx]
        ykf_val_raw       = y_reg_raw[val_idx]

        m = XGBRegressor(
            n_estimators=800, learning_rate=0.03, max_depth=7,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, gamma=0.1,
            random_state=42, n_jobs=-1,
        )
        m.fit(Xkf_tr, ykf_tr, sample_weight=wkf_tr)

        pred_log = m.predict(Xkf_val)
        pred_raw = np.round(np.expm1(pred_log)).clip(min=3).astype(int)

        fold_r2_log.append(r2_score(ykf_val, pred_log))
        fold_r2_raw.append(r2_score(ykf_val_raw, pred_raw))

    print(f"  K-Fold R² (log-scale): {np.mean(fold_r2_log):.4f} (± {np.std(fold_r2_log):.4f})")
    print(f"  K-Fold R² (raw-scale): {np.mean(fold_r2_raw):.4f} (± {np.std(fold_r2_raw):.4f})")

    # Modelo final do regressor
    final_regressor = XGBRegressor(
        n_estimators=800, learning_rate=0.03, max_depth=7,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        random_state=42, n_jobs=-1,
    )
    final_regressor.fit(X_reg, y_reg, sample_weight=sample_weights_reg)

    # -----------------------------------------------------------------------
    # Avaliação no holdout (pipeline completo stage1 + stage2)
    # -----------------------------------------------------------------------
    stage1_proba = clf_stage1.predict_proba(X_test)[:, 1]
    stage1_pred  = (stage1_proba >= 0.55).astype(int)

    preds_final = np.zeros(len(X_test), dtype=float)

    # Casos classificados como depth==2
    mask_pred2 = stage1_pred == 1
    preds_final[mask_pred2] = 2.0

    # Casos classificados como depth>2 → regressor
    mask_predgt2 = ~mask_pred2
    if mask_predgt2.sum() > 0:
        pred_log_gt2 = final_regressor.predict(X_test[mask_predgt2])
        preds_final[mask_predgt2] = np.expm1(pred_log_gt2).clip(min=3)

    preds_final = np.round(preds_final).astype(int)

    # Métricas
    final_r2      = r2_score(y_test_raw, preds_final)
    final_r2_log  = r2_score(np.log1p(y_test_raw), np.log1p(preds_final))
    final_mae     = mean_absolute_error(y_test_raw, preds_final)
    final_rmse    = np.sqrt(mean_squared_error(y_test_raw, preds_final))

    print(f"\n  {'─'*50}")
    print(f"  🏆 Holdout R²        (raw):  {final_r2:.4f}")
    print(f"  🏆 Holdout R²    (log-scale): {final_r2_log:.4f}  ← métrica principal")
    print(f"  🏆 Holdout MAE:              {final_mae:.1f}")
    print(f"  🏆 Holdout RMSE:             {final_rmse:.1f}")
    print(f"  {'─'*50}")

    # Top exemplos
    df_ex = pd.DataFrame({"Real": y_test_raw, "Pred": preds_final})
    df_ex = df_ex.sort_values("Real", ascending=False)
    print(f"  Top 5 depths:  Real → Pred")
    for _, row in df_ex.head(5).iterrows():
        diff = int(row["Pred"]) - int(row["Real"])
        sign = "+" if diff >= 0 else ""
        print(f"    {int(row['Real']):>8} → {int(row['Pred']):>8}  ({sign}{diff})")

    # -----------------------------------------------------------------------
    # Salvar pipeline completo
    # -----------------------------------------------------------------------
    model_filename = f"StreamingFIFO_depth_{topology_name}_model.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)

    model_data = {
        # Stage 1: classificador binário
        "stage1_classifier":   clf_stage1,
        "stage1_threshold":    0.55,
        # Stage 2: regressor log-scale
        "stage2_regressor":    final_regressor,
        # Metadados
        "feature_names":       feature_names,
        "target_cols":         [TARGET],
        "topology":            topology_name,
        "log_transform_target": True,   # flag para o predictor saber reverter
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"  ✓ Salvo: {model_filename}")

    return {
        "topology":    topology_name,
        "n_samples":   len(df_topo),
        "kfold_r2_log": np.mean(fold_r2_log),
        "holdout_r2":   final_r2,
        "holdout_r2_log": final_r2_log,
        "holdout_mae":  final_mae,
        "holdout_rmse": final_rmse,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FIFO Depth Specialist Trainer v2")
    parser.add_argument("--filtered", action="store_true", help="Use filtered dataset (FPS >= 500)")
    parser.add_argument("--input",    type=str, default=None, help="Custom input dataset path")
    args = parser.parse_args()

    dataset_path = DATASET_PATH
    if args.filtered:
        dataset_path = DATASET_PATH.replace(".csv", "_filtered.csv")
    if args.input:
        dataset_path = args.input

    print(f"[{'='*75}]")
    print(f"[HARA] Treinamento de Especialistas de FIFO Depth v2 ({'FILTRADO' if args.filtered else 'COMPLETO'})")
    print(f"[{'='*75}]")

    if not os.path.exists(dataset_path):
        print(f"[!] Dataset não encontrado: {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    print(f"[*] Dataset carregado: {len(df)} FIFOs mapeadas.")

    df["topology"] = df["session"].apply(get_topology)

    topologies = df["topology"].unique()
    print(f"[*] Topologias encontradas: {list(topologies)}")
    print(f"[*] Distribuição:")
    for topo in sorted(topologies):
        n = len(df[df["topology"] == topo])
        print(f"    - {topo}: {n} amostras")

    os.makedirs(MODELS_DIR, exist_ok=True)

    results = []
    for topo in sorted(topologies):
        if topo == "UNKNOWN":
            continue
        df_topo = df[df["topology"] == topo].copy().reset_index(drop=True)
        result = train_specialist(df_topo, topo)
        results.append(result)

    # Modelo unificado (fallback)
    print(f"\n{'='*60}")
    print(f" Treinando modelo UNIFICADO (fallback)")
    print(f"{'='*60}")
    result_unified = train_specialist(df.copy().reset_index(drop=True), "UNIFIED")
    results.append(result_unified)

    # Resumo
    print(f"\n\n{'='*80}")
    print(f" RESUMO FINAL — Especialistas de FIFO Depth v2")
    print(f"{'='*80}")
    print(f"{'Topologia':<20} {'Amostras':>10} {'KFold R²log':>12} {'Hold R²raw':>11} {'Hold R²log':>11} {'MAE':>10} {'RMSE':>10}")
    print("-" * 84)
    for r in results:
        print(
            f"{r['topology']:<20} {r['n_samples']:>10} "
            f"{r['kfold_r2_log']:>12.4f} {r['holdout_r2']:>11.4f} "
            f"{r['holdout_r2_log']:>11.4f} {r['holdout_mae']:>10.1f} {r['holdout_rmse']:>10.1f}"
        )
    print("-" * 84)


if __name__ == "__main__":
    main()