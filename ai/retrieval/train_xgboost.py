"""
train_xgboost.py
===================================================================
Treina os especialistas XGBoost do HARAv2 e integra o novo fluxo de
StreamingFIFO:

1. os módulos "clássicos" continuam sendo treinados aqui;
2. o classificador BRAM/LUT de FIFO passa a ser treinado aqui também;
3. o modelo de área por fatias (SplitFIFO_area) continua sendo treinado
   em train_split_fifo_area.py, mas este script agora detecta sua ausência
   e avisa claramente.

Observação:
- Os modelos salvos aqui usam log1p nos targets, exceto o modelo
  SplitFIFO_area, que é treinado separadamente com targets reais.
"""

import ast
import os
import pickle
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBRegressor

# =============================================================================
# CONFIGURAÇÃO GLOBAL
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_SPLITTED_DIR = os.path.join(BASE_DIR, "results", "splitted")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "trained_models")
FIFO_DEPTH_CSV = os.path.join(BASE_DIR, "results", "fifo_depth", "exhaustive_fifo_depths.csv")
SPLIT_FIFO_MODEL_PATH = os.path.join(OUTPUT_DIR, "SplitFIFO_area_model.pkl")
STREAMING_FIFO_RAW_PATH = os.path.join(RAW_SPLITTED_DIR, "exhaustive_StreamingFIFO_area_attrs.csv")

SKIP_EXTRA_DROPS = False
TARGET_COLS = ["Total LUT", "Total FFs", "BRAM (36k eq.)", "DSP Blocks"]

LEAKAGE_COLS = [
    "model_id",
    "session",
    "timestamp",
    "run_name",
    "run_number",
    "is_baseline",
    "fixed_ram_style",
    "fixed_resType",
    "Submodule Instance",
    "base_name",
    "layer_idx",
    "Hardware config",
]

N_SPLITS = 10

# =============================================================================
# CONFIGURAÇÕES ESPECIALISTAS POR MÓDULO
# =============================================================================

MODULE_CONFIGS = {
    "MVAU_MNIST_1W1A": {
        "raw_filename": "exhaustive_MVAU_area_attrs.csv",
        "ohe_cols": ["ram_style", "resType"],
        "extra_drops": [
            "cycles_estimate",
            "estimated_cycles",
            "op_type",
            "runtime_writeable_weights",
            "mem_mode",
        ],
        "bitwidth_cols": ["inputDataType", "weightDataType", "outputDataType"],
        "corr_threshold": None,
        "xgb_params": dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
        ),
    },
    "MVAU_SAT6_T2W2": {
        "raw_filename": "exhaustive_MVAU_area_attrs.csv",
        "ohe_cols": ["ram_style", "resType"],
        "extra_drops": [
            "cycles_estimate",
            "estimated_cycles",
            "op_type",
            "runtime_writeable_weights",
            "mem_mode",
        ],
        "bitwidth_cols": ["inputDataType", "weightDataType", "outputDataType"],
        "corr_threshold": None,
        "xgb_params": dict(
            n_estimators=350,
            max_depth=7,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
        ),
    },
    "ConvolutionInputGenerator": {
        "raw_filename": "exhaustive_ConvolutionInputGenerator_area_attrs.csv",
        "ohe_cols": ["ram_style"],
        "extra_drops": [
            "cycles_estimate",
            "estimated_cycles",
            "depth_trigger_bram",
            "depth_trigger_uram",
            "op_type",
            "runtime_writeable_weights",
            "parallel_window",
            "noActivation",
            "accDataType",
            "outputDataType",
            "weightDataType",
            "backend",
            "resType",
        ],
        "bitwidth_cols": ["inputDataType"],
        "corr_threshold": 0.98,
        "xgb_params": dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        ),
    },
    "FMPadding": {
        "raw_filename": "exhaustive_FMPadding_area_attrs.csv",
        "ohe_cols": [],
        "extra_drops": [
            "cycles_estimate",
            "estimated_cycles",
            "ram_style",
            "resType",
            "op_type",
            "runtime_writeable_weights",
            "parallel_window",
            "binaryXnorMode",
            "noActivation",
            "numSteps",
            "depthwise",
            "is1D",
            "backend",
            "mem_mode",
            "accDataType",
            "inputDataType",
            "outputDataType",
            "weightDataType",
        ],
        "bitwidth_cols": [],
        "corr_threshold": None,
        "xgb_params": dict(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    },
    "Thresholding": {
        "raw_filename": "exhaustive_Thresholding_area_attrs.csv",
        "ohe_cols": [],
        "extra_drops": [
            "cycles_estimate",
            "estimated_cycles",
            "ram_style",
            "resType",
            "op_type",
            "runtime_writeable_weights",
            "parallel_window",
            "binaryXnorMode",
            "depthwise",
            "is1D",
            "backend",
            "mem_mode",
            "ConvKernelDim",
            "Dilation",
            "IFMChannels",
            "IFMDim",
            "ImgDim",
            "OFMDim",
            "Stride",
            "Padding",
            "accDataType",
            "inputDataType",
            "outputDataType",
            "weightDataType",
        ],
        "bitwidth_cols": ["accDataType"],
        "corr_threshold": None,
        "xgb_params": dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
        ),
    },
    "LabelSelect": {
        "raw_filename": "exhaustive_LabelSelect_area_attrs.csv",
        "ohe_cols": [],
        "extra_drops": [
            "cycles_estimate",
            "estimated_cycles",
            "ram_style",
            "resType",
            "op_type",
            "runtime_writeable_weights",
            "parallel_window",
            "binaryXnorMode",
            "depthwise",
            "is1D",
            "noActivation",
            "backend",
            "mem_mode",
            "ConvKernelDim",
            "Dilation",
            "IFMChannels",
            "IFMDim",
            "ImgDim",
            "OFMDim",
            "Stride",
            "Padding",
            "accDataType",
            "inputDataType",
            "outputDataType",
            "weightDataType",
        ],
        "bitwidth_cols": ["accDataType", "outputDataType"],
        "corr_threshold": None,
        "xgb_params": dict(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
        ),
    },
    "StreamingDataWidthConverter": {
        "raw_filename": "exhaustive_StreamingDataWidthConverter_area_attrs.csv",
        "ohe_cols": [],
        "extra_drops": [
            "cycles_estimate",
            "estimated_cycles",
            "ram_style",
            "resType",
            "op_type",
            "runtime_writeable_weights",
            "parallel_window",
            "binaryXnorMode",
            "depthwise",
            "is1D",
            "noActivation",
            "numSteps",
            "backend",
            "mem_mode",
            "ConvKernelDim",
            "Dilation",
            "IFMChannels",
            "IFMDim",
            "ImgDim",
            "OFMDim",
            "Stride",
            "Padding",
            "accDataType",
            "weightDataType",
        ],
        "bitwidth_cols": ["inputDataType", "outputDataType"],
        "corr_threshold": None,
        "xgb_params": dict(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
        ),
    },
}


def extract_bitwidth(x):
    x_str = str(x).upper()
    if any(k in x_str for k in ["BINARY", "BIPOLAR", "B'BINARY", "INT1", "UINT1"]):
        return 1
    match = re.search(r"(UINT|INT)(\d+)", x_str)
    return int(match.group(2)) if match else None


def clean_listlike(val):
    try:
        if isinstance(val, str) and val.startswith("["):
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
                return float(np.prod(parsed))
    except Exception:
        pass
    return val


def merge_fifo_depths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not os.path.exists(FIFO_DEPTH_CSV):
        df["depth"] = pd.to_numeric(df.get("depth", 2), errors="coerce").fillna(2)
        return df

    df_depths = pd.read_csv(FIFO_DEPTH_CSV)
    keys = ["model_id", "session", "run_name"]
    expected_left = keys + ["Submodule Instance"]
    expected_right = keys + ["fifo_name", "depth"]

    if any(c not in df.columns for c in expected_left) or any(c not in df_depths.columns for c in expected_right):
        df["depth"] = pd.to_numeric(df.get("depth", 2), errors="coerce").fillna(2)
        return df

    df_depths = df_depths[expected_right].copy()
    if "depth" in df.columns:
        df = df.drop(columns=["depth"])

    df = df.merge(
        df_depths,
        how="left",
        left_on=keys + ["Submodule Instance"],
        right_on=keys + ["fifo_name"],
    )
    df.drop(columns=["fifo_name"], inplace=True, errors="ignore")
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce").fillna(2)
    return df


def prepare_module_df(df: pd.DataFrame, cfg: dict, module_name: str) -> pd.DataFrame:
    df = df.copy()

    if "MVAU_" in module_name:
        target_topo = module_name.replace("MVAU_", "")
        initial_count = len(df)
        df = df[df["model_id"] == target_topo]
        print(
            f"  [Filtro] {module_name}: {len(df)} amostras de {initial_count} retidas "
            f"(Model: {target_topo})."
        )
        if len(df) == 0:
            print(f"  [!] AVISO: Nenhuma amostra encontrada para {target_topo}!")
            return df

    df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns], inplace=True, errors="ignore")

    for col in cfg.get("bitwidth_cols", []):
        if col in df.columns:
            new_col = f"{col} (bits)"
            df.insert(df.columns.get_loc(col) + 1, new_col, df[col].apply(extract_bitwidth))
            df.drop(columns=[col], inplace=True, errors="ignore")

    for cat_col in cfg.get("ohe_cols", []):
        if cat_col not in df.columns:
            continue
        dummies = pd.get_dummies(df[cat_col], prefix=cat_col, prefix_sep="_", dtype=int)
        rename_map = {}
        for dummy_col in dummies.columns:
            value = dummy_col.split(f"{cat_col}_", 1)[1]
            clean = "".join(e for e in str(value) if e.isalnum()).capitalize()
            cat_pascal = cat_col[0].upper() + cat_col[1:]
            rename_map[dummy_col] = f"is{cat_pascal}{clean}"
        dummies.rename(columns=rename_map, inplace=True)

        orig_idx = df.columns.get_loc(cat_col)
        df.drop(columns=[cat_col], inplace=True)
        df = pd.concat([df.iloc[:, :orig_idx], dummies, df.iloc[:, orig_idx:]], axis=1)

    list_cols = [
        c
        for c in df.columns
        if df[c].dtype == object
        and df[c].dropna().apply(lambda v: isinstance(v, str) and v.startswith("[")).any()
    ]
    for col in list_cols:
        df[col] = df[col].apply(clean_listlike)

    if "MVAU" in module_name:
        in_bits = pd.to_numeric(df.get("inputDataType (bits)", 1), errors="coerce").fillna(1)
        w_bits = pd.to_numeric(df.get("weightDataType (bits)", 1), errors="coerce").fillna(1)
        pe = pd.to_numeric(df.get("PE", 1), errors="coerce").fillna(1)
        simd = pd.to_numeric(df.get("SIMD", 1), errors="coerce").fillna(1)
        df["mac_complexity"] = in_bits * w_bits * pe * simd

    for target in TARGET_COLS:
        if target in df.columns:
            df[target] = np.log1p(pd.to_numeric(df[target], errors="coerce").fillna(0))

    if not SKIP_EXTRA_DROPS:
        drops = [c for c in cfg.get("extra_drops", []) if c in df.columns]
        df.drop(columns=drops, inplace=True, errors="ignore")

    protected = [c for c in TARGET_COLS if c in df.columns]
    for col in df.columns:
        if col not in protected:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    low_var = [c for c in df.nunique()[df.nunique() <= 1].index if c not in TARGET_COLS]
    protected_features = [
        "inWidth",
        "depth",
        "bit_capacity",
        "mac_complexity",
        "weightDataType (bits)",
        "inputDataType (bits)",
    ]
    low_var = [c for c in low_var if c not in protected_features]
    df.drop(columns=low_var, inplace=True, errors="ignore")

    threshold = cfg.get("corr_threshold")
    if threshold is not None:
        numeric_df = df.select_dtypes(include=np.number)
        numeric_df_no_target = numeric_df.drop(
            columns=[c for c in TARGET_COLS if c in numeric_df.columns],
            errors="ignore",
        )
        if numeric_df_no_target.shape[1] > 1:
            corr = numeric_df_no_target.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_corr = [
                col
                for col in upper.columns
                if any(upper[col] > threshold) and col not in TARGET_COLS
            ]
            high_corr = [c for c in high_corr if c not in protected_features]
            if high_corr:
                print(f"  Removendo correlacionadas: {high_corr}")
            df.drop(columns=high_corr, inplace=True, errors="ignore")

    return df


def prepare_xy(df: pd.DataFrame):
    missing = [t for t in TARGET_COLS if t not in df.columns]
    if missing:
        raise ValueError(f"Colunas target ausentes: {missing}")
    y = df[TARGET_COLS].values.astype(np.float32)
    feature_df = df.drop(columns=[c for c in TARGET_COLS if c in df.columns])
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = feature_df.values.astype(np.float32)
    return X, y, list(feature_df.columns)


def train_fifo_classifier(df_raw: pd.DataFrame):
    print(f"\n{'=' * 65}")
    print("Treinando Decision Tree Classifier para StreamingFIFO (BRAM vs LUT)...")

    df = merge_fifo_depths(df_raw)
    df["inWidth"] = pd.to_numeric(df.get("inWidth", 8), errors="coerce").fillna(8)
    df["depth"] = pd.to_numeric(df.get("depth", 2), errors="coerce").fillna(2)

    X = df[["inWidth", "depth"]].copy()
    y = (pd.to_numeric(df["BRAM (36k eq.)"], errors="coerce").fillna(0) > 0).astype(int)

    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)

    acc = clf.score(X, y)
    print(f"  [✓] Classificador treinado. Acurácia na base inteira: {acc * 100:.2f}%")

    model_path = os.path.join(OUTPUT_DIR, "StreamingFIFO_Classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"  [✓] Classificador salvo em: {model_path}")


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names, module_name, xgb_params):
    weights = np.ones(len(y_train), dtype=np.float32)
    if "MVAU" in module_name:
        weights = np.where(np.expm1(y_train[:, 0]) > 5000, 15.0, 1.0).astype(np.float32)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_records = []
    print(f"  K-Fold CV ({N_SPLITS} folds) com pesos dinâmicos...")

    for fold_idx, (tr_idx, val_idx) in tqdm(
        enumerate(kf.split(X_train)),
        total=N_SPLITS,
        desc="  Treinando folds",
        leave=False,
    ):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        w_tr = weights[tr_idx]

        model = MultiOutputRegressor(XGBRegressor(**xgb_params))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr, sample_weight=w_tr)

        y_pred = model.predict(X_val)
        row = {"fold": fold_idx + 1}
        for i, target in enumerate(TARGET_COLS):
            safe = target.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
            row[f"R2_{safe}"] = r2_score(y_val[:, i], y_pred[:, i])
            row[f"RMSE_{safe}"] = float(np.sqrt(mean_squared_error(y_val[:, i], y_pred[:, i])))
        fold_records.append(row)

    final_model = MultiOutputRegressor(XGBRegressor(**xgb_params))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X_train, y_train, sample_weight=weights)

    y_test_pred = final_model.predict(X_test)
    test_metrics = {}
    print(f"\n  *** TEST SET CEGO ({module_name}) ***")
    for i, target in enumerate(TARGET_COLS):
        safe = target.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        r2 = r2_score(y_test[:, i], y_test_pred[:, i])
        rmse = float(np.sqrt(mean_squared_error(y_test[:, i], y_test_pred[:, i])))
        test_metrics[f"TEST_R2_{safe}"] = r2
        test_metrics[f"TEST_RMSE_{safe}"] = rmse
        print(f"    {target:20s}: TEST_R²={r2:.3f}  TEST_RMSE={rmse:.4f}")

    fi_rows = []
    for i, estimator in enumerate(final_model.estimators_):
        target = TARGET_COLS[i]
        for fn, imp in zip(feature_names, estimator.feature_importances_):
            fi_rows.append({"target": target, "feature": fn, "importance": imp})

    return final_model, pd.DataFrame(fold_records), pd.DataFrame(fi_rows), test_metrics


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.isfile(STREAMING_FIFO_RAW_PATH):
        try:
            fifo_df = pd.read_csv(STREAMING_FIFO_RAW_PATH)
            train_fifo_classifier(fifo_df)
        except Exception as e:
            print(f"[!] Falha ao treinar o classificador de FIFO: {e}")
    else:
        print(f"[!] Dataset de StreamingFIFO não encontrado: {STREAMING_FIFO_RAW_PATH}")

    if not os.path.exists(SPLIT_FIFO_MODEL_PATH):
        print(
            "[!] Aviso: SplitFIFO_area_model.pkl ainda não existe. "
            "Treine-o com train_split_fifo_area.py antes da validação final."
        )

    all_summary = []
    for module_name, cfg in MODULE_CONFIGS.items():
        raw_path = os.path.join(RAW_SPLITTED_DIR, cfg["raw_filename"])
        if not os.path.isfile(raw_path):
            print(f"[!] Dataset ausente para {module_name}: {raw_path}")
            continue

        print(f"\n{'=' * 65}\nMódulo: {module_name}")
        df_raw = pd.read_csv(raw_path)

        try:
            df_clean = prepare_module_df(df_raw, cfg, module_name)
        except Exception as e:
            print(f"  [!] Erro no pré-processamento: {e}")
            continue

        X, y, feature_names = prepare_xy(df_clean)
        if len(X) < 30:
            print(f"  [!] Ignorando {module_name}: dataset pequeno após filtro ({len(X)} amostras)")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.15,
            random_state=42,
        )

        model, df_cv, df_feat, test_metrics = train_and_evaluate(
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
            module_name,
            cfg["xgb_params"],
        )

        model_path = os.path.join(OUTPUT_DIR, f"{module_name}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "feature_names": feature_names,
                    "target_cols": TARGET_COLS,
                    "target_transform": "log1p",
                },
                f,
            )

        df_cv.to_csv(os.path.join(OUTPUT_DIR, f"{module_name}_cv_results.csv"), index=False)
        df_feat_pivot = df_feat.pivot_table(index="feature", columns="target", values="importance")
        df_feat_pivot.to_csv(os.path.join(OUTPUT_DIR, f"{module_name}_feature_importance.csv"))

        row = {"module": module_name, "n_features": X.shape[1], "model_path": model_path}
        row.update(test_metrics)
        all_summary.append(row)

    if all_summary:
        out_csv = os.path.join(OUTPUT_DIR, "training_summary.csv")
        pd.DataFrame(all_summary).to_csv(out_csv, index=False)
        print(f"\n[✓] Treinamento concluído. Resumo salvo em: {out_csv}")
    else:
        print("\n[!] Nenhum especialista foi treinado.")


if __name__ == "__main__":
    main()