"""
train_xgboost.py  – Modelos Especialistas por Tipo de Módulo HARAv2
===================================================================
Versão otimizada com Feature Engineering (Bit Capacity) para detecção de BRAM.
"""

import os
import re
import ast
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# =============================================================================
# CONFIGURAÇÃO GLOBAL
# =============================================================================

RAW_SPLITTED_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results", "splitted"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results", "trained_models"
)

SKIP_EXTRA_DROPS = False

TARGET_COLS = ["Total LUT", "Total FFs", "BRAM (36k eq.)", "DSP Blocks"]

LEAKAGE_COLS = [
    "model_id", "session", "timestamp", "run_name", "run_number",
    "is_baseline", "fixed_ram_style", "fixed_resType",
    "Submodule Instance", "base_name", "layer_idx",
    "Hardware config",
]

N_SPLITS = 10 

# =============================================================================
# CONFIGURAÇÕES ESPECIALISTAS POR MÓDULO
# =============================================================================

MODULE_CONFIGS = {

    "MVAU": {
        "raw_filename": "exhaustive_MVAU_area_attrs.csv",
        "ohe_cols": ["ram_style", "resType", "op_type", "mem_mode", "binaryXnorMode"],
        "extra_drops": [
            "cycles_estimate", "estimated_cycles",
            "depth_trigger_bram", "depth_trigger_uram",
            "depthwise", "is1D", "parallel_window",
            "runtime_writeable_weights",
            "ConvKernelDim", "Dilation", "IFMChannels", "IFMDim",
            "ImgDim", "OFMDim", "Stride", "Padding",
            "backend",
        ],
        "bitwidth_cols": ["accDataType", "inputDataType", "outputDataType", "weightDataType"],
        "corr_threshold": 0.98,
        "xgb_params": dict(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        ),
    },

    "ConvolutionInputGenerator": {
        "raw_filename": "exhaustive_ConvolutionInputGenerator_area_attrs.csv",
        "ohe_cols": ["ram_style"],
        "extra_drops": [
            "cycles_estimate", "estimated_cycles",
            "depth_trigger_bram", "depth_trigger_uram",
            "op_type", "runtime_writeable_weights",
            "parallel_window", "noActivation",
            "accDataType", "outputDataType", "weightDataType",
            "backend", "resType",
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

    "StreamingFIFO_LUT": {
        "raw_filename": "exhaustive_StreamingFIFO_area_attrs.csv",
        "ohe_cols": ["ram_style", "impl_style"],
        "extra_drops": [
            "cycles_estimate", "estimated_cycles", "op_type", "runtime_writeable_weights",
            "ConvKernelDim", "Dilation", "IFMChannels", "IFMDim", "ImgDim", "OFMDim", 
            "Stride", "Padding", "PE", "SIMD", "binaryXnorMode", "noActivation", "numSteps",
            "parallel_window", "depthwise", "is1D", "backend", "mem_mode", "resType",
            "accDataType", "weightDataType",
        ],
        "bitwidth_cols": ["inputDataType", "outputDataType"],
        "corr_threshold": None,
        "xgb_params": dict(
            n_estimators=400, max_depth=6, learning_rate=0.05, 
            subsample=0.8, colsample_bytree=0.9, random_state=42, n_jobs=-1
        ),
    },

    "StreamingFIFO_BRAM": {
        "raw_filename": "exhaustive_StreamingFIFO_area_attrs.csv",
        "ohe_cols": ["ram_style", "impl_style"],
        "extra_drops": [
            "cycles_estimate", "estimated_cycles", "op_type", "runtime_writeable_weights",
            "ConvKernelDim", "Dilation", "IFMChannels", "IFMDim", "ImgDim", "OFMDim", 
            "Stride", "Padding", "PE", "SIMD", "binaryXnorMode", "noActivation", "numSteps",
            "parallel_window", "depthwise", "is1D", "backend", "mem_mode", "resType",
            "accDataType", "weightDataType",
        ],
        "bitwidth_cols": ["inputDataType", "outputDataType"],
        "corr_threshold": None,
        "xgb_params": dict(
            # Menos estimadores e mais profundidade pois o dataset de BRAM é menor (136 samples)
            n_estimators=200, max_depth=4, learning_rate=0.05, 
            subsample=0.8, colsample_bytree=0.9, random_state=42, n_jobs=-1
        ),
    },

    "FMPadding": {
        "raw_filename": "exhaustive_FMPadding_area_attrs.csv",
        "ohe_cols": [],
        "extra_drops": [
            "cycles_estimate", "estimated_cycles",
            "ram_style", "resType", "op_type",
            "runtime_writeable_weights", "parallel_window",
            "binaryXnorMode", "noActivation", "numSteps",
            "depthwise", "is1D", "backend", "mem_mode",
            "accDataType", "inputDataType", "outputDataType", "weightDataType",
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
            "cycles_estimate", "estimated_cycles",
            "ram_style", "resType", "op_type",
            "runtime_writeable_weights", "parallel_window",
            "binaryXnorMode", "depthwise", "is1D",
            "backend", "mem_mode",
            "ConvKernelDim", "Dilation", "IFMChannels", "IFMDim",
            "ImgDim", "OFMDim", "Stride", "Padding",
            "accDataType", "inputDataType", "outputDataType", "weightDataType",
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
            "cycles_estimate", "estimated_cycles",
            "ram_style", "resType", "op_type",
            "runtime_writeable_weights", "parallel_window",
            "binaryXnorMode", "depthwise", "is1D", "noActivation",
            "backend", "mem_mode",
            "ConvKernelDim", "Dilation", "IFMChannels", "IFMDim",
            "ImgDim", "OFMDim", "Stride", "Padding",
            "accDataType", "inputDataType", "outputDataType", "weightDataType",
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
            "cycles_estimate", "estimated_cycles",
            "ram_style", "resType", "op_type",
            "runtime_writeable_weights", "parallel_window",
            "binaryXnorMode", "depthwise", "is1D", "noActivation",
            "numSteps", "backend", "mem_mode",
            "ConvKernelDim", "Dilation", "IFMChannels", "IFMDim",
            "ImgDim", "OFMDim", "Stride", "Padding",
            "accDataType", "weightDataType",
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

# =============================================================================
# HELPERS
# =============================================================================

def extract_bitwidth(x):
    x_str = str(x).upper()
    if any(k in x_str for k in ["BINARY", "BIPOLAR", "B'BINARY"]):
        return 1
    match = re.search(r'(UINT|INT)(\d+)', x_str)
    return int(match.group(2)) if match else None


def clean_listlike(val):
    try:
        if isinstance(val, str) and val.startswith("["):
            lst = ast.literal_eval(val)
            if isinstance(lst, list) and all(isinstance(x, (int, float)) for x in lst):
                return float(np.prod(lst))
    except Exception:
        pass
    return val


def prepare_module_df(df: pd.DataFrame, cfg: dict, module_name: str) -> pd.DataFrame:
    df = df.copy()

    # --- SEPARAÇÃO DOS UNIVERSOS (Mixture of Experts) ---
    if module_name == "StreamingFIFO_LUT":
        # Treina apenas com FIFOs que NÃO usaram BRAM
        df = df[df["BRAM (36k eq.)"] == 0]
    elif module_name == "StreamingFIFO_BRAM":
        # Treina apenas com FIFOs que USARAM BRAM
        df = df[df["BRAM (36k eq.)"] > 0]

    # --- INSERIR DEPTHS REAIS NAS FIFOS ---
    if module_name == "StreamingFIFO":
        fifo_depth_csv = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "results", "fifo_depth", "exhaustive_fifo_depths.csv"
        )
        if os.path.exists(fifo_depth_csv):
            df_depths = pd.read_csv(fifo_depth_csv)
            keys = ["model_id", "session", "run_name"]
            df_depths = df_depths[keys + ["fifo_name", "depth"]]
            if "depth" in df.columns:
                df = df.drop(columns=["depth"])
            df = df.merge(df_depths, how="left", left_on=keys + ["Submodule Instance"], right_on=keys + ["fifo_name"])
            df.drop(columns=["fifo_name"], inplace=True, errors="ignore")
            df["depth"] = df["depth"].fillna(2)

    # ── 1. Remover leakage global ──────────────────────────────────────────
    df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns], inplace=True, errors='ignore')

    # ── 2. Extrair bitwidths ───────────────────────────────────────────────
    for col in cfg.get("bitwidth_cols", []):
        if col in df.columns:
            new_col = f"{col} (bits)"
            df.insert(df.columns.get_loc(col) + 1, new_col, df[col].apply(extract_bitwidth))
            df.drop(columns=[col], inplace=True, errors='ignore')

    # ── 3. One-hot encoding ────────────────────────────────────────────────
    for cat_col in cfg.get("ohe_cols", []):
        if cat_col not in df.columns: continue
        dummies = pd.get_dummies(df[cat_col], prefix=cat_col, prefix_sep='_', dtype=int)
        rename_map = {}
        for dummy_col in dummies.columns:
            value = dummy_col.split(f"{cat_col}_", 1)[1]
            clean = ''.join(e for e in str(value) if e.isalnum()).capitalize()
            cat_pascal = cat_col[0].upper() + cat_col[1:]
            rename_map[dummy_col] = f"is{cat_pascal}{clean}"
        dummies.rename(columns=rename_map, inplace=True)
        orig_idx = df.columns.get_loc(cat_col)
        df.drop(columns=[cat_col], inplace=True)
        df = pd.concat([df.iloc[:, :orig_idx], dummies, df.iloc[:, orig_idx:]], axis=1)

    # ── 4. Expande list-like ───────────────────────────────────────────────
    list_cols = [c for c in df.columns if df[c].dtype == object and
                 df[c].dropna().apply(lambda v: isinstance(v, str) and v.startswith("[")).any()]
    for col in list_cols:
        df[col] = df[col].apply(clean_listlike)

    # ── 5. FEATURE ENGINEERING & LOG-SCALING (Otimização para generalização) ──
    if module_name == "StreamingFIFO":
        df["inWidth"] = pd.to_numeric(df["inWidth"], errors='coerce').fillna(8)
        df["depth"]   = pd.to_numeric(df["depth"],   errors='coerce').fillna(2)
        df["bit_capacity"] = df["inWidth"] * df["depth"]

        # ← NOVO: flag determinística antes do log-scaling
        BRAM_THRESHOLD = 512
        ram_block = df.get("ram_style", pd.Series(["auto"]*len(df))).str.lower() == "block"
        ram_dist  = df.get("ram_style", pd.Series(["auto"]*len(df))).str.lower() == "distributed"
        auto_bram = df["bit_capacity"] > BRAM_THRESHOLD
        df["is_bram"] = ((ram_block) | (~ram_dist & auto_bram)).astype(int)

        # Log-scaling (só depois de calcular is_bram com valores reais)
        df["inWidth"]      = np.log1p(df["inWidth"])
        df["depth"]        = np.log1p(df["depth"])
        df["bit_capacity"] = np.log1p(df["bit_capacity"])

    # ── 6. TARGET LOG TRANSFORMATION ──────────────────────────────────────
    # Treinar no espaço logarítmico impede que o erro de uma FIFO gigante 
    # destrua a precisão das FIFOs pequenas.
    #for target in TARGET_COLS:
    #    if target in df.columns:
    #        df[target] = np.log1p(df[target])

    # ── 7. Limpezas finais (Remover colunas extras e Converter para numérico) ──
    if not SKIP_EXTRA_DROPS:
        drops = [c for c in cfg.get("extra_drops", []) if c in df.columns]
        df.drop(columns=drops, inplace=True, errors='ignore')

    protected = [c for c in TARGET_COLS if c in df.columns]
    for col in df.columns:
        if col not in protected:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    nunique = df.nunique()
    low_var = [c for c in nunique[nunique <= 1].index if c not in TARGET_COLS]
    
    # NOVO: Proteger features base para que o modelo nunca fique com 0 colunas
    protected_features = ["inWidth", "depth", "bit_capacity"]
    low_var = [c for c in low_var if c not in protected_features]
    
    df.drop(columns=low_var, inplace=True, errors='ignore')

    # ── 8. Remover altamente correlacionadas (Protegendo bit_capacity) ──────
    threshold = cfg.get("corr_threshold")
    if threshold is not None:
        numeric_df = df.select_dtypes(include=np.number)
        numeric_df_no_target = numeric_df.drop(columns=[c for c in TARGET_COLS if c in numeric_df.columns], errors='ignore')
        if numeric_df_no_target.shape[1] > 1:
            corr = numeric_df_no_target.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_corr = [col for col in upper.columns
                         if any(upper[col] > threshold) and col not in TARGET_COLS]
            high_corr = [c for c in high_corr if c != "bit_capacity"]
            df.drop(columns=high_corr, inplace=True, errors='ignore')

    return df


def prepare_xy(df: pd.DataFrame):
    missing = [t for t in TARGET_COLS if t not in df.columns]
    if missing: raise ValueError(f"Colunas target ausentes: {missing}")
    y = df[TARGET_COLS].values.astype(np.float32)
    feature_df = df.drop(columns=[c for c in TARGET_COLS if c in df.columns])
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = feature_df.values.astype(np.float32)
    return X, y, list(feature_df.columns)


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names, module_name, xgb_params):
    # --- SAMPLE WEIGHTING: Compensação de Desbalanceamento ---
    # Dá peso 10x para as amostras que usam BRAM (coluna index 2: BRAM 36k eq.)
    # Como y_train está em log, verificamos se log(1+BRAM) > 0
    weights = np.where(y_train[:, 2] > 0, 10.0, 1.0)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_records = []
    print(f"  K-Fold CV ({N_SPLITS} folds) com Pesos e Log-Reversal…")
    
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        w_tr = weights[tr_idx]

        model = MultiOutputRegressor(XGBRegressor(**xgb_params))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Treina usando os pesos de importância
            model.fit(X_tr, y_tr, sample_weight=w_tr)
        
        # Inverte o Log para calcular métricas na escala real do Hardware
        y_pred_real = model.predict(X_val)
        y_val_real = y_val
        
        row = {"fold": fold_idx + 1}
        for i, target in enumerate(TARGET_COLS):
            safe = target.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
            row[f"R2_{safe}"] = r2_score(y_val_real[:, i], y_pred_real[:, i])
            row[f"RMSE_{safe}"] = float(np.sqrt(mean_squared_error(y_val_real[:, i], y_pred_real[:, i])))
        fold_records.append(row)

    # Treino final no set completo
    final_model = MultiOutputRegressor(XGBRegressor(**xgb_params))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X_train, y_train, sample_weight=weights)

    # Avaliação final no test set (Set Cego)
    y_test_pred_real = final_model.predict(X_test)
    y_test_real = y_test
    
    test_metrics = {}
    print(f"\n  *** TEST SET CEGO ({module_name}) ***")
    for i, target in enumerate(TARGET_COLS):
        safe = target.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        r2 = r2_score(y_test_real[:, i], y_test_pred_real[:, i])
        rmse = float(np.sqrt(mean_squared_error(y_test_real[:, i], y_test_pred_real[:, i])))
        test_metrics[f"TEST_R2_{safe}"] = r2
        test_metrics[f"TEST_RMSE_{safe}"] = rmse
        print(f"    {target:20s}: TEST_R²={r2:.3f}  TEST_RMSE={rmse:.1f}")

    fi_rows = []
    for i, estimator in enumerate(final_model.estimators_):
        target = TARGET_COLS[i]
        for fn, imp in zip(feature_names, estimator.feature_importances_):
            fi_rows.append({"target": target, "feature": fn, "importance": imp})
    
    return final_model, pd.DataFrame(fold_records), pd.DataFrame(fi_rows), test_metrics


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_summary = []
    for module_name, cfg in MODULE_CONFIGS.items():
        raw_path = os.path.join(RAW_SPLITTED_DIR, cfg["raw_filename"])
        if not os.path.isfile(raw_path): continue
        print(f"\n{'='*65}\nMódulo: {module_name}")
        df_raw = pd.read_csv(raw_path)
        try:
            df_clean = prepare_module_df(df_raw, cfg, module_name)
        except Exception as e:
            print(f"  [!] Erro no pre-processamento: {e}"); continue

        X, y, feature_names = prepare_xy(df_clean)
        if len(X) < 30: continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        model, df_cv, df_feat, test_metrics = train_and_evaluate(X_train, y_train, X_test, y_test, feature_names, module_name, cfg["xgb_params"])

        # Salvar modelo e metadados
        with open(os.path.join(OUTPUT_DIR, f"{module_name}_model.pkl"), "wb") as f:
            pickle.dump({"model": model, "feature_names": feature_names, "target_cols": TARGET_COLS}, f)

        df_feat_pivot = df_feat.pivot_table(index="feature", columns="target", values="importance")
        df_feat_pivot.to_csv(os.path.join(OUTPUT_DIR, f"{module_name}_feature_importance.csv"))

        row = {"module": module_name, "n_features": X.shape[1]}
        row.update(test_metrics)
        all_summary.append(row)

    if all_summary:
        pd.DataFrame(all_summary).to_csv(os.path.join(OUTPUT_DIR, "training_summary.csv"), index=False)
        print("\n[✓] Treinamento concluído com Bit Capacity Engineering.")

if __name__ == "__main__":
    main()