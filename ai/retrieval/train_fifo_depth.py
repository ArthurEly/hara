"""
train_fifo_depths.py
Treina o "Oráculo de Depths" para as StreamingFIFOs do FINN, 
com balanceamento dinâmico (depth > 2) e K-Fold Cross Validation no treino.
"""

import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "results", "fifo_depth", "fifo_backpressure_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "results", "trained_models")

NUMERIC_FEATURES = [
    "dataType_bits", "tensor_volume", 
    "produtor_PE", "produtor_cycles", "p_throughput", "p_transfers",
    "consumidor_SIMD", "consumidor_cycles", "c_throughput", "c_transfers",
    "parallelism_mismatch", "cycle_ratio", 
    "theoretical_accumulation", "theoretical_fifo_depth"
]

CATEGORICAL_FEATURES = ["produtor_op", "consumidor_op", "ram_style", "impl_style"]

TARGET = "real_depth"

def main():
    print(f"[{'='*75}]")
    print("[HARA] Treinamento do Oráculo de Depths (Com Balanceamento e K-Fold)")
    print(f"[{'='*75}]")

    if not os.path.exists(DATASET_PATH):
        print(f"[!] Erro: Dataset não encontrado em {DATASET_PATH}")
        return

    # 1. Carregar e Limpar Dados
    df = pd.read_csv(DATASET_PATH)
    print(f"[*] Dataset carregado: {len(df)} FIFOs mapeadas.")

    df = df.dropna(subset=[TARGET])
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    # 2. Transformação Logarítmica
    LOG_COLS = ["tensor_volume", "produtor_cycles", "consumidor_cycles", 
                "p_transfers", "c_transfers", "theoretical_accumulation", 
                "theoretical_fifo_depth"]
    
    for col in LOG_COLS:
        df[f"log_{col}"] = np.log1p(df[col])
        if col in NUMERIC_FEATURES:
            NUMERIC_FEATURES.remove(col)
        NUMERIC_FEATURES.append(f"log_{col}")

    y_log = np.log1p(df[TARGET]).values # .values para evitar problemas de índice no K-Fold

    # 3. One-Hot Encoding
    df_features = df[NUMERIC_FEATURES].copy()
    for cat_col in CATEGORICAL_FEATURES:
        dummies = pd.get_dummies(df[cat_col], prefix=cat_col, dtype=int)
        df_features = pd.concat([df_features, dummies], axis=1)

    X = df_features.values
    feature_names = list(df_features.columns)

    # 4. Divisão Treino / Teste (Holdout Set)
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.15, random_state=42)
    print(f"[*] Preparados {len(X_train)} exemplos de treino e {len(X_test)} de teste cego.")

    # =========================================================================
    # K-FOLD CROSS VALIDATION NO CONJUNTO DE TREINO
    # =========================================================================
    print(f"\n[K-Fold] Executando Validação Cruzada (5 Folds) no Treino...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_r2, fold_mae, fold_rmse = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_kf_train, X_kf_val = X_train[train_idx], X_train[val_idx]
        y_kf_train, y_kf_val = y_train[train_idx], y_train[val_idx]

        # Calculo do peso dinâmico APENAS para o subset de treino deste fold
        y_kf_train_real = np.round(np.expm1(y_kf_train))
        minority_mask = y_kf_train_real > 2
        weight_factor = (~minority_mask).sum() / max(1, minority_mask.sum())
        sample_weights = np.where(minority_mask, weight_factor, 1.0)

        model_kf = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6, 
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        model_kf.fit(X_kf_train, y_kf_train, sample_weight=sample_weights)

        # Avaliação do Fold
        preds_kf_real = np.round(np.expm1(model_kf.predict(X_kf_val))).astype(int)
        y_kf_val_real = np.round(np.expm1(y_kf_val)).astype(int)

        r2 = r2_score(y_kf_val_real, preds_kf_real)
        mae = mean_absolute_error(y_kf_val_real, preds_kf_real)
        rmse = np.sqrt(mean_squared_error(y_kf_val_real, preds_kf_real))

        fold_r2.append(r2)
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        print(f"  - Fold {fold+1}: R² = {r2:.4f} | MAE = {mae:>5.1f} | RMSE = {rmse:>5.1f}")

    print(f"\n[K-Fold] 📊 Resultados Médios de Validação (Robustez):")
    print(f"  - R² Médio   : {np.mean(fold_r2):.4f} (± {np.std(fold_r2):.4f})")
    print(f"  - MAE Médio  : {np.mean(fold_mae):.1f} elementos")
    print(f"  - RMSE Médio : {np.mean(fold_rmse):.1f} elementos")

    # =========================================================================
    # TREINAMENTO FINAL NO CONJUNTO DE TREINO COMPLETO (E TESTE CEGO)
    # =========================================================================
    print(f"\n[XGBoost] A treinar o Modelo Final com todo o conjunto de Treino...")
    y_train_real = np.round(np.expm1(y_train))
    minority_mask = y_train_real > 2
    weight_factor = (~minority_mask).sum() / max(1, minority_mask.sum())
    final_weights = np.where(minority_mask, weight_factor, 1.0)

    final_model = XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=6, 
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    final_model.fit(X_train, y_train, sample_weight=final_weights)

    # Avaliação no Test Set "Cego" (O Holdout de 15%)
    preds_real = np.round(np.expm1(final_model.predict(X_test))).astype(int)
    y_test_real = np.round(np.expm1(y_test)).astype(int)

    final_r2 = r2_score(y_test_real, preds_real)
    final_mae = mean_absolute_error(y_test_real, preds_real)
    final_rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))

    print(f"\n{'='*50}")
    print(f"🏆 RESULTADOS FINAIS NO TEST SET (HOLDOUT)")
    print(f"{'='*50}")
    print(f"  - R² Score : {final_r2:.4f}")
    print(f"  - MAE      : {final_mae:.1f} elementos")
    print(f"  - RMSE     : {final_rmse:.1f} elementos")
    
    print(f"\n👀 Exemplos Reais vs Preditos (Test Set):")
    df_test_examples = pd.DataFrame({"Real Depth": y_test_real, "Pred Depth": preds_real})
    df_test_examples = df_test_examples.sort_values(by="Real Depth", ascending=False)
    print("--- Maiores FIFOs ---")
    print(df_test_examples.head(3).to_string(index=False))

    # Salvar Modelo
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "StreamingFIFO_depth_model.pkl")
    
    model_data = {
        "model": final_model,
        "feature_names": feature_names,
        "target_cols": [TARGET]
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
        
    print(f"\n[✓] Modelo de Depths salvo em: {model_path}")

if __name__ == "__main__":
    main()