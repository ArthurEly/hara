"""
train_split_fifo_area.py
Treina o modelo preditor de área para fatias individuais de StreamingFIFOs.
Versão 3.0: Apenas controlabilidade explícita (sem ram_style = 'auto').
"""

import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "results", "fifo_area", "split_fifo_area_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "results", "trained_models")

TARGET_COLS = ["Logic LUTs", "LUTRAMs", "SRLs", "Total FFs", "RAMB36", "RAMB18", "DSP Blocks"]

def main():
    print(f"[{'='*75}]")
    print("[HARA] Treinamento de Área: Fatias Individuais de StreamingFIFO (V3)")
    print(f"[{'='*75}]")

    if not os.path.exists(DATASET_PATH):
        print(f"[!] Erro: Dataset não encontrado em {DATASET_PATH}")
        return

    # 1. Carregar Dados
    df = pd.read_csv(DATASET_PATH)
    total_inicial = len(df)

    # 2. Limpeza de Dados
    df = df.dropna(subset=TARGET_COLS)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    
    # --- NOVO FILTRO: Remover 'auto' ---
    # Só queremos instâncias onde ordenamos explicitamente o hardware
    df = df[df["ram_style"].str.lower() != "auto"].copy()
    
    # Remoção de "Fantasmas" (Módulos otimizados a zero pelo Vivado)
    df['total_area'] = df[TARGET_COLS].sum(axis=1)
    df = df[df['total_area'] > 0.0].copy()
    
    print(f"[*] Dataset: {total_inicial} fatias brutas -> {len(df)} fatias controladas (sem 'auto' e sem fantasmas).")

    if len(df) == 0:
        print("[!] Aviso: O dataset ficou vazio após os filtros. Verifique o seu JSON de hardware.")
        return

    # Dividir em duas categorias para especialistas puros
    df_block = df[df["ram_style"].str.lower() == "block"].copy()
    df_dist = df[df["ram_style"].str.lower() == "distributed"].copy()

    def train_specialist(df_sub, model_filename, is_block):
        print(f"\n{'-'*60}")
        print(f" Treinando Especialista: {model_filename} (Amostras: {len(df_sub)})")
        print(f"{'-'*60}")
        
        if len(df_sub) == 0:
            print(f"[!] Dataset vazio para {model_filename}.")
            return
            
        # Engenharia de Features
        df_sub["srl_complexity"] = df_sub["bit_capacity"] / 32.0

        # Para evitar problemas com get_dummies faltando categorias
        df_sub["ram_style_block"] = 1 if is_block else 0
        df_sub["ram_style_distributed"] = 0 if is_block else 1
        df_sub["impl_style_rtl"] = (df_sub["impl_style"].str.lower() == "rtl").astype(int)
        df_sub["impl_style_vivado"] = (df_sub["impl_style"].str.lower() == "vivado").astype(int)
        
        rename_map = {
            "ram_style_block": "is_ram_style_block",
            "ram_style_distributed": "is_ram_style_distributed",
            "impl_style_rtl": "is_impl_style_rtl",
            "impl_style_vivado": "is_impl_style_vivado"
        }
        df_sub.rename(columns=rename_map, inplace=True)

        LOG_FEATURES = ["inWidth", "depth", "bit_capacity", "srl_complexity"]
        for col in LOG_FEATURES:
            df_sub[f"log_{col}"] = np.log1p(df_sub[col])

        feature_cols = [c for c in df_sub.columns if c.startswith("is_") or c.startswith("log_")] + ["dataType_bits", "simd"]
        X = df_sub[feature_cols].values
        feature_names = feature_cols

        y = df_sub[TARGET_COLS].values

        # Pesos Dinâmicos
        has_memory = (df_sub['RAMB18'] > 0) | (df_sub['RAMB36'] > 0) | (df_sub['SRLs'] > 0)
        minority_count = has_memory.sum()
        majority_count = (~has_memory).sum()
        weight_factor = majority_count / max(1, minority_count)
        sample_weights = np.where(has_memory, weight_factor, 1.0)
        
        print(f"[*] Minorias com Memória/SRL: {minority_count} vs {majority_count}")

        X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
            X, y, sample_weights, test_size=0.15, random_state=42
        )

        base_model = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1
        )
        
        multi_model = MultiOutputRegressor(base_model)
        multi_model.fit(X_train, y_train, sample_weight=sw_train)

        preds_real = multi_model.predict(X_test)
        preds_real = np.maximum(0, preds_real) 
        
        for i, col in enumerate(TARGET_COLS):
            if "RAMB" in col or "DSP" in col:
                preds_real[:, i] = np.round(preds_real[:, i] * 2) / 2
            else:
                preds_real[:, i] = np.round(preds_real[:, i])

        print(f"\n🏆 RESULTADOS: {model_filename}")
        for i, target in enumerate(TARGET_COLS):
            if np.var(y_test[:, i]) == 0:
                r2 = 1.0 if np.all(y_test[:, i] == preds_real[:, i]) else 0.0
            else:
                r2 = r2_score(y_test[:, i], preds_real[:, i])
            mae = mean_absolute_error(y_test[:, i], preds_real[:, i])
            print(f" 🔹 {target:<12} | R²: {r2:>6.4f} | MAE: {mae:>6.2f} unidades")

        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, f"{model_filename}.pkl")
        
        model_data = {
            "model": multi_model,
            "feature_names": feature_names,
            "target_cols": TARGET_COLS
        }
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"[✓] Salvo em: {model_path}")

    train_specialist(df_block, "SplitFIFO_block_model", is_block=True)
    train_specialist(df_dist, "SplitFIFO_distributed_model", is_block=False)

if __name__ == "__main__":
    main()