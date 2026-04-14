"""
predict_total_fifo_area.py
Simula a lógica de particionamento do FINN (C++) para uma StreamingFIFO
e utiliza o modelo XGBoost de Fatias para prever a área total acumulada.
"""

import os
import math
import pickle
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "retrieval", "results", "trained_models", "SplitFIFO_area_model.pkl")

TARGET_COLS = ["Logic LUTs", "LUTRAMs", "SRLs", "Total FFs", "RAMB36", "RAMB18", "DSP Blocks"]

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"[!] Modelo não encontrado em {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["feature_names"]

def finn_partition_fifo(total_depth: int, in_width: int):
    """
    Simula a lógica do FINN para fatiar FIFOs gigantes:
    - Extrai as maiores potências de 2 possíveis (>= 512) como BRAMs (block/vivado).
    - O resto (ou profundidades pequenas) vai para SRLs/LUTRAMs (distributed/rtl).
    """
    slices = []
    remaining_depth = total_depth
    
    # 1. Fatiar em potências de 2 maiores que o limiar de BRAM (ex: 512)
    while remaining_depth >= 512:
        # Encontra a maior potência de 2 que cabe no remaining_depth
        power = int(math.log2(remaining_depth))
        chunk_depth = int(math.pow(2, power))
        
        slices.append({
            "depth": chunk_depth,
            "ram_style": "block",
            "impl_style": "vivado"
        })
        remaining_depth = int(remaining_depth - chunk_depth)

    # 2. O resto (o "troco") vai para Registradores de Deslocamento (SRLs)
    if remaining_depth > 0:
        slices.append({
            "depth": remaining_depth,
            "ram_style": "distributed",
            "impl_style": "rtl"
        })
        
    return slices

def predict_slice_area(slice_dict: dict, in_width: int, data_type_bits: int, simd: int, model, feature_names: list[str]) -> np.ndarray:
    """
    Monta as features (mesma ordem do treino) e faz a inferência para 1 fatia.
    """
    depth = slice_dict["depth"]
    ram_style = slice_dict["ram_style"]
    impl_style = slice_dict["impl_style"]
    
    bit_capacity = depth * in_width
    srl_complexity = bit_capacity / 32.0
    
    # Dicionário de features base
    feat_dict = {
        "is_ram_style_auto": 0,
        "is_ram_style_block": 1 if ram_style == "block" else 0,
        "is_ram_style_distributed": 1 if ram_style == "distributed" else 0,
        "is_impl_style_rtl": 1 if impl_style == "rtl" else 0,
        "is_impl_style_vivado": 1 if impl_style == "vivado" else 0,
        "log_inWidth": np.log1p(in_width),
        "log_depth": np.log1p(depth),
        "log_bit_capacity": np.log1p(bit_capacity),
        "log_srl_complexity": np.log1p(srl_complexity),
        "dataType_bits": data_type_bits,
        "simd": simd
    }
    
    # Garantir a mesma ordem do treino
    X_input = np.array([[feat_dict.get(f, 0) for f in feature_names]])
    
    preds = model.predict(X_input)[0]
    preds = np.maximum(0, preds)
    
    # Arredondar BRAMs/DSPs para metades, resto para inteiros
    for i, col in enumerate(TARGET_COLS):
        if "RAMB" in col or "DSP" in col:
            preds[i] = np.round(preds[i] * 2) / 2
        else:
            preds[i] = np.round(preds[i])
            
    return preds

def simulate_total_fifo(total_depth, in_width, data_type_bits, simd, model, feature_names):
    print(f"\n⚙️ Simulando FIFO: Total Depth = {total_depth} | inWidth = {in_width}")
    
    # 1. Obter as fatias
    slices = finn_partition_fifo(total_depth, in_width)
    
    print("-" * 50)
    print(f"  🔪 O FINN vai fatiar esta FIFO em {len(slices)} submódulos:")
    
    total_area = np.zeros(len(TARGET_COLS))
    
    for i, s in enumerate(slices):
        preds = predict_slice_area(s, in_width, data_type_bits, simd, model, feature_names)
        total_area += preds
        print(f"     -> Fatia {i}: Depth {s['depth']:<5} | {s['ram_style']:<12} | BRAMs: {(preds[4]+preds[5]):>4.1f} | LUTs: {preds[0]:>5.0f} | SRLs: {preds[2]:>4.0f}")
        
    print("-" * 50)
    print("  📊 ÁREA FÍSICA TOTAL ACUMULADA (Previsão do XGBoost):")
    for col, val in zip(TARGET_COLS, total_area):
        if val > 0:
            print(f"     ✔️ {col:<12}: {val}")

def main():
    try:
        model, feature_names = load_model()
        print("[✓] Modelo Especialista de FIFOs carregado com sucesso!")
    except Exception as e:
        print(e)
        return

    # =========================================================================
    # TESTES DE MESA (Simulações)
    # =========================================================================
    
    # Teste 1: Uma FIFO pequena (Tipicamente apenas 1 fatia de SRL)
    simulate_total_fifo(total_depth=2, in_width=8, data_type_bits=8, simd=1, 
                        model=model, feature_names=feature_names)
    
    # Teste 2: Uma FIFO média (Uma fatia de BRAM e um restinho de SRL)
    simulate_total_fifo(total_depth=600, in_width=32, data_type_bits=8, simd=4, 
                        model=model, feature_names=feature_names)

    # Teste 3: Uma FIFO gigante (Múltiplas BRAMs cascateadas + SRL no final)
    simulate_total_fifo(total_depth=5000, in_width=64, data_type_bits=8, simd=8, 
                        model=model, feature_names=feature_names)

if __name__ == "__main__":
    main()