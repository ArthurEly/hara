import os
import pickle
import numpy as np

# Copiado de predict_fifo_utils.py para ser standalone
def prepare_fifo_features(depth, in_width, ram_style, impl_style, bits, simd):
    bit_capacity = depth * in_width
    return {
        "is_ram_style_auto": 0,
        "is_ram_style_block": 1 if ram_style == "block" else 0,
        "is_ram_style_distributed": 1 if ram_style == "distributed" else 0,
        "is_impl_style_rtl": 1 if impl_style == "rtl" else 0,
        "is_impl_style_vivado": 1 if impl_style == "vivado" else 0,
        "log_inWidth": np.log1p(in_width),
        "log_depth": np.log1p(depth),
        "log_bit_capacity": np.log1p(bit_capacity),
        "log_srl_complexity": np.log1p(bit_capacity / 32.0),
        "dataType_bits": bits,
        "simd": simd
    }

def main():
    model_path = os.path.join("retrieval", "results", "trained_models", "SplitFIFO_area_model.pkl")
    
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        
    model = model_data["model"]
    feature_names = model_data["feature_names"]
    target_cols = model_data["target_cols"]
    
    print(f"[*] Features esperadas pelo XGBoost ({len(feature_names)}):")
    print(feature_names)
    
    # Simulando a StreamingFIFO_rtl_0
    depth = 784
    in_width = 8   # Suposição comum do MNIST na entrada
    bits = 1
    simd = 8
    
    feat = prepare_fifo_features(depth, in_width, "block", "rtl", bits, simd)
    
    X_input = []
    print("\n[*] Valores injetados no vetor:")
    for f in feature_names:
        val = float(feat.get(f, 0.0))
        X_input.append(val)
        print(f"  - {f}: {val}")
        
    X_arr = np.array(X_input, dtype=np.float32).reshape(1, -1)
    
    preds = model.predict(X_arr)[0]
    preds = np.maximum(0, preds)
    
    print("\n🏆 PREDIÇÃO DIRETA DO MODELO PARA O BLOCO (784 depth | BLOCK | RTL):")
    for col, pred in zip(target_cols, preds):
        if "RAMB" in col or "DSP" in col:
            val = np.round(pred * 2) / 2
        else:
            val = np.round(pred)
        print(f"  -> {col}: {val}")

if __name__ == "__main__":
    main()
