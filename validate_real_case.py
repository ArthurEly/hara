import json
import pandas as pd
from ai.multi_module_learner import MultiModuleLearner

def main():
    print("==================================================")
    print("              HARAv2 ML Predictor Validation")
    print("==================================================")
    
    learner = MultiModuleLearner("ai/retrieval/results/trained_models")
    
    # 1. Definimos o diretório de uma síntese concluída no dataset
    base_dir = "exhaustive_hw_builds/MNIST_1W1A_2026-04-06_09-36-34"
    run_name = "run34_optimized"
    run_dir = f"{base_dir}/{run_name}"
    
    onnx_path = f"{run_dir}/intermediate_models/step_generate_estimate_reports.onnx"
    config_path = f"{run_dir}/final_hw_config.json"
    
    print(f"\n=> Carregando configuração de: {run_name}")
    with open(config_path, "r") as f:
        folding = json.load(f)
        
    print("=> Realizando Inferencia (XGBoost) para toda a rede neural...")
    preds = learner.predict(onnx_path, [folding])
    if not preds:
        print("Falha na previsão.")
        return
        
    pred = preds[0]
    
    print("\n=> Estimativas por Componente Individual:")
    print(f"{'Componente':<40} | {'LUTs':<6} | {'FFs':<6} | {'BRAM':<5} | {'DSP':<4}")
    print("-" * 75)
    for node_name, p in pred.get("_details", {}).items():
        l = int(round(p.get("Total LUT", 0)))
        f = int(round(p.get("Total FFs", 0)))
        b = round(p.get("BRAM (36k eq.)", 0), 1)
        d = int(round(p.get("DSP Blocks", 0)))
        print(f"{node_name:<40} | {l:<6} | {f:<6} | {b:<5} | {d:<4}")
        
    print(f"Total de camadas mapeadas com ML: {pred['_n_layers_covered']}")

    # 2. Carregar valores Reais do Vivado OOC
    print("=> Extraindo síntese real do Vivado...")
    df = pd.read_csv(f"{base_dir}/hardware_summary.csv")
    row = df[df["hw_name"] == run_name].iloc[0]
    
    print("\n" + "-"*55)
    print(f"{'Recurso':<12} | {'ML Predito':<15} | {'Vivado Real':<12} | {'Acurácia'}")
    print("-" * 55)
    
    def print_metric(name, p, r):
        # Acurácia (1 - erro_relativo)
        acc = max(0, 100 - (abs(p - r) / max(r, 1) * 100))
        print(f"{name:<12} | {p:<15} | {r:<12} | {acc:.1f}%")
        
    print_metric("LUTs", pred["Total LUTs"], row["Total LUTs"])
    print_metric("FFs", pred["FFs"], row["FFs"])
    print_metric("BRAMs (36k)", pred["BRAM (36k)"], row["BRAM (36k)"])
    print_metric("DSPs", pred["DSP Blocks"], row["DSP Blocks"])
    print("-" * 55)

    print("\n* OBS: FFs e BRAMs são comumente podados globalmente pelo ")
    print("Vivado, o que diminui a precisão em comparação com a soma analítica.")
    
if __name__ == '__main__':
    main()
