import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from datetime import datetime

# Importa as ferramentas e configurações necessárias
from utils.hw_utils import utils, get_finn_ready_model
from config import BUILD_CONFIG

def _run_estimate_build(base_build_dir, onnx_model_path, hw_name, topology_id, quant, fpga_part, folding_path=None, target_fps=None):
    """
    Executa um build de estimativa para um dado modelo e configuração.
    """
    print(f"--- Executando Estimativa para: {hw_name} na FPGA {fpga_part} ---")
    build_output_dir = os.path.join(base_build_dir, hw_name)
    steps = BUILD_CONFIG['first_run_estimate']['steps']
    
    args = [
        "python3", "./hara/run_build.py",
        "--model_path", str(onnx_model_path),
        "--build_dir", str(base_build_dir),
        "--topology", str(topology_id),
        "--quant", str(quant),
        "--steps", json.dumps(steps),
        "--hw_name", hw_name,
        "--fpga-part", fpga_part,
        "--folding_file", str(folding_path) if folding_path else "",
        "--target_fps", str(target_fps) if target_fps else "None",
    ]
    log_path = os.path.join(base_build_dir, f"build_{hw_name}.log")
    try:
        utils.run_and_capture(args, log_path=log_path)
        print(f"[✓] Estimativa para {hw_name} concluída.")
        return build_output_dir
    except RuntimeError:
        print(f"[✗] Estimativa para {hw_name} falhou.")
        return None

def generate_map(model_info, base_build_dir, fpga_part):
    topology_id = model_info.get("topology_id")
    quant = model_info.get("quant")
    try:
        master_onnx_path = get_finn_ready_model(model_info, base_build_dir)
        if not master_onnx_path:
            raise RuntimeError("Falha ao preparar o modelo ONNX inicial.")
    except Exception as e:
        print(f"[✗] ERRO: {e}"); return

    print("\n--- Estabelecendo baseline (Full Folded) ---")
    est_dir_1 = _run_estimate_build(base_build_dir, master_onnx_path, "run0_get_initial_fold", topology_id, quant, fpga_part, target_fps=1)
    
    print(f"Diretório da build inicial: {est_dir_1}")
    if not est_dir_1: return

    initial_folding = utils.read_folding_config(est_dir_1)
    intermediate_onnx_path = os.path.join(est_dir_1, "intermediate_models", "step_generate_estimate_reports.onnx")
    if not os.path.exists(intermediate_onnx_path):
        print("[✗] ONNX intermediário para reset não encontrado.")
        return
    reset_folding = utils.reset_folding(initial_folding, intermediate_onnx_path)
    
    baseline_hw_name = "run1_baseline_folded"
    folding_path = os.path.join(base_build_dir, f"{baseline_hw_name}.json")
    with open(folding_path, 'w') as f: json.dump(reset_folding, f, indent=2)
    
    baseline_dir = _run_estimate_build(base_build_dir, master_onnx_path, baseline_hw_name, topology_id, quant, fpga_part, folding_path=folding_path)
    if not baseline_dir: return

    results = []
    perf_path = os.path.join(baseline_dir, "report", "estimate_network_performance.json")
    with open(perf_path, 'r') as f:
        fps = json.load(f).get("estimated_throughput_fps")
    results.append({"run_id": 1, "estimated_fps": fps, "folding_config": json.dumps(reset_folding)})
    print(f"-> Ponto #1: {fps:.2f} FPS (Full Folded)")

    print("\n--- Iniciando Loop de Exploração de Performance ---")
    current_folding = reset_folding
    last_build_dir = baseline_dir
    run_counter = 2

    while True:
        print(f"\n--- Tentativa de Otimização #{run_counter} ---")
        onnx_path_loop = os.path.join(last_build_dir, "intermediate_models/step_generate_estimate_reports.onnx")
        cycles_path = os.path.join(last_build_dir, "report/estimate_layer_cycles.json")
        
        if not os.path.exists(onnx_path_loop) or not os.path.exists(cycles_path):
            print("[!] Arquivos da última execução não encontrados. Encerrando loop.")
            break
        
        with open(cycles_path, 'r') as f: estimate_layer_cycles = json.load(f)
        new_folding = utils.modify_folding(current_folding, onnx_path_loop, estimate_layer_cycles)
        
        if new_folding == current_folding:
            print("[✓] Design estável alcançado (Full Unfolded). Fim da exploração.")
            break
        
        current_folding = new_folding
        current_hw_name = f"run{run_counter}_optimized"
        folding_path = os.path.join(base_build_dir, f"{current_hw_name}.json")
        with open(folding_path, 'w') as f: json.dump(current_folding, f, indent=2)
        
        current_build_dir = _run_estimate_build(base_build_dir, master_onnx_path, current_hw_name, topology_id, quant, fpga_part, folding_path=folding_path)
        if not current_build_dir:
            print("[✗] Falha na estimativa. Encerrando loop.")
            break
        
        last_build_dir = current_build_dir
        perf_path = os.path.join(current_build_dir, "report", "estimate_network_performance.json")
        with open(perf_path, 'r') as f:
            fps = json.load(f).get("estimated_throughput_fps")
        results.append({"run_id": run_counter, "estimated_fps": fps, "folding_config": json.dumps(current_folding)})
        print(f"-> Ponto #{run_counter}: {fps:.2f} FPS")
        run_counter += 1

    if not results:
        print("Nenhum resultado foi gerado.")
        return

    df = pd.DataFrame(results)
    csv_path = os.path.join(base_build_dir, "fps_map.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nMapa de performance salvo em: {csv_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(df["run_id"], df["estimated_fps"], marker='o', linestyle='-')
    plt.title(f"Curva de Performance Teórica para {topology_id} na {fpga_part}")
    plt.xlabel("Passo de Otimização (Aumento de Paralelismo)")
    plt.ylabel("FPS Estimado")
    plt.grid(True)
    plt.xticks(df["run_id"])
    plot_path = os.path.join(base_build_dir, "fps_map.png")
    plt.savefig(plot_path)
    print(f"Gráfico de performance salvo em: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera um mapa de FPS teóricos para um dado modelo."
    )
    
    parser.add_argument('--build_dir', type=str, required=True)
        
    args = parser.parse_args()

    request_file_path = os.path.join(args.build_dir, "request.json")
    if not os.path.exists(request_file_path):
        raise FileNotFoundError(f"Arquivo de requisição não encontrado em: {request_file_path}")

    with open(request_file_path, 'r') as f:
        request_data = json.load(f)
    
    model_id = request_data.get('model_id')
    
    # --- NOVO: Lendo fpga_part diretamente do request.json ---
    # Adicione um valor padrão caso não esteja no JSON
    fpga_part = request_data.get('fpga_part', 'xc7z020clg400-1') 
    
    if not fpga_part:
        raise ValueError("fpga_part não encontrado no request.json.")


    try:
        with open('./hara/models/registry_models.yaml', 'r') as f:
            model_registry = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Arquivo 'hara/models/registry_models.yaml' não encontrado.")

    model_info = model_registry.get(model_id)

    if not model_info:
        raise ValueError(f"Modelo '{model_id}' não encontrado no registro.")
        
    print(f"Diretório da análise: {args.build_dir}")
    print(f"FPGA Part para a análise: {fpga_part}") # Log para confirmar que está pegando
    
    # --- Passa fpga_part lido do JSON para generate_map ---
    generate_map(model_info, args.build_dir, fpga_part)