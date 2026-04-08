#!/usr/bin/env python3
"""
Gera uma sequência completa de arquivos de dobramento (folding configs)
utilizando puramente o modelo analítico (sem invocar o FINN).

Esses arquivos podem ser usados posteriormente para consultar o HardwareLearner.
"""

import argparse
import json
import os
import copy
import numpy as np

# Supondo que estamos no diretório hara/ ou chamando de dentro dele
from utils.analytic_utils import FinnCycleEstimator
from utils.hw_utils import utils

def modify_folding_analytical(current_folding, analyzer, cycle_formulas):
    """
    Dá um "step" na otimização de dobramento, incrementando PE ou SIMD
    da camada que for o gargalo atual (maior número de ciclos), usando
    as fórmulas do modelo analítico em vez dos relatórios do FINN.
    """
    new_folding = copy.deepcopy(current_folding)
    
    # 1. Calcula o ciclo atual de todas as camadas
    layer_cycles = {}
    for layer_name, data in cycle_formulas.items():
        formula = data['formula']
        cfg = current_folding.get(layer_name, {})
        defaults = current_folding.get("Defaults", {"PE": 1, "SIMD": 1})
        
        pe = cfg.get("PE", defaults.get("PE", 1))
        simd = cfg.get("SIMD", defaults.get("SIMD", 1))
        current_params = {"PE": pe, "SIMD": simd}
        
        # Tratamento especial para parallel_window de CIG
        if "ConvolutionInputGenerator" in data.get("op_type", "") and cfg.get("parallel_window", 0) == 1:
            # Como não temos o ONNX em mãos aqui de forma barata para ler IFMDim, 
            # estimamos via o pior caso ou assumimos a leitura estática
            # (No seu código original, precisa recarregar o onnx. Aqui vamos tentar pegar do dictionary se possível)
            is_parallel = True
            cycles = 2 # se parallel window ativado, é quase instante (pipelined stream)
        else:
            cycles = analyzer._eval_formula(formula, current_params)
        
        layer_cycles[layer_name] = cycles

    if not layer_cycles:
        return new_folding

    # 2. Identifica o(s) gargalo(s)
    sorted_layers = sorted(layer_cycles.items(), key=lambda item: item[1], reverse=True)
    bottleneck_name, bottleneck_cycles = sorted_layers[0]

    if bottleneck_cycles <= 1:
        # Já está 100% desenrolado
        return new_folding

    # 3. Tenta otimizar o gargalo
    data = cycle_formulas.get(bottleneck_name, {})
    op_type = data.get("op_type", "")
    cfg = new_folding.get(bottleneck_name)
    if not cfg:
        # Puxa o Default caso ainda não tenha config explícita
        new_folding[bottleneck_name] = {"PE": 1, "SIMD": 1, "ram_style": "auto", "resType": "auto"}
        cfg = new_folding[bottleneck_name]

    current_pe = cfg.get("PE", 1)
    current_simd = cfg.get("SIMD", 1)

    if "SIMD" in data.get("formula", ""):
        if "MVAU" in op_type or "MatrixVectorActivation" in op_type:
            next_simd = analyzer._find_next_valid_parallelism(bottleneck_name, current_simd, op_type, data, "SIMD")
            if next_simd > current_simd:
                cfg["SIMD"] = next_simd
            else:
                next_pe = analyzer._find_next_valid_parallelism(bottleneck_name, current_pe, op_type, data, "PE")
                if next_pe > current_pe:
                    cfg["PE"] = next_pe
        elif "ConvolutionInputGenerator" in op_type:
            next_simd = analyzer._find_next_valid_parallelism(bottleneck_name, current_simd, op_type, data, "SIMD")
            if next_simd > current_simd:
                cfg["SIMD"] = next_simd
            else:
                if cfg.get("parallel_window", 0) == 0:
                    cfg["parallel_window"] = 1
        else: # Outras camadas baseadas em SIMD (ex: FMPadding)
            next_simd = analyzer._find_next_valid_parallelism(bottleneck_name, current_simd, op_type, data, "SIMD")
            if next_simd > current_simd:
                cfg["SIMD"] = next_simd
    elif "PE" in data.get("formula", ""):
        next_pe = analyzer._find_next_valid_parallelism(bottleneck_name, current_pe, op_type, data, "PE")
        if next_pe > current_pe:
            cfg["PE"] = next_pe

    return new_folding


def _get_base_onnx(request_path, base_build_dir):
    import yaml
    from utils.hw_utils import get_finn_ready_model
    import run_fps_map_job
    
    with open(request_path, 'r') as f:
        request_data = json.load(f)
        
    model_id = request_data.get('model_id')
    fpga_part = request_data.get('fpga_part', 'xc7z020clg400-1')
    
    with open('./hara/models/registry_models.yaml', 'r') as f:
        model_registry = yaml.safe_load(f)
        
    model_info = model_registry.get(model_id)
    quant = model_info.get("quant", model_info.get("weight_quant"))
    
    print("-> Preparando ONNX principal...")
    master_onnx_path = get_finn_ready_model(model_info, base_build_dir)
    
    print("-> Estabelecendo baseline com FINN (run0) para gerar o base.onnx do modelo HW...")
    # Roda apenas a primeira passagem de folding para converter as camadas lógicas para hardware-style (MVAU, etc)
    est_dir_1 = run_fps_map_job._run_estimate_build(
        base_build_dir, master_onnx_path, "run0_get_initial_fold", 
        model_info.get("topology_id"), quant, fpga_part, target_fps=1
    )
    
    intermediate_onnx_path = os.path.join(est_dir_1, "intermediate_models", "step_generate_estimate_reports.onnx")
    return intermediate_onnx_path

def main():
    parser = argparse.ArgumentParser(description="Gera todos os foldings analiticamente.")
    parser.add_argument("--request", required=True, help="Caminho para o arquivo request.json (ex: requests/MNIST/req_fps_...).")
    parser.add_argument("--output-dir", required=True, help="Pasta onde as configurações JSON serão salvas.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Gera o ONNX da mesma forma que o job faria
    onnx_path = _get_base_onnx(args.request, args.output_dir)
    
    if not onnx_path or not os.path.exists(onnx_path):
        print("[✗] Falha ao gerar ou encontrar o ONNX HW-ready inicial.")
        return

    print(f"-> Analisando ONNX de Hardware: {onnx_path}")
    analyzer = FinnCycleEstimator(onnx_path, debug=False)
    cycle_formulas = analyzer.get_cycle_formulas()

    if not cycle_formulas:
        print("[✗] Falha: Nenhuma fórmula de Hardware pôde ser extraída do ONNX.")
        return

    print("-> Gerando folding base (Reset Folding)...")
    current_folding = utils.reset_folding({}, onnx_path)
    
    step = 1
    out_path = os.path.join(args.output_dir, f"folding_step_{step:03d}.json")
    with open(out_path, "w") as f:
        json.dump(current_folding, f, indent=2)
    print(f"[✓] Step {step}: {out_path} salvo.")

    while True:
        step += 1
        new_folding = modify_folding_analytical(current_folding, analyzer, cycle_formulas)
        
        if new_folding == current_folding:
            print(f"\n[✓] Otimização analítica chegou ao limite! Máximo de paralelismo atingido.")
            print(f"Total de {step-1} configurations geradas em {args.output_dir}")
            break
            
        current_folding = new_folding
        out_path = os.path.join(args.output_dir, f"folding_step_{step:03d}.json")
        with open(out_path, "w") as f:
            json.dump(current_folding, f, indent=2)
        
        print(f"[✓] Step {step}: {out_path} salvo.")

if __name__ == "__main__":
    main()
