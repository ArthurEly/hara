import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import time

from utils.hw_utils import utils, get_finn_ready_model
from config import BUILD_CONFIG

def soft_clean_build_artifacts(build_dir):
    """Limpa artefatos pesados mantendo os relatórios (.rpt, .log) e o ONNX essencial da síntese."""
    import shutil
    print(f"  -> 🧹 Limpeza Seletiva (mantendo logs, relatórios HLS e ONNXs críticos): {build_dir}")
    
    # 1. Filtro no intermediate_models: manter apenas estimate_reports e stitched_ip
    im_dir = os.path.join(build_dir, "intermediate_models")
    if os.path.exists(im_dir):
        for f in os.listdir(im_dir):
            if not f.endswith("step_create_stitched_ip.onnx") and not f.endswith("step_generate_estimate_reports.onnx"):
                p = os.path.join(im_dir, f)
                try:
                    if os.path.isfile(p): os.remove(p)
                    elif os.path.isdir(p): shutil.rmtree(p)
                except Exception: pass
                
    # 2. Filtro agressivo porém seguro em Pastas de IP (output_ip e stitched_ip)
    # Nessas pastas ficam os valiosos relatórios de Per-Layer Vivado HLS (.rpt e .log), 
    # que o FINN não envia pro /report/. Vamos apagar todo o lixo RTL/Verilog e deixar só .rpt/.log/.txt
    for ip_dir_name in ["output_ip", "stitched_ip"]:
        t_dir = os.path.join(build_dir, ip_dir_name)
        if os.path.exists(t_dir):
            for root, dirs, files in os.walk(t_dir, topdown=False):
                for name in files:
                    # Se não for report, log ou xml/csv associado a log, mata (como .v, .sv, .dat, .zi, etc)
                    if not (name.endswith('.rpt') or name.endswith('.log') or name.endswith('.xml') or name.endswith('.csv') or name.endswith('.txt')):
                        try: os.remove(os.path.join(root, name))
                        except Exception: pass
                        
    # 3. Exclusão direta de caches irreversíveis
    for f in ["vivado_ip_cache", "bitfile", "driver", "pyverilator_ipstitched"]:
        p = os.path.join(build_dir, f)
        if os.path.exists(p): 
            try: shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            except Exception: pass
            
    # 4. Limpeza suave no vivado_zynq_proj (arquivos .dcp e .zip são gigabytes de binário do Vivado, lixo puro)
    vz_dir = os.path.join(build_dir, "vivado_zynq_proj")
    if os.path.exists(vz_dir):
        for root, dirs, files in os.walk(vz_dir, topdown=False):
            for name in files:
                if name.endswith('.dcp') or name.endswith('.zip') or name.endswith('.bit') or name.endswith('.bin'):
                    try: os.remove(os.path.join(root, name))
                    except Exception: pass

    # 5. Limpeza do cache global do FINN (somente os arquivos internos para não quebrar o mount do Docker)
    finn_tmp = "/tmp/finn_dev_arthurely"
    if os.path.exists(finn_tmp):
        print(f"  -> 🗑️ Limpando cache global do FINN: {finn_tmp}")
        try:
            import subprocess
            subprocess.run(f"rm -rf {finn_tmp}/*", shell=True, check=False)
        except Exception as e: 
            print(f"  -> [!] Erro ao limpar {finn_tmp}: {e}")

def get_failure_reason(build_dir):
    log_path = os.path.join(build_dir, "build_dataflow.log")
    if not os.path.exists(log_path):
        return "failed_unknown"
    
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        if "RAMB36E1 over-utilized" in content or "RAMB18 and RAMB36/FIFO over-utilized" in content or "RAMB36/FIFO over-utilized" in content or "RAMB" in content or "BRAM" in content:
            return "failed_BRAM"
        if "Slice LUTs over-utilized" in content or "LUT as Memory over-utilized" in content or "LUT as Logic over-utilized" in content or "LUT" in content:
            return "failed_LUT"
        if "DSP48E1 over-utilized" in content or "DSP" in content:
            return "failed_DSP"
        if "FF " in content or "Flip Flop" in content or "FFs over-utilized" in content:
            return "failed_FF"
        if "timeout" in content.lower():
            return "failed_timeout"
            
        return "failed_other"
    except Exception:
        return "failed_error_reading_log"

def _run_full_hw_build(base_build_dir, onnx_model_path, hw_name, topology_id, quant, fpga_part, folding_path=None, target_fps=None):
    """
    Executa um build COMPLETO para HW (até out-of-context synthesis)
    """
    print(f"\n--- Executando Hardware Build COMPLETO: {hw_name} na FPGA {fpga_part} ---")
    build_output_dir = os.path.join(base_build_dir, hw_name)
    steps = BUILD_CONFIG['first_run_build']['steps']
    
    args = [
        "python3", "run_build.py",
        "--model_path", str(onnx_model_path),
        "--build_dir", str(base_build_dir),
        "--topology", str(topology_id),
        "--steps", json.dumps(steps),
        "--hw_name", hw_name,
        "--fpga-part", fpga_part,
        "--folding_file", str(folding_path) if folding_path else "",
        "--target_fps", str(target_fps) if target_fps else "None",
    ]
    
    if quant is not None:
        args.extend(["--quant", str(quant)])
    
    log_path = os.path.join(base_build_dir, f"build_{hw_name}.log")
    try:
        # Usa timeout longo caso Vitis HLS/Vivado demore 40min+
        utils.run_and_capture(args, timeout_sec=10800, log_path=log_path)
        print(f"[✓] Build Completo {hw_name} concluído.")
        return build_output_dir, True
    except RuntimeError:
        print(f"[✗] Build Completo {hw_name} falhou.")
        return build_output_dir, False

def _run_estimate_build(base_build_dir, onnx_model_path, hw_name, topology_id, quant, fpga_part, target_fps=None):
    """Executa apenas a pipeline FINN até estimate_reports (super rápido) para extrair MVAUs"""
    print(f"\n--- Extraindo Template ONNX/Folding Base: {hw_name} ---")
    build_output_dir = os.path.join(base_build_dir, hw_name)
    steps = BUILD_CONFIG['first_run_estimate']['steps']
    
    args = [
        "python3", "run_build.py",
        "--model_path", str(onnx_model_path),
        "--build_dir", str(base_build_dir),
        "--topology", str(topology_id),
        "--steps", json.dumps(steps),
        "--hw_name", hw_name,
        "--fpga-part", fpga_part,
        "--target_fps", str(target_fps) if target_fps else "None",
    ]
    if quant is not None: args.extend(["--quant", str(quant)])
    
    log_path = os.path.join(base_build_dir, f"build_{hw_name}.log")
    try:
        utils.run_and_capture(args, log_path=log_path)
        print(f"[✓] Template base extraído com sucesso.")
        return build_output_dir
    except RuntimeError:
        print(f"[✗] Falha na extração de template inicial.")
        return None

def generate_map(model_info, base_build_dir, fpga_part, starting_folding=None):
    from config import MAX_RESOURCES
    
    topology_id = model_info.get("topology_id")
    quant = model_info.get("quant") or model_info.get("weight_quant")
    csv_path = os.path.join(base_build_dir, "hardware_summary.csv")
    
    try:
        master_onnx_path = get_finn_ready_model(model_info, base_build_dir)
        if not master_onnx_path:
            raise RuntimeError("Falha ao preparar o modelo ONNX inicial.")
    except Exception as e:
        print(f"[✗] ERRO: {e}"); return

    print("\n--- Estabelecendo template analítico da Rede (Sem HW Síntese) ---")
    est_dir_1 = _run_estimate_build(base_build_dir, master_onnx_path, "run0_get_initial_fold", topology_id, quant, fpga_part, target_fps=1)
    if not est_dir_1: return

    initial_folding = utils.read_folding_config(est_dir_1)
    intermediate_onnx_path = os.path.join(est_dir_1, "intermediate_models", "step_generate_estimate_reports.onnx")
    if not os.path.exists(intermediate_onnx_path):
        print("[✗] ONNX para reset não encontrado.")
        return
    
    # Lê area_constraints e fixed_resources do request.json antes de qualquer branch
    request_file_path = os.path.join(base_build_dir, "request.json")
    fixed_resources = None
    area_constraints = None
    if os.path.exists(request_file_path):
        with open(request_file_path, "r") as f:
            req_data = json.load(f)
            fixed_resources = req_data.get("fixed_resources")
            area_constraints = req_data.get("area_constraints")
    
    if starting_folding and os.path.exists(starting_folding):
        print(f"--- Usando ponto de partida (folding) customizado: {starting_folding} ---")
        with open(starting_folding, 'r') as f:
            reset_folding = json.load(f)
    else:
        reset_folding = utils.reset_folding(initial_folding, intermediate_onnx_path, fixed_resources=fixed_resources)
    
    # Limpa a run0 usando a limpeza suave para manter logs vitais
    soft_clean_build_artifacts(est_dir_1)

    baseline_hw_name = "run1_baseline_folded"
    folding_path = os.path.join(base_build_dir, f"{baseline_hw_name}.json")
    with open(folding_path, 'w') as f: json.dump(reset_folding, f, indent=2)
    
    start_t = time.time()
    baseline_dir, baseline_success = _run_full_hw_build(base_build_dir, master_onnx_path, baseline_hw_name, topology_id, quant, fpga_part, folding_path=folding_path)
    dur = int(time.time() - start_t)
    
    if not baseline_success:
        reason = get_failure_reason(baseline_dir)
        utils.append_run_summary(csv_path, baseline_hw_name, reason, reset_folding, dur, baseline_dir, MAX_RESOURCES)
        print(f"-> Ponto #1 salvo em {csv_path} com erro ({reason})!")
        if reason in ["failed_LUT", "failed_FF", "failed_DSP"]:
            print(f"[!] FATAL: Limite de recursos de lógica/matemática atingido no baseline ({reason}). Interrompendo.")
            return
    else:
        # Grava log oficial no formato legível pro HARA V1
        utils.append_run_summary(csv_path, baseline_hw_name, "success", reset_folding, dur, baseline_dir, MAX_RESOURCES)
        print(f"-> Ponto #1 salvo em {csv_path}!")
        
        # CHECAGEM DE ORÇAMENTO (BUDGET OVERFLOW)
        if area_constraints is not None:
            area_data = utils.extract_area_from_rpt(baseline_dir)
            if area_data:
                exceeded_all = [f"{k} ({area_data.get(k,0)}/{v})" for k, v in area_constraints.items() if area_data.get(k, 0) > v]
                
                # Recursos "fatais" que interrompem o processo
                fatal_keys = ["Total LUTs", "FFs", "DSP Blocks"]
                exceeded_fatal = [f"{k} ({area_data.get(k,0)}/{v})" for k, v in area_constraints.items() if k in fatal_keys and area_data.get(k, 0) > v]
                
                if exceeded_fatal:
                    print(f"[!] Baseline violou limites CRÍTICOS (Lógica/DSP) do request.json: {exceeded_fatal}. Interrompendo.")
                    return
                elif exceeded_all:
                    print(f"[!] AVISO: Baseline violou limites do request.json (BRAM), mas continuará: {exceeded_all}")

    current_folding = reset_folding
    last_build_dir = baseline_dir
    run_counter = 2

    while True:
        print(f"\n--- Tentativa de Otimização HW #{run_counter} ---")
        onnx_path_loop = os.path.join(last_build_dir, "intermediate_models/step_generate_estimate_reports.onnx")
        cycles_path = os.path.join(last_build_dir, "report/estimate_layer_cycles.json")
        
        if not os.path.exists(onnx_path_loop) or not os.path.exists(cycles_path):
            print("[!] Arquivos da última execução não encontrados. Encerrando loop.")
            break
        
        with open(cycles_path, 'r') as f: estimate_layer_cycles = json.load(f)
        new_folding = utils.modify_folding(current_folding, onnx_path_loop, estimate_layer_cycles)
        
        # LIMPEZA SELETIVA DO HW ANTERIOR
        soft_clean_build_artifacts(last_build_dir)
        
        if new_folding == current_folding:
            print("[✓] Design estável alcançado (Full Unfolded). Fim da exploração exaustiva.")
            break
        
        current_folding = new_folding
        current_hw_name = f"run{run_counter}_optimized"
        folding_path = os.path.join(base_build_dir, f"{current_hw_name}.json")
        with open(folding_path, 'w') as f: json.dump(current_folding, f, indent=2)
        
        start_t = time.time()
        current_build_dir, success = _run_full_hw_build(base_build_dir, master_onnx_path, current_hw_name, topology_id, quant, fpga_part, folding_path=folding_path)
        dur = int(time.time() - start_t)
        
        if not success:
            reason = get_failure_reason(current_build_dir)
            print(f"[✗] Vivado falhou ({reason}). Registrando e continuando a exploração...")
            utils.append_run_summary(csv_path, current_hw_name, reason, current_folding, dur, current_build_dir, MAX_RESOURCES)
            if reason in ["failed_LUT", "failed_FF", "failed_DSP"]:
                print(f"[!] FATAL: Limite de recursos de lógica/matemática atingido ({reason}). O algoritmo deve parar para este modelo.")
                break
        else:
            utils.append_run_summary(csv_path, current_hw_name, "success", current_folding, dur, current_build_dir, MAX_RESOURCES)
            print(f"-> Ponto #{run_counter} anotado com sucesso!")
            
            # CHECAGEM DE ORÇAMENTO (BUDGET OVERFLOW)
            if area_constraints is not None:
                area_data = utils.extract_area_from_rpt(current_build_dir)
                if area_data:
                    constraints_dict: dict[str, float] = area_constraints # type: ignore
                    exceeded_all = [f"{k} ({area_data.get(k,0)}/{v})" for k, v in constraints_dict.items() if area_data.get(k, 0) > v]
                    
                    # Recursos "fatais" que interrompem o processo
                    fatal_keys = ["Total LUTs", "FFs", "DSP Blocks"]
                    exceeded_fatal = [f"{k} ({area_data.get(k,0)}/{v})" for k, v in constraints_dict.items() if k in fatal_keys and area_data.get(k, 0) > v]
                    
                    if exceeded_fatal:
                        print(f"[!] Run #{run_counter} violou limites CRÍTICOS (Lógica/DSP) do request.json: {exceeded_fatal}. Interrompendo algoritmo.")
                        break
                    elif exceeded_all:
                        print(f"[!] AVISO: Run #{run_counter} violou limites do request.json (BRAM), mas continuará: {exceeded_all}")
        
        last_build_dir = current_build_dir
        
        run_counter += 1

    print(f"\nMapa EXAUSTIVO DE HARDWARE salvo com sucesso no padrão HARA V1 em: {csv_path}")

if __name__ == "__main__":
    import shutil
    from datetime import datetime
    import glob
    
    parser = argparse.ArgumentParser(description="Script Exaustivo HARA V1 - Realiza Builds de Vivado Inteiros Iterativamente")
    parser.add_argument('--build_dir', type=str, help="Diretório que já contém o request.json do modelo alvo")
    parser.add_argument('--request', nargs='+', type=str, help="Caminho direto para o(s) json(s) ou pastas. Criará um build_dir automático com timestamp para cada um.")
    parser.add_argument('--starting_folding', type=str, help="Opcional. Caminho para um .json de folding que servirá como ponto de partida inicial do hardware.")
    args = parser.parse_args()

    if not args.build_dir and not args.request:
        parser.error("Forneça --build_dir OU --request")

    if args.request:
        requests_list = []
        for req in args.request:
            if os.path.isdir(req):
                requests_list.extend(glob.glob(os.path.join(req, "*.json")))
            elif "*" in req:
                requests_list.extend(glob.glob(req))
            else:
                requests_list.append(req)

        for req_path in requests_list:
            if not os.path.exists(req_path):
                print(f"[!] Arquivo de requisição ausente: {req_path}")
                continue
            
            with open(req_path, 'r') as f: req_data = json.load(f)
            
            # Criação da pasta de build automática baseada no nome do modelo e data/hora
            model_id = req_data.get('model_id', 'UNKNOWN_MODEL')
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            new_build_dir = f"exhaustive_hw_builds/{model_id}_{timestamp}"
            
            os.makedirs(new_build_dir, exist_ok=True)
            request_file_path = os.path.join(new_build_dir, "request.json")
            shutil.copy(req_path, request_file_path)
            
            fpga_part = req_data.get('fpga_part', 'xc7z020clg400-1') 
            
            with open('models/registry_models.yaml', 'r') as f:
                model_registry = yaml.safe_load(f)

            model_info = model_registry.get(model_id)
            if not model_info: 
                print(f"[!] Modelo '{model_id}' não localizado. Ignorando...")
                continue
                
            print(f"\n========================================================")
            print(f"Diretório Raiz da Exaustão de HW: {new_build_dir} (via {req_path})")
            print(f"========================================================\n")
            
            generate_map(model_info, new_build_dir, fpga_part, starting_folding=args.starting_folding)
    else:
        request_file_path = os.path.join(args.build_dir, "request.json")
        if not os.path.exists(request_file_path):
            raise FileNotFoundError(f"request.json não encontrado em: {request_file_path}")

        with open(request_file_path, 'r') as f: request_data = json.load(f)
        model_id = request_data.get('model_id')
        fpga_part = request_data.get('fpga_part', 'xc7z020clg400-1') 
        
        with open('models/registry_models.yaml', 'r') as f:
            model_registry = yaml.safe_load(f)

        model_info = model_registry.get(model_id)
        if not model_info: raise ValueError(f"Modelo '{model_id}' não localizado.")
            
        print(f"\n========================================================")
        print(f"Diretório Raiz da Exaustão de HW: {args.build_dir}")
        print(f"========================================================\n")
        
        generate_map(model_info, args.build_dir, fpga_part, starting_folding=args.starting_folding)
