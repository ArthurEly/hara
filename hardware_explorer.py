import os
import json
import time
import random
import csv
import numpy as np

# Importa as ferramentas e as configurações
from utils.hw_utils import utils, get_finn_ready_model
from config import (
    TARGET_RESOURCE_PERCENTAGES,
    BUILD_CONFIG,
    HARA_LOOP_CONFIG
)

class HardwareExplorer:
    def __init__(self, build_dir, config, resource_limits, hara_loop_config, simulation_mode=False, fixed_resources=None, fpga_part="xc7z020clg400-1"):
        self.base_build_dir = build_dir
        self.build_config = config
        self.resource_limits_max = resource_limits
        self.hara_loop_config = hara_loop_config
        self.summary_file = os.path.join(self.base_build_dir, "hardware_summary.csv")
        self.run_number = 1
        self.last_valid_folding = None
        self.last_valid_hw_name = None
        self.last_valid_build_dir = None
        
        self.simulation_mode = simulation_mode
        self.fixed_resources = fixed_resources if fixed_resources else {}
        self.fpga_part = fpga_part

        print(f"HardwareExplorer inicializado. Resultados em: {self.base_build_dir}")

    # ... [MÉTODOS _run_fake_build e _run_single_build (Mantidos iguais)] ...
    def _run_fake_build(self, onnx_model_path, hw_name, quant, topology_id, steps, folding_path=None, target_fps=None, resource_limits=None):
        import datetime
        print(f"-> [SIMULAÇÃO DIRETA] Iniciando build FALSO para: {hw_name}")
        time.sleep(random.uniform(0.5, 1.0)) 
        
        build_output_dir = os.path.join(self.base_build_dir, hw_name)
        os.makedirs(os.path.join(build_output_dir, "report"), exist_ok=True)

        input_folding = {}
        if folding_path and os.path.exists(folding_path):
            with open(folding_path, 'r') as f: input_folding = json.load(f)
        
        with open(os.path.join(build_output_dir, "final_hw_config.json"), 'w') as f:
            json.dump(input_folding, f, indent=4)
        
        total_parallelism = 1.0
        for layer, config in input_folding.items():
            if layer != "Defaults": total_parallelism += config.get("PE", 1) * config.get("SIMD", 1)

        base_luts = 15000 + (total_parallelism * 25)
        # Simulando falha se o paralelismo for muito alto para forçar o fallback
        multiplier = 1.5 if total_parallelism > 500 else 1.0 
        
        fake_luts = min(int(self.resource_limits_max["Total LUTs"] * 1.5), int(base_luts * multiplier * (1 + random.uniform(-0.05, 0.05))))
        fake_ffs = min(self.resource_limits_max["FFs"], int(20000 + (total_parallelism * 35)))
        fake_bram = min(self.resource_limits_max["BRAM (36k)"], 15 + (total_parallelism * 0.05))
        
        fake_area_data = {
            "Total LUTs": fake_luts, "Logic LUTs": int(fake_luts * 0.95),
            "LUTRAMs": int(fake_luts * 0.05), "SRLs": int(fake_luts * 0.05),
            "FFs": fake_ffs, "BRAM (36k)": round(fake_bram, 1), "DSP Blocks": 0
        }

        fake_fps = 10 + total_parallelism * 0.5 * (1 + random.uniform(-0.05, 0.05))
        perf_report = {"estimated_throughput_fps": fake_fps, "max_cycles_node_name": "MVAU_fake"}
        with open(os.path.join(build_output_dir, "report", "estimate_network_performance.json"), 'w') as f: json.dump(perf_report, f)
        
        fake_cycles = {name: int(random.uniform(1000, 50000)) for name in input_folding.keys() if name != "Defaults"}
        with open(os.path.join(build_output_dir, "report", "estimate_layer_cycles.json"), 'w') as f: json.dump(fake_cycles, f)
            
        resource_diffs = utils.check_resource_usage(fake_area_data, resource_limits or {})
        exceeded = utils.raise_if_exceeds_limits(resource_diffs)
        status = "resources_exceeded" if exceeded else "success"
        
        summary_row = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hw_name": hw_name, "status": status, "duration_in_seconds": 1.5,
            "folding_summary": json.dumps(input_folding), "folding_diff": json.dumps({}),
            "build_dir": build_output_dir, "resource_limits": json.dumps(resource_limits or {}),
            **fake_area_data, **perf_report
        }
        
        field_order = [
            "date", "hw_name", "status", "duration_in_seconds", "folding_summary",
            "folding_diff", "build_dir", "resource_limits", "Total LUTs", "Logic LUTs",
            "LUTRAMs", "SRLs", "FFs", "BRAM (36k)", "DSP Blocks",
            "estimated_throughput_fps", "max_cycles_node_name"
        ]
        file_exists = os.path.isfile(self.summary_file)
        with open(self.summary_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_order)
            if not file_exists: writer.writeheader()
            writer.writerow(summary_row)

        if status == "success": print(f"[✓] [SIM] {hw_name} (FPS: {fake_fps:.0f}) -> OK")
        else: print(f"[!] [SIM] {hw_name} (FPS: {fake_fps:.0f}) -> FALHA (Recursos)")
        
        return {"status": status, "folding": input_folding, "hw_name": hw_name, "build_dir": build_output_dir}

    def _run_single_build(self, onnx_model_path, hw_name, quant, topology_id, steps, folding_path=None, target_fps=None, resource_limits=None):
        if self.simulation_mode:
            return self._run_fake_build(onnx_model_path, hw_name, quant, topology_id, steps, folding_path, target_fps, resource_limits)

        print(f"-> [REAL] Iniciando build para: {hw_name}")
        args = [
            "python3", "./hara/run_build.py",
            "--model_path", str(onnx_model_path), "--build_dir", str(self.base_build_dir),
            "--topology", str(topology_id), "--steps", json.dumps(steps),
            "--hw_name", hw_name, "--fpga-part", self.fpga_part,             
            "--folding_file", str(folding_path) if folding_path else "",
            "--target_fps", str(target_fps) if target_fps else "None",
            "--run", str(self.run_number)
        ]
        if quant is not None: args.extend(["--quant", str(quant)])

        log_path = os.path.join(self.base_build_dir, f"build_{hw_name}.log")
        start_time = time.time()
        build_output_dir = os.path.join(self.base_build_dir, hw_name)
        
        try:
            utils.run_and_capture(args, log_path=log_path)
            duration = round(time.time() - start_time, 2)
            area_data = utils.extract_area_from_rpt(build_output_dir)
            resource_diffs = utils.check_resource_usage(area_data, resource_limits or {})
            exceeded = utils.raise_if_exceeds_limits(resource_diffs)
            status = "resources_exceeded" if exceeded else "success"
            
            if status == "success": print(f"[✓] Build {hw_name} SUCESSO ({duration}s).")
            else: print(f"[!] Build {hw_name} FALHA POR RECURSOS.")

            folding_config = utils.read_folding_config(build_output_dir)
            utils.append_run_summary(self.summary_file, hw_name, status, folding_config, duration, build_output_dir, resource_limits or {})
            return {"status": status, "folding": folding_config, "hw_name": hw_name, "build_dir": build_output_dir}

        except RuntimeError:
            print(f"[✗] Build {hw_name} CRASH.")
            folding_config = {}
            if os.path.exists(build_output_dir): folding_config = utils.read_folding_config(build_output_dir)
            utils.save_crash_report(build_output_dir)
            utils.append_run_summary(self.summary_file, hw_name, "crash", folding_config, 0, build_output_dir, resource_limits or {})
            return {"status": "crash"}

    # --- NOVO MÉTODO: Salvar Debug CSV ---
    def _save_debug_map_csv(self, map_configs):
        debug_csv_path = os.path.join(self.base_build_dir, "debug_theoretical_fps_map.csv")
        try:
            with open(debug_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["index", "hw_name_base", "estimated_fps"])
                for idx, cfg in enumerate(map_configs):
                    writer.writerow([idx, cfg['hw_name_base'], f"{cfg['fps']:.2f}"])
            print(f"   -> Mapa teórico salvo para debug em: {debug_csv_path}")
        except Exception as e:
            print(f"   [!] Erro ao salvar CSV de debug: {e}")

    # --- DSE MODIFICADO ---

    def _generate_theoretical_map(self, onnx_model_path, topology_id, quant, min_fps, max_fps):
        print(f"\n🗺️  [DSE] Gerando Mapa Teórico no intervalo [{min_fps}, {max_fps}]...")
        map_configs = [] 
        
        # 1. Reset
        hw_name_reset = "dse_step0_reset"
        cfg_est = self.build_config['first_run_estimate']
        res_reset = self._run_single_build(onnx_model_path, hw_name_reset, quant, topology_id, cfg_est['steps'], target_fps=1)
        if res_reset.get('status') != 'success': return []

        initial_folding = utils.read_folding_config(res_reset['build_dir'])
        intermediate_onnx = os.path.join(res_reset['build_dir'], "intermediate_models", "step_generate_estimate_reports.onnx")
        
        if not os.path.exists(intermediate_onnx) and not self.simulation_mode: return []
        
        current_folding = utils.reset_folding(initial_folding, intermediate_onnx if not self.simulation_mode else onnx_model_path, self.fixed_resources)
        
        # Loop
        step_count = 1
        while True:
            hw_name_est = f"dse_est_step{step_count}"
            folding_path = os.path.join(self.base_build_dir, f"{hw_name_est}.json")
            with open(folding_path, 'w') as f: json.dump(current_folding, f, indent=2)
            
            res = self._run_single_build(onnx_model_path, hw_name_est, quant, topology_id, cfg_est['steps'], folding_path=folding_path, target_fps=None)
            utils.clean_build_artifacts(res['build_dir'])
            
            if res.get('status') != 'success': break
            
            fps = 0
            perf_path = os.path.join(res['build_dir'], "report", "estimate_network_performance.json")
            if os.path.exists(perf_path):
                with open(perf_path) as f: fps = json.load(f).get("estimated_throughput_fps", 0)
            
            # Adiciona ao mapa se >= min_fps (ou se for o primeiro, pra ter baseline)
            if fps >= min_fps or not map_configs:
                # print(f"      [Step {step_count}: {fps:.2f} FPS] -> Mapeado")
                map_configs.append({'fps': fps, 'folding': current_folding, 'hw_name_base': hw_name_est})
            
            if max_fps is not None and fps >= max_fps: break

            cycles_path = os.path.join(res['build_dir'], "report", "estimate_layer_cycles.json")
            safe_onnx_path = intermediate_onnx if not self.simulation_mode else onnx_model_path

            if not os.path.exists(cycles_path): break
            with open(cycles_path) as f: cycles = json.load(f)
            
            new_folding = utils.modify_folding(current_folding, safe_onnx_path, cycles)
            if new_folding == current_folding: break
            current_folding = new_folding
            step_count += 1
            
        return map_configs

    def run_sparse_dse(self, onnx_model_path, model_info, min_fps, max_fps, num_builds, save_builds):
        """
        DSE Esparso com Seleção Baseada em VALOR de FPS e Teto Dinâmico.
        """
        topology_id = model_info.get("topology_id")
        quant = model_info.get("quant")
        if quant is None: quant = model_info.get("weight_quant", "mixed")

        # 1. Gerar o Mapa Teórico Completo
        map_configs = self._generate_theoretical_map(onnx_model_path, topology_id, quant, min_fps, max_fps)
        if not map_configs:
            print("[✗] Mapa vazio. Encerrando.")
            return

        map_configs.sort(key=lambda x: x['fps']) 
        self._save_debug_map_csv(map_configs) # Salva CSV para debug
        
        print(f"\n📊 Mapa Teórico: {len(map_configs)} pontos. Range: [{map_configs[0]['fps']:.1f} - {map_configs[-1]['fps']:.1f}] FPS")

        cfg_build = self.build_config['hara_build']
        processed_indices = set()
        successful_builds = 0
        
        # --- FASE 1: Encontrar o Teto Real (Upper Bound) ---
        print("\n🔍 [FASE 1] Determinando o Teto Real (Upper Bound)...")
        
        valid_max_fps = 0
        valid_max_index = -1
        
        # Itera do maior para o menor até achar um que cabe
        for i in reversed(range(len(map_configs))):
            config = map_configs[i]
            print(f"   -> Testando candidato a Teto: #{i} ({config['fps']:.1f} FPS)...")
            
            hw_name = f"dse_SEARCH_MAX_try{i}_{int(config['fps'])}fps"
            folding_path = os.path.join(self.base_build_dir, f"{hw_name}_folding.json")
            with open(folding_path, 'w') as f: json.dump(config['folding'], f, indent=2)

            result = self._run_single_build(
                onnx_model_path, hw_name, quant, topology_id, cfg_build['steps'],
                folding_path=folding_path, resource_limits=self.resource_limits_max
            )
            
            if not save_builds: utils.clean_build_artifacts(result['build_dir'])

            if result['status'] == 'success':
                print(f"   ✅ Teto encontrado: {config['fps']:.1f} FPS (Index #{i})")
                valid_max_fps = config['fps']
                valid_max_index = i
                processed_indices.add(i) # Já foi buildado
                successful_builds += 1
                utils.plot_area_usage_from_csv(self.summary_file, self.base_build_dir)
                break
            else:
                print(f"   ❌ Falhou (Index #{i}). Tentando anterior...")
        
        if valid_max_index == -1:
            print("[✗] Nenhum design coube na FPGA (nem o mínimo). DSE Falhou.")
            return

        # Se só queríamos 1 build ou se o teto já é o mínimo, acabamos
        if num_builds <= 1 or valid_max_index == 0:
            print("   -> Teto encontrado e critério de parada atingido.")
            return

        # --- FASE 2: Builds Esparsas Baseadas em Valor ---
        print(f"\n🔍 [FASE 2] Preenchendo {num_builds-1} builds no intervalo [{min_fps} - {valid_max_fps:.1f}] FPS...")
        
        # Gera os ALVOS matemáticos (ex: 500, 1625, 2750...)
        # num_builds inclui o teto que já fizemos, então geramos num_builds pontos
        target_fps_values = np.linspace(min_fps, valid_max_fps, num_builds)
        
        # Remove o último (pois teoricamente é o max que já fizemos, ou muito perto)
        # Vamos processar do maior para o menor (excluindo o max que já foi)
        targets_to_process = target_fps_values[:-1] # Remove o teto da lista de tarefas
        
        # Inverte para tentar os maiores primeiro (opcional, mas bom pra ver gargalos logo)
        targets_to_process = targets_to_process[::-1] 
        
        print(f"   -> Alvos calculados: {[int(t) for t in targets_to_process]}")

        for target in targets_to_process:
            # Encontra a config no mapa (até o valid_max_index) que tem o FPS mais próximo do target
            # Filtramos map_configs[:valid_max_index] pois sabemos que acima disso não cabe
            candidates = map_configs[:valid_max_index+1]
            
            # A função chave calcula a distância absoluta do FPS
            best_config = min(candidates, key=lambda x: abs(x['fps'] - target))
            
            # Encontra o índice original dessa config
            best_idx = map_configs.index(best_config)
            
            if best_idx in processed_indices:
                print(f"   -> Alvo {int(target)} FPS mapeia para Index #{best_idx} ({best_config['fps']:.1f} FPS) - JÁ PROCESSADO.")
                continue
                
            print(f"\n   🚀 Build Alvo: {int(target)} FPS -> Melhor Canditato: #{best_idx} ({best_config['fps']:.1f} FPS)")
            
            hw_name = f"dse_sparse_tgt{int(target)}_try{best_idx}_{int(best_config['fps'])}fps"
            folding_path = os.path.join(self.base_build_dir, f"{hw_name}_folding.json")
            with open(folding_path, 'w') as f: json.dump(best_config['folding'], f, indent=2)

            result = self._run_single_build(
                onnx_model_path, hw_name, quant, topology_id, cfg_build['steps'],
                folding_path=folding_path, resource_limits=self.resource_limits_max
            )
            
            if not save_builds: utils.clean_build_artifacts(result['build_dir'])
            
            processed_indices.add(best_idx)
            
            if result['status'] == 'success':
                print(f"      ✅ Sucesso.")
                successful_builds += 1
                utils.plot_area_usage_from_csv(self.summary_file, self.base_build_dir)
            else:
                # Se falhar aqui (o que é raro, pois está abaixo do teto), podemos tentar um vizinho?
                # Por simplicidade e para manter "esparso", apenas registramos a falha.
                # Se quiser robustez total, poderia adicionar um mini-fallback aqui.
                print(f"      ❌ Falhou inesperadamente (estava abaixo do teto).")

        print(f"\n✅ DSE Finalizado. {successful_builds}/{num_builds} builds concluídas com sucesso.")

    def run_exploration(self, model_info, target_fps=None, min_fps=0, num_builds=-1, save_builds=True):
        topology_id = model_info.get("topology_id")
        quant = model_info.get("quant")
        if quant is None: quant = model_info.get("weight_quant", "mixed")

        print(f"--- Iniciando exploração de hardware para: {topology_id} ---")

        try:
            onnx_model_path = get_finn_ready_model(model_info, self.base_build_dir)
            if not onnx_model_path: return
        except Exception as e:
            print(f"[✗] Erro crítico na preparação do modelo: {e}"); return

        self.run_number = 1
        
        if num_builds != -1:
            safe_max_fps = target_fps if target_fps else 999999
            print(f"\n[MODO] DSE Esparso Dinâmico (Baseado em Valor).")
            print(f"       Range FPS: [{min_fps}, {safe_max_fps}]")
            print(f"       Num Builds: {num_builds}")
            print(f"       Salvar Builds: {save_builds}")
            
            self.run_sparse_dse(onnx_model_path, model_info, min_fps, safe_max_fps, num_builds, save_builds)
        else:
            print(f"\n[MODO] HARA Loop Clássico.")
            initial_folding, initial_hw_name = self._perform_first_run(onnx_model_path, topology_id, quant)
            if initial_folding:
                self.last_valid_folding = initial_folding
                self.last_valid_hw_name = initial_hw_name
                self.last_valid_build_dir = os.path.join(self.base_build_dir, initial_hw_name)
                self._perform_hara_loop(onnx_model_path, quant, topology_id)
                
    def _perform_first_run(self, onnx_model_path, topology_id, quant):
        print(f"\n🚀 [FIRST RUN] Gerando baseline de hardware para o modelo fornecido...")
        cfg_est = self.build_config['first_run_estimate']
        hw_name_est = utils.get_hardware_config_name(topology_id, quant, cfg_est['target_fps'], "_run0_estimate")
        result_est = self._run_single_build(onnx_model_path, hw_name_est, quant, topology_id, cfg_est['steps'], target_fps=cfg_est['target_fps'])
        if result_est.get('status') != 'success': return None, None

        initial_folding = utils.read_folding_config(result_est['build_dir'])
        intermediate_onnx_path = os.path.join(result_est['build_dir'], "intermediate_models", "step_generate_estimate_reports.onnx")
        if not os.path.exists(intermediate_onnx_path):
            if self.simulation_mode: intermediate_onnx_path = onnx_model_path
            else: return None, None
        
        current_folding = utils.reset_folding(initial_folding, intermediate_onnx_path, fixed_resources=self.fixed_resources)
        last_build_dir = result_est['build_dir']
        
        for i in range(45): # Balanceamento
            onnx_path_loop = os.path.join(last_build_dir, "intermediate_models/step_generate_estimate_reports.onnx")
            cycles_path = os.path.join(last_build_dir, "report/estimate_layer_cycles.json")
            if not os.path.exists(onnx_path_loop) or not os.path.exists(cycles_path):
                if self.simulation_mode: onnx_path_loop = onnx_model_path 
                else: break
            
            estimate_layer_cycles = {}
            if os.path.exists(cycles_path):
                with open(cycles_path, 'r') as f: estimate_layer_cycles = json.load(f)

            new_folding = utils.modify_folding(current_folding, onnx_path_loop, estimate_layer_cycles)
            if new_folding == current_folding: break
            current_folding = new_folding
            
            hw_name_balance = f"_run0_balance_iter{i+1}"
            folding_path_balance = os.path.join(self.base_build_dir, f"{hw_name_balance}.json")
            with open(folding_path_balance, "w") as f: json.dump(current_folding, f, indent=4)
            
            iter_res = self._run_single_build(onnx_model_path, hw_name_balance, quant, topology_id, cfg_est['steps'], folding_path=folding_path_balance)
            utils.clean_build_artifacts(iter_res['build_dir']) # Limpeza no balanceamento tb
            if not iter_res['build_dir']: break
            last_build_dir = iter_res['build_dir']

        hw_name_final = utils.get_hardware_config_name(topology_id, quant, None, "_run0_final_balanced")
        folding_path = os.path.join(self.base_build_dir, f"{hw_name_final}_folding_reset.json")
        with open(folding_path, "w") as f: json.dump(current_folding, f, indent=4)
            
        result_final = self._run_single_build(onnx_model_path, hw_name_final, quant, topology_id, self.build_config['first_run_build']['steps'], folding_path=folding_path, resource_limits=self.resource_limits_max)

        if result_final.get('status') == 'success': return result_final['folding'], result_final['hw_name']
        else: return None, None

    def _select_best_strategy(self, modify_func, folding_input, onnx_path, estimate_cycles, base_hw_name, quant, topology_id):
        folding_opt = modify_func(folding_input, onnx_path, estimate_cycles)
        if folding_opt == folding_input: return None, None
        return folding_opt, base_hw_name
        
    def _perform_hara_loop(self, onnx_model_path, quant, topology_id):
        max_runs_per_stage = self.hara_loop_config.get('max_runs_per_stage', -1)
        for modify_func_name in self.hara_loop_config['modify_functions']:
            modify_func = getattr(utils, modify_func_name)
            for percent in TARGET_RESOURCE_PERCENTAGES:
                current_limits = {k: int(v * percent) for k, v in self.resource_limits_max.items()}
                consecutive_errors = 0; runs_in_stage = 0
                max_errors = self.hara_loop_config['max_consecutive_errors']

                while consecutive_errors < max_errors:
                    if max_runs_per_stage != -1 and runs_in_stage >= max_runs_per_stage: break
                    last_folding = self.last_valid_folding
                    last_build_dir = self.last_valid_build_dir
                    onnx_path = os.path.join(last_build_dir, "intermediate_models/step_generate_estimate_reports.onnx")
                    cycles_path = os.path.join(last_build_dir, "report/estimate_layer_cycles.json")
                    if not (os.path.exists(onnx_path) and os.path.exists(cycles_path)): break
                    with open(cycles_path, 'r') as f: estimate_layer_cycles = json.load(f)

                    base_hw_name = utils.get_hardware_config_name(topology_id, quant, None, f"_run{self.run_number}")
                    new_folding, hw_name = self._select_best_strategy(modify_func, last_folding, onnx_path, estimate_layer_cycles, base_hw_name, quant, topology_id)
                    if new_folding is None: break
                    
                    folding_path = os.path.join(self.base_build_dir, f"{hw_name}_folding.json")
                    with open(folding_path, 'w') as f: json.dump(new_folding, f, indent=2)

                    result = self._run_single_build(onnx_model_path, hw_name, quant, topology_id, self.build_config['hara_build']['steps'], folding_path=folding_path, resource_limits=current_limits)

                    if result['status'] == 'success':
                        consecutive_errors = 0; self.last_valid_folding = result['folding']
                        self.last_valid_hw_name = result['hw_name']; self.last_valid_build_dir = result['build_dir']
                        utils.plot_area_usage_from_csv(self.summary_file, self.base_build_dir)
                    else:
                        consecutive_errors += 1
                    runs_in_stage += 1; self.run_number += 1