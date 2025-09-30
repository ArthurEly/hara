import os
import json
import time
import random

# Importa as ferramentas e as configurações
from utils.hw_utils import utils, get_finn_ready_model
from utils.ml_utils import SatImgDataset, load_pruned_model
from config import (
    TARGET_RESOURCE_PERCENTAGES,
    BUILD_CONFIG,
    HARA_LOOP_CONFIG
)

class HardwareExplorer:
    def __init__(self, build_dir, config, resource_limits, hara_loop_config, simulation_mode=False, fixed_resources=None):
        self.base_build_dir = build_dir
        self.build_config = config
        self.resource_limits_max = resource_limits
        self.hara_loop_config = hara_loop_config
        self.summary_file = os.path.join(self.base_build_dir, "hardware_summary.csv")
        self.run_number = 1
        self.last_valid_folding = None
        self.last_valid_hw_name = None
        self.last_valid_build_dir = None
        
        # --- MODIFICADO ---
        self.simulation_mode = simulation_mode
        self.fixed_resources = fixed_resources if fixed_resources else {}
        # --- FIM DA MODIFICAÇÃO ---

        print(f"HardwareExplorer inicializado. Os resultados serão salvos em: {self.base_build_dir}")
        if self.simulation_mode:
            print("[!] =================================================== [!]")
            print("[!] HExplorer ATIVADO EM MODO DE SIMULAÇÃO (DRY RUN) [!]")
            print("[!] NENHUM HARDWARE REAL SERÁ GERADO                  [!]")
            print("[!] =================================================== [!]")

    def _run_fake_build(self, onnx_model_path, hw_name, quant, topology_id, steps, folding_path=None, target_fps=None, resource_limits=None):
        """
        SIMULA um build do FINN, gerando os JSONs necessários e escrevendo
        DIRETAMENTE no hardware_summary.csv sem criar um arquivo .rpt.
        """
        import csv
        import datetime
        
        print(f"-> [SIMULAÇÃO DIRETA] Iniciando build FALSO para: {hw_name}")
        time.sleep(random.uniform(1.0, 2.0))
        
        build_output_dir = os.path.join(self.base_build_dir, hw_name)
        os.makedirs(os.path.join(build_output_dir, "report"), exist_ok=True)

        input_folding = {}
        if folding_path and os.path.exists(folding_path):
            with open(folding_path, 'r') as f:
                input_folding = json.load(f)
        
        with open(os.path.join(build_output_dir, "final_hw_config.json"), 'w') as f:
            json.dump(input_folding, f, indent=4)
        
        total_parallelism = 1.0
        for layer, config in input_folding.items():
            if layer != "Defaults":
                total_parallelism += config.get("PE", 1) * config.get("SIMD", 1)

        # 1. Gera os dados de recursos sintéticos em um dicionário
        base_luts = 15000 + (total_parallelism * 25)
        fake_luts = min(self.resource_limits_max["Total LUTs"], int(base_luts * (1 + random.uniform(-0.05, 0.05))))
        base_ffs = 20000 + (total_parallelism * 35)
        fake_ffs = min(self.resource_limits_max["FFs"], int(base_ffs * (1 + random.uniform(-0.05, 0.05))))
        base_bram = 15 + (total_parallelism * 0.05)
        fake_bram_36k_equiv = min(self.resource_limits_max["BRAM (36k)"], base_bram * (1 + random.uniform(-0.05, 0.05)))
        
        fake_area_data = {
            "Total LUTs": fake_luts, "Logic LUTs": int(fake_luts * 0.95),
            "LUTRAMs": int(fake_luts * 0.05), "SRLs": int(fake_luts * 0.05),
            "FFs": fake_ffs, "BRAM (36k)": round(fake_bram_36k_equiv, 1),
            "DSP Blocks": 0
        }

        # 2. Gera os JSONs de performance sintéticos (necessários para o loop HARA e para o CSV)
        fake_fps = 10 + total_parallelism * 0.5 * (1 + random.uniform(-0.05, 0.05))
        perf_report = {"estimated_throughput_fps": fake_fps, "max_cycles_node_name": "MVAU_hls_1_fake"}
        with open(os.path.join(build_output_dir, "report", "estimate_network_performance.json"), 'w') as f:
            json.dump(perf_report, f)
        
        fake_cycles = {name: int(random.uniform(1000, 50000)) for name in input_folding.keys() if name != "Defaults"}
        if fake_cycles:
            bottleneck_layer = random.choice(list(fake_cycles.keys()))
            fake_cycles[bottleneck_layer] = 100000
        with open(os.path.join(build_output_dir, "report", "estimate_layer_cycles.json"), 'w') as f:
            json.dump(fake_cycles, f)
            
        # 3. Verifica o status e monta a linha completa do CSV
        resource_diffs = utils.check_resource_usage(fake_area_data, resource_limits or {})
        exceeded = utils.raise_if_exceeds_limits(resource_diffs)
        status = "resources_exceeded" if exceeded else "success"
        
        summary_row = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hw_name": hw_name,
            "status": status,
            "duration_in_seconds": round(random.uniform(1.0, 2.0), 2),
            "folding_summary": json.dumps(input_folding),
            "folding_diff": json.dumps({}), # Simplificado para o modo fake
            "build_dir": build_output_dir,
            "resource_limits": json.dumps(resource_limits or {}),
            **fake_area_data, # Adiciona todos os dados de área ao dicionário
            **perf_report
        }
        
        # 4. Escreve a linha diretamente no hardware_summary.csv
        field_order = [
            "date", "hw_name", "status", "duration_in_seconds", "folding_summary",
            "folding_diff", "build_dir", "resource_limits", "Total LUTs", "Logic LUTs",
            "LUTRAMs", "SRLs", "FFs", "BRAM (36k)", "DSP Blocks",
            "estimated_throughput_fps", "max_cycles_node_name"
        ]
        file_exists = os.path.isfile(self.summary_file)
        with open(self.summary_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_order)
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary_row)

        if status == "success": print(f"[✓] [SIMULAÇÃO] Build {hw_name} concluído. Sumário salvo.")
        else: print(f"[!] [SIMULAÇÃO] Build {hw_name} concluído (recursos excedidos). Sumário salvo.")
        
        # Retorna o dicionário de status esperado pelo loop HARA
        return {"status": status, "folding": input_folding, "hw_name": hw_name, "build_dir": build_output_dir}

    def _run_single_build(self, onnx_model_path, hw_name, quant, topology_id, steps, folding_path=None, target_fps=None, resource_limits=None):
        """
        Método "roteador": decide se executa o build real do FINN ou a simulação.
        """
        if self.simulation_mode:
            return self._run_fake_build(onnx_model_path, hw_name, quant, topology_id, steps, folding_path, target_fps, resource_limits)

        # --- O CÓDIGO DO BUILD REAL PERMANECE O MESMO ---
        print(f"-> [REAL] Iniciando build para: {hw_name} usando {os.path.basename(onnx_model_path)}")
        
        args = [
            "python3", "run_build.py",
            "--model_path", str(onnx_model_path),
            "--build_dir", str(self.base_build_dir),
            "--topology", str(topology_id),
            "--quant", str(quant),
            "--steps", json.dumps(steps),
            "--hw_name", hw_name,
            "--folding_file", str(folding_path) if folding_path else "",
            "--target_fps", str(target_fps) if target_fps else "None",
            "--run", str(self.run_number)
        ]
        
        log_path = os.path.join(self.base_build_dir, f"build_{hw_name}.log")
        start_time = time.time()
        
        build_output_dir = os.path.join(self.base_build_dir, hw_name)
        folding_config = utils.read_folding_config(build_output_dir) if folding_path else {}

        try:
            utils.run_and_capture(args, log_path=log_path)
            duration = round(time.time() - start_time, 2)
            
            area_data = utils.extract_area_from_rpt(build_output_dir)
            resource_diffs = utils.check_resource_usage(area_data, resource_limits or {})
            exceeded = utils.raise_if_exceeds_limits(resource_diffs)
            
            status = "resources_exceeded" if exceeded else "success"
            if status == "success":
                print(f"[✓] Build {hw_name} concluído com sucesso em {duration}s.")
            else:
                 print(f"[!] Build {hw_name} concluído, mas excedeu os recursos: {exceeded}")

            folding_config = utils.read_folding_config(build_output_dir)
            utils.append_run_summary(
                self.summary_file, hw_name, status, folding_config,
                duration, build_output_dir, resource_limits or {}
            )
            return {"status": status, "folding": folding_config, "hw_name": hw_name, "build_dir": build_output_dir}

        except RuntimeError:
            duration = round(time.time() - start_time, 2)
            print(f"[✗] Build {hw_name} falhou (crash).")
            utils.save_crash_report(build_output_dir)
            utils.append_run_summary(
                self.summary_file, hw_name, "crash", folding_config,
                duration, build_output_dir, resource_limits or {}
            )
            return {"status": "crash"}

    def _perform_first_run(self, onnx_model_path, topology_id, quant):
        """
        Executa o build inicial para obter um baseline de hardware.
        AGORA inclui um loop de 5 iterações para balancear o uso de BRAM.
        """
        print(f"\n🚀 [FIRST RUN] Gerando baseline de hardware para o modelo fornecido...")
        
        # 1. Obtém a configuração de folding mais sequencial possível (reset_folding)
        cfg_est = self.build_config['first_run_estimate']
        hw_name_est = utils.get_hardware_config_name(topology_id, quant, cfg_est['target_fps'], "_run0_estimate")
        result_est = self._run_single_build(onnx_model_path, hw_name_est, quant, topology_id, cfg_est['steps'], target_fps=cfg_est['target_fps'])

        if result_est.get('status') != 'success':
            print("[✗] Falha ao gerar a estimativa inicial de folding.")
            return None, None

        initial_folding = utils.read_folding_config(result_est['build_dir'])
        intermediate_onnx_path = os.path.join(
            result_est['build_dir'], "intermediate_models", "step_generate_estimate_reports.onnx"
        )
        if not os.path.exists(intermediate_onnx_path):
            # Lógica de fallback para o modo de simulação
            if self.simulation_mode:
                intermediate_onnx_path = onnx_model_path
            else:
                print(f"[✗] ERRO CRÍTICO: O modelo ONNX intermediário não foi encontrado.")
                return None, None
        
        # --- MODIFICADO ---
        # Passa as configurações de recursos fixos para a função reset_folding
        current_folding = utils.reset_folding(
            initial_folding, intermediate_onnx_path, fixed_resources=self.fixed_resources
        )
        # --- FIM DA MODIFICAÇÃO ---

        last_build_dir = result_est['build_dir']

        # --- NOVO: LOOP DE BALANCEAMENTO DE BRAM ---
        print("\n--- Iniciando 5 iterações de balanceamento para reduzir BRAM ---")
        num_balance_runs = 19
        for i in range(num_balance_runs):
            print(f"-> Executando iteração de balanceamento #{i + 1}/{num_balance_runs}...")
            
            # Prepara os inputs para a função de modificação
            onnx_path_loop = os.path.join(last_build_dir, "intermediate_models/step_generate_estimate_reports.onnx")
            cycles_path = os.path.join(last_build_dir, "report/estimate_layer_cycles.json")
            if not os.path.exists(onnx_path_loop) or not os.path.exists(cycles_path):
                if self.simulation_mode: # Fallback para simulação
                    onnx_path_loop = onnx_model_path 
                else:
                    print("[!] Arquivos da última execução não encontrados. Interrompendo balanceamento.")
                    break
            
            # Carrega os ciclos apenas se o arquivo existir
            estimate_layer_cycles = {}
            if os.path.exists(cycles_path):
                with open(cycles_path, 'r') as f: estimate_layer_cycles = json.load(f)

            # Aumenta o paralelismo em um passo
            new_folding = utils.modify_folding(current_folding, onnx_path_loop, estimate_layer_cycles)
            
            if new_folding == current_folding:
                print("  -> Design já estável. Interrompendo loop de balanceamento.")
                break
            
            current_folding = new_folding
            
            # Executa um build de ESTIMATIVA (rápido) para atualizar os dados para a próxima iteração
            hw_name_balance = f"_run0_balance_iter{i+1}"
            folding_path_balance = os.path.join(self.base_build_dir, f"{hw_name_balance}.json")
            with open(folding_path_balance, "w") as f: json.dump(current_folding, f, indent=4)
            
            iter_build_dir = self._run_single_build(
                onnx_model_path, hw_name_balance, quant, topology_id, 
                cfg_est['steps'], folding_path=folding_path_balance
            )['build_dir']

            if not iter_build_dir:
                print("[✗] Falha na iteração de balanceamento. Usando a última configuração válida.")
                break
            
            last_build_dir = iter_build_dir
        # --- FIM DO LOOP ---

        print("\n--- Executando build final do baseline com a configuração balanceada ---")
        balanced_baseline_folding = current_folding
        hw_name_final = utils.get_hardware_config_name(topology_id, quant, None, "_run0_final_balanced")
        folding_path = os.path.join(self.base_build_dir, f"{hw_name_final}_folding_reset.json")
        with open(folding_path, "w") as f:
            json.dump(balanced_baseline_folding, f, indent=4)
            
        cfg_final = self.build_config['first_run_build']
        result_final = self._run_single_build(
            onnx_model_path, hw_name_final, quant, topology_id, cfg_final['steps'],
            folding_path=folding_path, resource_limits=self.resource_limits_max
        )

        if result_final.get('status') == 'success':
            print(f"[✓] Baseline de hardware balanceado gerado com sucesso: {hw_name_final}")
            return result_final['folding'], result_final['hw_name']
        else:
            print("[✗] Falha ao construir o hardware de baseline mesmo após o balanceamento.")
            # A lógica de trade-off BRAM->LUTRAM pode ser chamada aqui como um último recurso, se desejado.
            return None, None

    def _select_best_strategy(self, modify_func, folding_input, onnx_path, estimate_cycles, base_hw_name, quant, topology_id):
        """Testa estratégias de modificação (PE, SIMD, BOTH) e retorna a melhor."""
        # Esta função foi simplificada para refletir a lógica final do seu script original,
        # que focava em uma única estratégia de otimização por vez.
        # Pode ser expandida para testar PE vs SIMD se necessário.
        
        folding_opt = modify_func(folding_input, onnx_path, estimate_cycles)
        
        # Se não houver modificação, não há nada a fazer.
        if folding_opt == folding_input:
            return None, None
            
        return folding_opt, base_hw_name
        
    def _perform_hara_loop(self, onnx_model_path, quant, topology_id):
        """Executa o loop principal de exploração HARA."""
        
        max_runs_per_stage = self.hara_loop_config.get('max_runs_per_stage', -1)
        
        for modify_func_name in self.hara_loop_config['modify_functions']:
            modify_func = getattr(utils, modify_func_name)
            print(f"\n🔧 [HARA] Usando a estratégia de modificação: {modify_func_name}")
            
            for percent in TARGET_RESOURCE_PERCENTAGES:
                current_limits = {k: int(v * percent) for k, v in self.resource_limits_max.items()}
                print(f"\n🎯 [HARA] Estágio com {percent*100:.0f}% dos recursos. Limites: {current_limits}")
                
                consecutive_errors = 0
                runs_in_stage = 0
                
                max_errors = self.hara_loop_config['max_consecutive_errors']

                while consecutive_errors < max_errors:
                    if max_runs_per_stage != -1 and runs_in_stage >= max_runs_per_stage:
                        print(f"[i] Limite de {max_runs_per_stage} execuções por estágio atingido. Passando para o próximo estágio.")
                        break
                    # Carrega os dados da última execução válida
                    last_folding = self.last_valid_folding
                    last_build_dir = self.last_valid_build_dir
                    
                    onnx_path = os.path.join(last_build_dir, "intermediate_models/step_generate_estimate_reports.onnx")
                    cycles_path = os.path.join(last_build_dir, "report/estimate_layer_cycles.json")
                    if not (os.path.exists(onnx_path) and os.path.exists(cycles_path)):
                         print("[!] Arquivos essenciais da última build não encontrados. Encerrando estágio.")
                         break
                    with open(cycles_path, 'r') as f:
                        estimate_layer_cycles = json.load(f)

                    # Tenta modificar o folding
                    base_hw_name = utils.get_hardware_config_name(topology_id, quant, None, f"_run{self.run_number}")
                    new_folding, hw_name = self._select_best_strategy(modify_func, last_folding, onnx_path, estimate_layer_cycles, base_hw_name, quant, topology_id)

                    if new_folding is None:
                        print("[✓] Folding estável alcançado para esta estratégia/limite. Passando para o próximo estágio.")
                        break
                    
                    # Executa o build com a nova configuração
                    folding_path = os.path.join(self.base_build_dir, f"{hw_name}_folding.json")
                    with open(folding_path, 'w') as f:
                        json.dump(new_folding, f, indent=2)

                    cfg = self.build_config['hara_build']
                    result = self._run_single_build(
                        onnx_model_path, hw_name, quant, topology_id, cfg['steps'],
                        folding_path=folding_path, resource_limits=current_limits
                    )

                    # Processa o resultado do build
                    if result['status'] == 'success':
                        consecutive_errors = 0
                        self.last_valid_folding = result['folding']
                        self.last_valid_hw_name = result['hw_name']
                        self.last_valid_build_dir = result['build_dir']
                        utils.plot_area_usage_from_csv(self.summary_file, self.base_build_dir)
                    else:
                        consecutive_errors += 1
                        print(f"[!] Falha no build. Erros consecutivos: {consecutive_errors}/{max_errors}")

                    runs_in_stage += 1
                    self.run_number += 1

    def run_exploration(self, model_info):
        """
        Método público que orquestra a exploração de hardware.
        Recebe o dicionário model_info completo.
        """
        topology_id = model_info.get("topology_id")
        quant = model_info.get("quant")
        print(f"--- Iniciando exploração de hardware para: {topology_id} ---")

        try:
            # A função get_finn_ready_model agora recebe o dicionário inteiro
            # e sabe como lidar com cada caso (com ou sem model_path).
            onnx_model_path = get_finn_ready_model(model_info, self.base_build_dir)
            if not onnx_model_path:
                print("[✗] Falha ao preparar o modelo ONNX pronto para o FINN. Abortando.")
                return
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"[✗] Erro durante a preparação do modelo: {e}. Abortando.")
            return

        self.run_number = 1
        self.last_valid_folding = None
        self.last_valid_hw_name = None
        
        initial_folding, initial_hw_name = self._perform_first_run(onnx_model_path, topology_id, quant)
        
        if initial_folding:
            self.last_valid_folding = initial_folding
            self.last_valid_hw_name = initial_hw_name
            self.last_valid_build_dir = os.path.join(self.base_build_dir, initial_hw_name)
            
            self._perform_hara_loop(onnx_model_path, quant, topology_id)
        else:
            print(f"[✗] Falha crítica na criação do baseline de hardware. Abortando exploração.")