import os
import json
import time
import random
import csv
import numpy as np

# Importa as ferramentas e as configurações
from utils.hw_utils import utils, get_finn_ready_model
from utils.analytic_utils import FinnCycleEstimator
from config import (
    TARGET_RESOURCE_PERCENTAGES,
    BUILD_CONFIG,
    HARA_LOOP_CONFIG
)

class HardwareExplorer:
    def __init__(self, build_dir, config, resource_limits, hara_loop_config, simulation_mode=False, fixed_resources=None, fpga_part="xc7z020clg400-1", hardware_learner=None):
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

        # Hardware Learner (opcional)
        self.hardware_learner = hardware_learner

        print(f"HardwareExplorer inicializado. Resultados em: {self.base_build_dir}")
        if self.hardware_learner and self.hardware_learner.is_loaded():
            print("[HardwareExplorer] Hardware Learner ativo.")

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

    # ------------------------------------------------------------------
    # Método: Mapeia design space teoricamente via FinnCycleEstimator
    # ------------------------------------------------------------------

    def _map_theoretical_design_space(self, onnx_path: str) -> list:
        """
        Gera analiticamente (sem builds) todos os estados válidos de folding
        usando o FinnCycleEstimator de utils/analytic_utils.py.

        Segue a heurística de paralelismo descrita no paper:
          - Estado inicial: todos os PEs e SIMDs = 1
          - A cada passo: aumenta SIMD da camada gargalo (maior latência)
          - Se SIMD chegou ao máximo: aumenta PE
          - Para quando o gargalo não pode mais ser reduzido (lat = 1 ciclo)

        Retorna:
            lista de {
              'folding': dict,
              'fps_estimated': float (100 MHz / ciclos_do_gargalo),
              'onnx_path': str
            }
        """
        print(f"\n📊 [ML] Mapeando design space teórico (FinnCycleEstimator)...")
        try:
            estimator = FinnCycleEstimator(onnx_path, debug=False)
        except Exception as e:
            print(f"   [!] FinnCycleEstimator falhou: {e}")
            return []

        cycle_formulas = estimator.get_cycle_formulas()
        if not cycle_formulas:
            print("   [!] Nenhuma camada de processamento encontrada no ONNX.")
            return []

        # Estado inicial de paralelismo
        simd_state = {name: 1 for name, d in cycle_formulas.items() if "SIMD" in d.get("formula", "")}
        pe_state   = {name: 1 for name, d in cycle_formulas.items() if "PE"   in d.get("formula", "")}
        for name in cycle_formulas:
            if "ConvolutionInputGenerator" in cycle_formulas[name].get("op_type", ""):
                cycle_formulas[name]["parallel_window"] = 0

        F_CLOCK = 100e6  # 100 MHz

        def _eval(formula, pe, simd):
            import math
            try:
                return eval(formula, {"__builtins__": None, "math": math}, {"PE": pe, "SIMD": simd})
            except Exception:
                return float("inf")

        def _compute_layer_cycles():
            cycles = {}
            for name, data in cycle_formulas.items():
                pe   = pe_state.get(name, 1)
                simd = simd_state.get(name, 1)
                cycles[name] = _eval(data["formula"], pe, simd)
            return cycles

        def _current_folding():
            """Monta o dict de folding com o estado atual de PE/SIMD."""
            fold = {"Defaults": {"PE": 1, "SIMD": 1}}
            for name in cycle_formulas:
                cfg = {}
                if name in pe_state:   cfg["PE"]   = pe_state[name]
                if name in simd_state: cfg["SIMD"] = simd_state[name]
                if cfg:
                    fold[name] = cfg
            return fold

        map_configs = []
        last_bottleneck = None
        last_cycles = float("inf")

        for _ in range(500):   # limite de segurança
            layer_cycles = _compute_layer_cycles()
            bottleneck, bot_cycles = max(layer_cycles.items(), key=lambda kv: kv[1])

            # Registra o estado atual
            fps_est = F_CLOCK / bot_cycles if bot_cycles > 0 else 0.0
            map_configs.append({
                "folding": _current_folding(),
                "fps_estimated": fps_est,
                "onnx_path": onnx_path,
            })

            # Critério de parada
            if bot_cycles <= 1:
                break
            if bottleneck == last_bottleneck and abs(bot_cycles - last_cycles) < 1e-6:
                break  # gargalo não mudou — sem mais progressão

            last_bottleneck = bottleneck
            last_cycles = bot_cycles

            # Tenta aumentar SIMD do gargalo
            data = cycle_formulas[bottleneck]
            if bottleneck in simd_state:
                nxt = estimator._find_next_valid_parallelism(
                    bottleneck, simd_state[bottleneck], data["op_type"], data, "SIMD")
                if nxt > simd_state[bottleneck]:
                    simd_state[bottleneck] = nxt
                    continue
                # SIMD chegou ao limite: ativa parallel_window se conv
                if "ConvolutionInputGenerator" in data.get("op_type", "") and data.get("parallel_window", 0) == 0:
                    cycle_formulas[bottleneck]["parallel_window"] = 1
                    continue

            # Tenta aumentar PE do gargalo
            if bottleneck in pe_state:
                nxt = estimator._find_next_valid_parallelism(
                    bottleneck, pe_state[bottleneck], data["op_type"], data, "PE")
                if nxt > pe_state[bottleneck]:
                    pe_state[bottleneck] = nxt
                    continue

            # Não há mais otimização possível para o gargalo
            break

        print(f"   -> {len(map_configs)} estados mapeados. FPS range: "
              f"[{map_configs[0]['fps_estimated']:.0f}, {map_configs[-1]['fps_estimated']:.0f}]")
        return map_configs

    # ------------------------------------------------------------------
    # Método: Loop ML-guided principal (Hardware Explorer do paper)
    # ------------------------------------------------------------------

    def _perform_ml_guided_loop(self, onnx_path: str, topology_id: str, quant):
        """
        Implementa o loop ML-guided descrito no paper:
          1. Map  — gera todo o design space via FinnCycleEstimator
          2. Query — prediz recursos e seleciona o candidato de maior FPS dentro do budget
          3. Build — executa o build real
          4. Discrepancy — calcula a diferença predito vs. real e identifica o dominant resource
          5. Fine-tune — re-treina o modelo com o novo ponto (peso elevado)
          6. Refinamento — queries iterativas para ↑/↓ paralelismo até boundary condition
        """
        learner = self.hardware_learner
        if learner is None or not learner.is_loaded():
            print("[ML] Hardware Learner não disponível. Usando loop clássico.")
            self._perform_hara_loop(onnx_path, quant, topology_id)
            return

        cfg_build = self.build_config['hara_build']

        # ----------------------------------------------------------------
        # Passo 1: Mapear design space teórico
        # ----------------------------------------------------------------
        map_configs = self._map_theoretical_design_space(onnx_path)
        if not map_configs:
            print("[ML] Mapa vazio. Abortando exploracão ML-guided.")
            return

        all_foldings  = [c["folding"]       for c in map_configs]
        all_fps_est   = [c["fps_estimated"] for c in map_configs]

        print(f"\n🤖 [ML] Iniciando Hardware Explorer ML-guided.")
        print(f"   Design space: {len(map_configs)} configurações | "
              f"FPS estimado: [{all_fps_est[0]:.0f}, {all_fps_est[-1]:.0f}]")

        # ----------------------------------------------------------------
        # Passo 2: Query — prediz recursos para todo o design space
        # ----------------------------------------------------------------
        print("\n🔍 [ML] Passo 2: Query do Hardware Learner...")
        predictions = learner.predict(onnx_path, all_foldings)

        # ----------------------------------------------------------------
        # Passo 3: Seleciona o candidato de maior FPS dentro do budget
        # ----------------------------------------------------------------
        best_idx, best_folding, best_pred = learner.select_best_config(
            predictions, all_foldings, self.resource_limits_max
        )

        if best_idx == -1:
            print("[ML] Nenhuma config predita cabe no budget. "
                  "Iniciando pelo índice 0 (configuração mínima).")
            best_idx     = 0
            best_folding = all_foldings[0]
            best_pred    = predictions[0]

        # Guarda índice atual no mapa para navegar para cima/baixo depois
        current_idx = best_idx
        optimized_build_dir = None
        optimized_folding   = None

        # ----------------------------------------------------------------
        # Loop de refinamento (boundary condition)
        # ----------------------------------------------------------------
        iteration = 0
        prev_fit   = None   # True = úlltimo build coube; False = excedeu
        prev_idx   = None

        while True:
            iteration += 1
            folding_to_build = all_foldings[current_idx]
            fps_to_build     = all_fps_est[current_idx]

            print(f"\n🚀 [ML] Build #{iteration} — Index #{current_idx} "
                  f"({fps_to_build:.0f} FPS estimado)")

            # Escreve o folding e dispara o build
            hw_name = f"ml_guided_iter{iteration}_idx{current_idx}"
            folding_path = os.path.join(self.base_build_dir, f"{hw_name}_folding.json")
            with open(folding_path, "w") as f:
                json.dump(folding_to_build, f, indent=2)

            result = self._run_single_build(
                onnx_path, hw_name, quant, topology_id,
                cfg_build["steps"],
                folding_path=folding_path,
                resource_limits=self.resource_limits_max,
            )
            self.run_number += 1
            build_dir = result.get("build_dir", "")

            # ----------------------------------------------------------------
            # Passo 4: Extrair recursos reais e calcular discrepância
            # ----------------------------------------------------------------
            actual_resources = {}
            if result["status"] in ("success", "resources_exceeded"):
                extracted = utils.extract_area_from_rpt(build_dir)
                if extracted:
                    actual_resources = extracted
                else:
                    # Em simulação ou se o rpt não existir, usa valores do result
                    for k in ["Total LUTs", "FFs", "BRAM (36k)", "DSP Blocks"]:
                        actual_resources[k] = result.get(k, 0)
            elif result["status"] == "crash":
                # Em crash, assume utilização 200% do dominante do predito
                for k in ["Total LUTs", "FFs", "BRAM (36k)", "DSP Blocks"]:
                    actual_resources[k] = self.resource_limits_max.get(k, 0) * 2.0

            # Calcula discrepância apenas se temos os recursos reais
            dominant = None
            if actual_resources:
                disc = learner.calculate_discrepancy(
                    predictions[current_idx] if current_idx < len(predictions) else best_pred,
                    actual_resources,
                    self.resource_limits_max,
                )
                dominant = disc["dominant_resource"]
                dominant_pct = disc["dominant_utilization_pct"]

                # ----------------------------------------------------------------
                # Passo 5: Fine-tune com o novo ponto real (peso elevado)
                # ----------------------------------------------------------------
                learner.fine_tune(
                    onnx_path,
                    folding_to_build,
                    actual_resources,
                    current_build_weight=10.0,
                )

            # ----------------------------------------------------------------
            # Passo 6: Decidir a próxima direção e verificar boundary condition
            # ----------------------------------------------------------------
            build_fits = (result["status"] == "success")

            if build_fits:
                # Salva como melhor acelerador válido até agora
                optimized_build_dir = build_dir
                optimized_folding   = folding_to_build
                print(f"   ✅ [ML] Build coube! Acelerador salvo. Tentando maior paralelismo...")

                # Boundary condition: prev build não coube e este coube
                if prev_fit is False and abs(current_idx - prev_idx) <= 1:
                    print("   🎉 [ML] Boundary condition atingida! Configuração ótima encontrada.")
                    break

                next_idx = current_idx + 1
            else:
                print(f"   ❌ [ML] Build excedeu recursos. Reduzindo paralelismo...")

                # Boundary condition: prev build coube e este não coube
                if prev_fit is True and abs(current_idx - prev_idx) <= 1:
                    print("   🎉 [ML] Boundary condition atingida! Configuração ótima encontrada.")
                    break

                next_idx = current_idx - 1

            # Verifica limites do mapa
            if next_idx < 0 or next_idx >= len(map_configs):
                print(f"   [ML] Limite do design space atingido (idx={next_idx}).")
                break

            # Re-query com o modelo fine-tunado para o próximo índice
            if actual_resources:  # só re-queremos se temos modelo atualizado
                next_preds = learner.predict(onnx_path, [all_foldings[next_idx]])
                predictions[next_idx] = next_preds[0]

            prev_fit = build_fits
            prev_idx = current_idx
            current_idx = next_idx

            utils.plot_area_usage_from_csv(self.summary_file, self.base_build_dir)

        # ----------------------------------------------------------------
        # Resultado final
        # ----------------------------------------------------------------
        if optimized_folding is not None:
            print(f"\n✅ [ML] Hardware Explorer concluído. Acelerador ótimo em: {optimized_build_dir}")
        else:
            print("\n⚠ [ML] Nenhuma configuração válida encontrada.")

        self.last_valid_folding  = optimized_folding
        self.last_valid_build_dir = optimized_build_dir

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

    def run_exploration(self, model_info, target_fps=None, min_fps=0, num_builds=-1, save_builds=True, use_ml_learner=None):
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

        # Determina o modo de exploração
        # use_ml_learner=None -> auto: usa ML-guided se learner estiver carregado
        ml_active = use_ml_learner if use_ml_learner is not None else (
            self.hardware_learner is not None and self.hardware_learner.is_loaded()
        )

        if ml_active:
            print(f"\n[MODO] Hardware Explorer ML-guided (Hardware Learner ativo).")
            self._perform_ml_guided_loop(onnx_model_path, topology_id, quant)
        elif num_builds != -1:
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