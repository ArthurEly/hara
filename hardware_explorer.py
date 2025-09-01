import os
import json
import time
from datetime import datetime
import torch

# Importa as ferramentas e as configurações
from utils.hw_utils import utils, export_to_onnx
# Importa SatImgDataset para que torch.load consiga carregar o modelo
from utils.ml_utils import SatImgDataset
from config import (
    MAX_RESOURCES,
    TARGET_RESOURCE_PERCENTAGES,
    BUILD_CONFIG,
    HARA_LOOP_CONFIG
)

class HardwareExplorer:
    """
    Gerencia a exploração de hardware (Fase 2) para encontrar a melhor
    configuração de folding para um dado modelo ONNX, respeitando os
    limites de recursos.
    """
    def __init__(self, build_dir, config, resource_limits, hara_loop_config):
        # Armazena as configurações
        self.base_build_dir = build_dir
        self.build_config = config
        self.resource_limits_max = resource_limits
        self.hara_loop_config = hara_loop_config
        
        # Inicializa o estado da exploração
        self.summary_file = os.path.join(self.base_build_dir, "hardware_summary.csv")
        self.run_number = 1
        
        # Estado que muda durante o loop
        self.last_valid_folding = None
        self.last_valid_hw_name = None
        self.last_valid_build_dir = None
        
        print(f"HardwareExplorer inicializado. Os resultados serão salvos em: {self.base_build_dir}")

    def _run_single_build(self, onnx_model_path, hw_name, quant, topology_id, steps, folding_path=None, target_fps=None, resource_limits=None):
        """
        Método centralizado que executa um único build de hardware.
        Agora também passa o caminho do modelo ONNX para o script de build.
        """
        print(f"-> Iniciando build para: {hw_name} usando {os.path.basename(onnx_model_path)}")
        
        args = [
            "python3", "run_build.py",
            "--model_path", str(onnx_model_path), # <--- NOVA FLAG
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
        Executa o build inicial para um dado ONNX para obter um baseline de folding.
        """
        print(f"\n🚀 [FIRST RUN] Gerando baseline de hardware para o modelo fornecido...")
        
        # 1. Estima o folding inicial
        cfg = self.build_config['first_run_estimate']
        hw_name_est = utils.get_hardware_config_name(topology_id, quant, cfg['target_fps'], "_run0_estimate")
        result = self._run_single_build(onnx_model_path, hw_name_est, quant, topology_id, cfg['steps'], target_fps=cfg['target_fps'])

        if result.get('status') != 'success':
            print("[✗] Falha ao gerar a estimativa inicial de folding.")
            return None, None

        # 2. Reseta o folding e executa o build completo
        initial_folding = utils.read_folding_config(result['build_dir'])
        reset_folding = utils.reset_folding(initial_folding, onnx_model_path)
        
        hw_name_final = utils.get_hardware_config_name(topology_id, quant, None, "_run0_final")
        folding_path = os.path.join(self.base_build_dir, f"{hw_name_final}_folding_reset.json")
        with open(folding_path, "w") as f:
            json.dump(reset_folding, f, indent=4)
            
        cfg_final = self.build_config['first_run_build']
        result_final = self._run_single_build(
            onnx_model_path, hw_name_final, quant, topology_id, cfg_final['steps'],
            folding_path=folding_path, resource_limits=self.resource_limits_max
        )

        if result_final.get('status') == 'success':
            print(f"[✓] Baseline de hardware gerado com sucesso: {hw_name_final}")
            return result_final['folding'], result_final['hw_name']
        else:
            print("[✗] Falha ao construir o hardware de baseline.")
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
        
    def _perform_hara_loop(self, quant, topology_id):
        """Executa o loop principal de exploração HARA."""
        
        for modify_func_name in self.hara_loop_config['modify_functions']:
            modify_func = getattr(utils, modify_func_name)
            print(f"\n🔧 [HARA] Usando a estratégia de modificação: {modify_func_name}")
            
            for percent in TARGET_RESOURCE_PERCENTAGES:
                current_limits = {k: int(v * percent) for k, v in self.resource_limits_max.items()}
                print(f"\n🎯 [HARA] Estágio com {percent*100:.0f}% dos recursos. Limites: {current_limits}")
                
                consecutive_errors = 0
                max_errors = self.hara_loop_config['max_consecutive_errors']

                while consecutive_errors < max_errors:
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
                        hw_name, quant, topology_id, cfg['steps'],
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

                    self.run_number += 1

    def run_exploration(self, model_path, quant, topology_id):
        """
        Método público que orquestra a exploração de hardware.
        Agora, ele primeiro converte o modelo .pth para .onnx, se necessário.
        """
        # --- ALTERAÇÃO PRINCIPAL ---
        # Verifica se o modelo de entrada é .pth ou .onnx
        if model_path.endswith(".pth"):
            print(f"-> Recebido modelo PyTorch: {os.path.basename(model_path)}. Convertendo para formato FINN ONNX...")
            # Carrega o modelo PyTorch salvo
            # A importação de SatImgDataset é necessária para o unpickling do torch
            model = torch.load(model_path)
            # Usa a função de hw_utils para converter e preparar o ONNX para o FINN
            onnx_model_path = export_to_onnx(model, self.base_build_dir, topology_id, quant)
            if not onnx_model_path:
                print("[✗] Falha ao converter o modelo PyTorch para ONNX. Abortando.")
                return

        elif model_path.endswith(".onnx"):
            print(f"-> Recebido modelo ONNX: {os.path.basename(model_path)}. Pulando a etapa de conversão.")
            onnx_model_path = model_path
        else:
            raise ValueError(f"Formato de arquivo de modelo inválido: {model_path}. Use .pth ou .onnx")

        # O resto do fluxo continua a partir daqui, usando o `onnx_model_path`
        self.run_number = 1
        self.last_valid_folding = None
        self.last_valid_hw_name = None
        
        initial_folding, initial_hw_name = self._perform_first_run(onnx_model_path, topology_id, quant)
        
        if initial_folding:
            self.last_valid_folding = initial_folding
            self.last_valid_hw_name = initial_hw_name
            self.last_valid_build_dir = os.path.join(self.base_build_dir, initial_hw_name)
            
            # O HARA loop já usa o onnx intermediário da última build, então não precisa de mudanças
            self._perform_hara_loop(quant, topology_id)
        else:
            print(f"[✗] Falha crítica na criação do baseline de hardware. Abortando exploração.")

if __name__ == "__main__":
    # Ponto de entrada do script
    explorer = HardwareExplorer(
        config=BUILD_CONFIG,
        resource_limits=MAX_RESOURCES,
        hara_loop_config=HARA_LOOP_CONFIG,
        topologies=TOPOLOGIES_TO_EXPLORE
    )
    explorer.run_exploration()