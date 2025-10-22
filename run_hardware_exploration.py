import os
import json # <-- ADICIONADO
from hardware_explorer import HardwareExplorer
from config import BUILD_CONFIG, HARA_LOOP_CONFIG

def start_exploration(build_dir, model_info, resource_limits, simulation_mode=False, fpga_part="xc7z020clg400-1"):    
    """
    Ponto de entrada para a exploração de hardware.
    Agora lê o arquivo de requisição para obter configurações de recursos fixos
    e propaga o modo de simulação.
    """
    
    # --- NOVA LÓGICA PARA CARREGAR RECURSOS FIXOS ---
    request_path = os.path.join(build_dir, "request.json")
    fixed_resources = {}
    if os.path.exists(request_path):
        print(f"-> Encontrado arquivo de requisição em: {request_path}")
        with open(request_path, 'r') as f:
            try:
                request_data = json.load(f)
                fixed_resources = request_data.get("fixed_resources", {})
                if fixed_resources == "auto":
                    fixed_resources = {} # Trata 'auto' como um dicionário vazio
                print(f"   -> Configurações de recursos fixos carregadas: {fixed_resources}")
            except json.JSONDecodeError:
                print(f"[AVISO] Falha ao decodificar {request_path}. Usando configurações padrão.")
    else:
        print("[AVISO] request.json não encontrado no diretório de build. Usando configurações de recursos padrão.")
    # --- FIM DA NOVA LÓGICA ---
        
    hardware_explorer = HardwareExplorer(
        build_dir=build_dir,
        config=BUILD_CONFIG, 
        resource_limits=resource_limits,
        hara_loop_config=HARA_LOOP_CONFIG,
        simulation_mode=simulation_mode,
        fixed_resources=fixed_resources,
        fpga_part=fpga_part 
    )

    print("\n--- INICIANDO MÓDULO DE EXPLORAÇÃO DE HARDWARE ---")
    
    hardware_explorer.run_exploration(model_info=model_info)
    
    model_id = model_info.get("topology_id", "modelo desconhecido")
    print(f"\nExploração de hardware para o modelo '{model_id}' concluída.")
    print(f"Os resultados estão no diretório: {build_dir}")