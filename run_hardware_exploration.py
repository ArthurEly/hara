import os
import json
from hardware_explorer import HardwareExplorer
from config import BUILD_CONFIG, HARA_LOOP_CONFIG, HARDWARE_LEARNER_CONFIG
from ai.hardware_learner import HardwareLearner

def start_exploration(build_dir, model_info, resource_limits, simulation_mode=False, fpga_part="xc7z020clg400-1"):    
    request_path = os.path.join(build_dir, "request.json")
    fixed_resources = {}
    
    # Valores padrão
    req_target_fps = None
    req_min_fps = 0          
    req_num_builds = -1
    req_save_builds = True   

    if os.path.exists(request_path):
        print(f"-> Encontrado arquivo de requisição em: {request_path}")
        with open(request_path, 'r') as f:
            try:
                request_data = json.load(f)
                
                fixed_resources = request_data.get("fixed_resources", {})
                if fixed_resources == "auto": fixed_resources = {}
                
                # Leitura dos novos parâmetros
                req_target_fps = request_data.get("target_fps") 
                req_min_fps = request_data.get("min_fps", 0) 
                req_num_builds = request_data.get("num_builds", -1)
                req_save_builds = request_data.get("save_builds", True) 

                print(f"   -> Estratégia: Num Builds={req_num_builds}, Range=[{req_min_fps}, {req_target_fps}]")
                print(f"   -> Salvar Builds: {req_save_builds}")
                
            except json.JSONDecodeError:
                print(f"[AVISO] Falha ao decodificar {request_path}. Usando padrões.")
    else:
        print("[AVISO] request.json não encontrado.")

    # ---------------------------------------------------------------
    # Instancia o Hardware Learner se estiver habilitado
    # ---------------------------------------------------------------
    hardware_learner = None
    hl_cfg = HARDWARE_LEARNER_CONFIG
    if hl_cfg.get('enabled', False):
        model_path = hl_cfg.get('model_path')
        if model_path:
            hardware_learner = HardwareLearner()
            try:
                hardware_learner.load(model_path)
                print(f"[run] Hardware Learner carregado de: {model_path}")
            except FileNotFoundError as e:
                print(f"[run] AVISO: {e}. Continuando sem Hardware Learner.")
                hardware_learner = None
        else:
            print("[run] AVISO: HARDWARE_LEARNER_CONFIG.enabled=True mas model_path=None. "
                  "Continuando sem Hardware Learner.")

    hardware_explorer = HardwareExplorer(
        build_dir=build_dir,
        config=BUILD_CONFIG,
        resource_limits=resource_limits,
        hara_loop_config=HARA_LOOP_CONFIG,
        simulation_mode=simulation_mode,
        fixed_resources=fixed_resources,
        fpga_part=fpga_part,
        hardware_learner=hardware_learner,
    )

    print("\n--- INICIANDO MÓDULO DE EXPLORAÇÃO DE HARDWARE ---")
    
    hardware_explorer.run_exploration(
        model_info=model_info, 
        target_fps=req_target_fps,
        min_fps=req_min_fps,        
        num_builds=req_num_builds,
        save_builds=req_save_builds 
    )
    
    model_id = model_info.get("topology_id", "modelo desconhecido")
    print(f"\nExploração de hardware para o modelo '{model_id}' concluída.")
    print(f"Os resultados estão no diretório: {build_dir}")