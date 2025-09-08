import os
from hardware_explorer import HardwareExplorer
from config import BUILD_CONFIG, HARA_LOOP_CONFIG

def start_exploration(build_dir, model_info, resource_limits):
    """
    Ponto de entrada para a exploração de hardware. Agora, apenas passa
    o dicionário model_info para o HardwareExplorer.
    """
    # A validação e extração de caminhos foi movida para as funções seguintes.
    hardware_explorer = HardwareExplorer(
        build_dir=build_dir,
        config=BUILD_CONFIG, 
        resource_limits=resource_limits,
        hara_loop_config=HARA_LOOP_CONFIG 
    )

    print("\n--- INICIANDO MÓDULO DE EXPLORAÇÃO DE HARDWARE ---")
    
    # Passa o dicionário model_info inteiro para a próxima etapa.
    hardware_explorer.run_exploration(model_info=model_info)
    
    model_id = model_info.get("topology_id", "modelo desconhecido")
    print(f"\nExploração de hardware para o modelo '{model_id}' concluída.")
    print(f"Os resultados estão no diretório: {build_dir}")