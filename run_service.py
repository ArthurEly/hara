import argparse
import os
import json
import yaml
from run_hardware_exploration import start_exploration

def main():
    parser = argparse.ArgumentParser(
        description="HARA Runner: Executa a exploração a partir de uma requisição JSON."
    )
    parser.add_argument(
        '--build_dir',
        type=str,
        required=True,
        help="Caminho para o diretório de build, fornecido pelo servidor."
    )
    # --- NOVA FLAG ---
    parser.add_argument(
        '--simulation',
        action='store_true', # A presença da flag define seu valor como True
        help="Executa em modo de simulação, gerando dados sintéticos sem chamar o FINN."
    )
    args = parser.parse_args()
    
    request_file_path = os.path.join(args.build_dir, "request.json")
    # Agora você pode usar o request_file_path para carregar os dados
    with open(request_file_path, 'r') as f:
        request_data = json.load(f)

    # --- NOVO: Lendo o fpga_part ---
    # Usamos .get() com um valor padrão para não quebrar se o campo não existir
    fpga_part = request_data.get('fpga_part', 'xc7z020clg400-1') 

    model_id = request_data.get('model_id')
    area_constraints = request_data.get('area_constraints')
    
    with open('hara/models/registry_models.yaml', 'r') as f:
        model_registry = yaml.safe_load(f)
    model_info = model_registry.get(model_id)

    print("--- Requisição de Serviço Carregada ---")
    print(f"  Modelo: {model_id}")
    print(f"  Diretório de Build: {args.build_dir}")
    if args.simulation:
        print("  MODO: SIMULAÇÃO ATIVADO")
    print("-------------------------------------\n")

    # Passa a nova flag para a próxima função
    start_exploration(
        build_dir=args.build_dir,
        model_info=model_info,
        resource_limits=area_constraints,
        simulation_mode=args.simulation,
        fpga_part=fpga_part
    )

if __name__ == "__main__":
    main()