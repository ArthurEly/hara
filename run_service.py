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
        '--request',
        type=str,
        required=True,
        help="Caminho para o arquivo JSON de entrada."
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

    # Carrega a requisição e o registro
    with open(args.request, 'r') as f:
        request_data = json.load(f)
    model_id = request_data.get('model_id')
    area_constraints = request_data.get('area_constraints')
    
    with open('models/registry_models.yaml', 'r') as f:
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
        simulation_mode=args.simulation
    )

if __name__ == "__main__":
    main()