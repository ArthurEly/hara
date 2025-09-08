import argparse
import os
import json
import yaml
from datetime import datetime

# Importa a função que inicia a exploração de hardware
from run_hardware_exploration import start_exploration

def create_main_build_dir(base_dir="hw/builds"):
    """Cria um diretório de build único para a execução atual."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(base_dir, f"run_service_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Diretório principal da execução do serviço: {run_dir}")
    return run_dir

def main():
    parser = argparse.ArgumentParser(
        description="HARA Service Runner: Executa a exploração de hardware a partir de uma requisição JSON."
    )
    parser.add_argument(
        '--request',
        type=str,
        required=True,
        help="Caminho para o arquivo JSON de entrada com as restrições e o modelo."
    )
    args = parser.parse_args()

    # 1. Carregar e validar o JSON de requisição
    if not os.path.exists(args.request):
        raise FileNotFoundError(f"Arquivo de requisição não encontrado: {args.request}")
    with open(args.request, 'r') as f:
        request_data = json.load(f)
    
    model_id = request_data.get('model_id')
    area_constraints = request_data.get('area_constraints')

    if not model_id or not area_constraints:
        raise ValueError("O JSON de entrada deve conter os campos 'model_id' e 'area_constraints'.")

    # 2. Carregar o registro de modelos
    registry_path = 'models/registry_models.yaml'
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Arquivo de registro de modelos não encontrado: {registry_path}")
    with open(registry_path, 'r') as f:
        model_registry = yaml.safe_load(f)
        
    # 3. Encontrar as informações do modelo no registro
    model_info = model_registry.get(model_id)
    if not model_info:
        raise ValueError(f"O model_id '{model_id}' não foi encontrado no {registry_path}.")

    print("--- Requisição de Serviço Carregada ---")
    print(f"  Modelo: {model_id}")
    print(f"  Loader: {model_info.get('loader')}")
    print(f"  Limites de Recursos: {area_constraints}")
    print("-------------------------------------\n")

    # 4. Criar diretório de build e iniciar a exploração
    build_dir = create_main_build_dir()
    
    # Chama a função principal do fluxo de hardware
    start_exploration(
        build_dir=build_dir,
        model_info=model_info,
        resource_limits=area_constraints
    )

if __name__ == "__main__":
    main()