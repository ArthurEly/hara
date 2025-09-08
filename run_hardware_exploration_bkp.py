# run_hardware_exploration.py
import argparse
import os

from hardware_explorer import HardwareExplorer
# A importação do ModelOptimizer foi removida.
# A importação do SatImgDataset é necessária para o torch.load funcionar.
from utils.ml_utils import SatImgDataset 
from config import BUILD_CONFIG, MAX_RESOURCES, HARA_LOOP_CONFIG

def main():
    parser = argparse.ArgumentParser(
        description="Fase 2: Executa a exploração de hardware do FINN a partir de um modelo treinado (.pth ou .onnx)."
    )
    
    # Os argumentos agora são todos obrigatórios para este script
    parser.add_argument(
        '--build-dir', type=str, required=True,
        help="Caminho para o diretório de build principal (criado pelo script da Fase 1)."
    )
    parser.add_argument(
        '--input-file', type=str, required=True,
        help="Caminho para o arquivo de modelo de entrada (.pth ou .onnx)."
    )
    parser.add_argument(
        '--topology-id', type=str, required=True,
        help="ID da topologia do modelo (ex: 'SAT6_T2')."
    )
    parser.add_argument(
        '--quant', type=int, required=True,
        help="Largura de bits de quantização do modelo (ex: 4)."
    )
    
    args = parser.parse_args()
    
    # Verifica se o diretório de build existe
    if not os.path.isdir(args.build_dir):
        raise FileNotFoundError(f"O diretório de build especificado não foi encontrado: {args.build_dir}")

    # Instancia o HardwareExplorer. Note que as configs são carregadas dentro da classe.
    # Você precisará ajustar a inicialização do HardwareExplorer para carregar as configs
    # do arquivo config.py, pois elas não são mais passadas aqui.
    hardware_explorer = HardwareExplorer(
        build_dir=args.build_dir,
        config=BUILD_CONFIG, 
        resource_limits=MAX_RESOURCES, 
        hara_loop_config=HARA_LOOP_CONFIG 
    )

    print("\n--- MODO: EXPLORAÇÃO DE HARDWARE (A PARTIR DE ARQUIVO) ---")
    hardware_explorer.run_exploration(
        model_path=args.input_file,
        quant=args.quant,
        topology_id=args.topology_id
    )
    
    print(f"\nExploração de hardware para {os.path.basename(args.input_file)} concluída.")
    print(f"Os resultados estão no subdiretório correspondente dentro de: {args.build_dir}")

if __name__ == "__main__":
    main()