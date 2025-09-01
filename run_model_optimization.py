# run_model_optimization.py
import argparse
import os
from datetime import datetime

from config import TOPOLOGIES_TO_EXPLORE
from model_optimizer import ModelOptimizer
from utils.ml_utils import SatImgDataset

def create_main_build_dir(base_dir="hw/builds"):
    """Cria um diretório único para esta execução, baseado no timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    build_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(build_dir, exist_ok=True)
    print(f"Diretório principal da execução: {build_dir}")
    return build_dir

def main():
    parser = argparse.ArgumentParser(
        description="Fase 1: Encontra os melhores modelos de SW (.pth) que atendem às restrições de classificação."
    )
    # Este script não precisa mais de argumentos de modo.
    args = parser.parse_args()

    build_dir = create_main_build_dir()
    
    model_optimizer = ModelOptimizer(
        build_dir=build_dir,
        topologies_config=TOPOLOGIES_TO_EXPLORE
    )

    print("\n--- MODO: OTIMIZAÇÃO DE MODELO (GERANDO ARQUIVOS .pth) ---")
    for topology in TOPOLOGIES_TO_EXPLORE:
        print(f"\n{'='*58}\nIniciando Otimização de Modelo para Topologia: {topology['id']}\n{'='*58}")
        
        pytorch_model_path, _ = model_optimizer.find_optimal_model(topology)
        
        if pytorch_model_path:
            print(f"[✓] Processo concluído para Topologia {topology['id']}. Modelo ótimo salvo em: {pytorch_model_path}")
        else:
            print(f"[✗] Processo concluído para Topologia {topology['id']}. Não foi possível gerar um modelo que satisfaça as restrições.")
    
    print(f"\nOtimização de todos os modelos concluída. Os resultados estão no diretório: {build_dir}")

if __name__ == "__main__":
    main()