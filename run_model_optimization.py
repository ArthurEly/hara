# run_model_optimization.py
import argparse
import os
from datetime import datetime

# --- Configs ---
from config import (
    TOPOLOGIES_TO_EXPLORE, 
    TRAINING_CONFIG, 
    CLASSIFICATION_CONSTRAINTS, 
    FINETUNING_CONFIG
)
# --- Utils ---
from utils.ml_utils import Trainer, Pruner, Evaluator, load_dataloaders, SatImgDataset
# --- Nossas Classes Refatoradas ---
from training_logger import TrainingSummaryLogger
from model_optimizer import ModelOptimizer
# --- Classes de Modelo ---
from cnns_classes import CommonWeightQuant 

def create_main_build_dir(base_dir="sw/builds"): # <-- MUDANÇA AQUI
    """Cria um diretório único para esta execução, baseado no timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    build_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(build_dir, exist_ok=True)
    print(f"Diretório principal da execução: {build_dir}")
    return build_dir

def main():
    parser = argparse.ArgumentParser(
        description="Fase 1: Encontra os melhores modelos de SW (.pth)."
    )
    args = parser.parse_args()

    # --- 1. Montagem das Dependências ---
    build_dir = create_main_build_dir() # Esta função agora usa 'sw/builds'
    
    print("Montando dependências...")
    # Carrega os dados
    train_loader, test_loader, _ = load_dataloaders(
        TRAINING_CONFIG['batch_size']
    )
    
    # Instancia as classes de utilidade
    trainer = Trainer(train_loader, test_loader, device='cuda')
    evaluator = Evaluator(test_loader, device='cuda')
    pruner = Pruner()
    
    # Instancia o logger
    summary_path = os.path.join(build_dir, "training_summary.csv")
    logger = TrainingSummaryLogger(summary_path, CLASSIFICATION_CONSTRAINTS)

    # Define a configuração do quantizador que o Pruner precisa saber
    quantizer_cfg = {
        'weight': CommonWeightQuant
    }

    # --- 2. Injeção das Dependências ---
    model_optimizer = ModelOptimizer(
        build_dir=build_dir,
        trainer=trainer,
        evaluator=evaluator,
        pruner=pruner,
        logger=logger,
        quantizer_cfg=quantizer_cfg, 
        training_cfg=TRAINING_CONFIG,
        class_constraints=CLASSIFICATION_CONSTRAINTS,
        finetuning_cfg=FINETUNING_CONFIG
    )

    # --- 3. Execução da Lógica de Negócios ---
    print("\n--- MODO: OTIMIZAÇÃO DE MODELO (GERANDO ARQUIVOS .pth) ---")
    for topology in TOPOLOGIES_TO_EXPLORE:
        print(f"\n{'='*58}\nIniciando Otimização para: {topology['id']}\n{'='*58}")
        
        model_path, _ = model_optimizer.find_optimal_model(topology)
        
        if model_path:
            print(f"[✓] Concluído para {topology['id']}. Modelo salvo em: {model_path}")
        else:
            print(f"[✗] Concluído para {topology['id']}. Nenhum modelo viável encontrado.")
    
    print(f"\nOtimização concluída. Resultados em: {build_dir}")

if __name__ == "__main__":
    main()