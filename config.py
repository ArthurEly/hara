# hara_config.py

# ==============================================================================
# CONFIGURAÇÕES DE TOPOLOGIA E QUANTIZAÇÃO
# ==============================================================================
# Mapeia o ID da topologia para a classe correspondente.
from cnns_classes import t1_quantizedCNN, t2_quantizedCNN

# --- Configurações de Performance de Classificação ---
CLASSIFICATION_CONSTRAINTS = {
    'enabled': True,  # Habilita ou desabilita a Fase 1
    'primary_metric': 'accuracy',  # pode ser 'accuracy', 'f1_score', 'precision', 'recall'
    'target_value': 0.9, # Ex: acurácia de 92%
    'metric_average_mode': 'macro', # 'macro', 'micro', 'weighted' para F1/precision/recall
    'prioritize_class': None # Ex: 2 (prioriza métricas para a classe 2)
}

# --- Configurações de Treino e Pruning ---
TRAINING_CONFIG = {
    'epochs_per_bitwidth': 50,
    'patience_epochs': 15,
    'learning_rate': 3e-5,
    'batch_size': 512,
    'pruning_enabled': False,
    
    'pruning_strategy': {
        # Método: 'iterative' (1 por 1, lento, preciso) ou 'percentage' (em lotes, rápido)
        'method': 'percentage', 
        
        # Valor usado apenas se o método for 'percentage'.
        # Remove 5% dos canais/neurônios restantes a cada passo de poda.
        'step_percentage': 0.05 
    }
}

FINETUNING_CONFIG = {
    'enabled': False,
    'epochs': 1,                # Um número menor de épocas é geralmente suficiente
    'learning_rate': 1e-4,       # Uma taxa de aprendizado menor para um ajuste fino
    'patience_epochs': 5,         # Paciência para o early stopping do fine-tuning
    'max_pruning_finetuning_cycles': -1
}

TOPOLOGY_MAP = {
    "SAT6_T1": t1_quantizedCNN,
    "SAT6_T2": t2_quantizedCNN
}

# Define as topologias e os níveis de quantização a serem explorados.
TOPOLOGIES_TO_EXPLORE = [
    {
        'id': "SAT6_T2",
        'tp_class': TOPOLOGY_MAP["SAT6_T2"],
        'quant_strategy': {
            # [CORRIGIDO] O método agora é 'sweep'
            'method': 'sweep',
            'sweep_target': 'both', # Pode ser 'weight', 'activation' ou 'both'
            'start_bits': [2, 2],
            'end_bits': [8, 8],
        },
        # Esta lista é ignorada quando o método é 'sweep'
        'quant_list': [ ]
    },
    #{
    #    'id': 1,
    #    'tp_class': TOPOLOGY_MAP[1],
    #    'quant_strategy': {
    #        # [CORRIGIDO] O método agora é 'sweep'
    #        'method': 'sweep',
    #        'sweep_target': 'both', # Pode ser 'weight', 'activation' ou 'both'
    #        'start_bits': [2, 2],
    #        'end_bits': [8, 8],
    #    },
    #    # Esta lista é ignorada quando o método é 'sweep'
    #    'quant_list': [ ]
    #},
]

# ==============================================================================
# CONFIGURAÇÕES DE RECURSOS E LIMITES
# ==============================================================================
# Define os limites máximos de recursos da placa alvo (ex: Pynq-Z1).
MAX_RESOURCES = {
    "Total LUTs": 134600,
    "FFs": 269200,
    "BRAM (36k)": 365,
    "DSP Blocks": 740
}

# Define os percentuais da placa a serem usados como teto em diferentes estágios.
# Por exemplo, o HARA tentará encontrar a melhor solução usando até 10% da placa,
# depois até 20%, e assim por diante.
TARGET_RESOURCE_PERCENTAGES = [1]

# ==============================================================================
# CONFIGURAÇÕES DOS PASSOS DE BUILD (STEPS) DO FINN
# ==============================================================================
# Passos apenas para estimativa (mais rápido)
ESTIMATE_ONLY_STEPS = [
    'step_qonnx_to_finn',
    'step_tidy_up',
    'step_streamline',
    'step_convert_to_hw',
    'step_create_dataflow_partition',
    'step_specialize_layers',
    'step_target_fps_parallelization',
    'step_apply_folding_config',
    'step_minimize_bit_width',
    'step_generate_estimate_reports',
]

# Passos para um build completo de hardware (lento)
FULL_HW_BUILD_STEPS = [
    'step_qonnx_to_finn',
    'step_tidy_up',
    'step_streamline',
    'step_convert_to_hw',
    'step_create_dataflow_partition',
    'step_specialize_layers',
    'step_target_fps_parallelization',
    'step_apply_folding_config',
    'step_minimize_bit_width',
    'step_generate_estimate_reports',
    'step_hw_codegen',
    'step_hw_ipgen',
    'step_set_fifo_depths',
    'step_create_stitched_ip',
    'step_out_of_context_synthesis'
    # Adicione ou remova passos conforme necessário
    # 'step_measure_rtlsim_performance',
    # 'step_synthesize_bitfile',
    # 'step_make_pynq_driver',
    # 'step_deployment_package'
]

# ==============================================================================
# CONFIGURAÇÕES GERAIS DO HARA
# ==============================================================================
# Agrupa as configurações de passos para fácil acesso
BUILD_CONFIG = {
    'first_run_estimate': {
        'steps': ESTIMATE_ONLY_STEPS,
        'target_fps': 1
    },
    'first_run_build': {
        'steps': FULL_HW_BUILD_STEPS,
        'target_fps': None
    },
    'check_strategy': {
        'steps': ESTIMATE_ONLY_STEPS,
        'target_fps': None
    },
    'hara_build': {
        'steps': FULL_HW_BUILD_STEPS,
        'target_fps': None
    }
}

# Parâmetros do loop de exploração HARA
HARA_LOOP_CONFIG = {
    'max_consecutive_errors': 1,
    'max_runs_per_stage': -1, # -1 para ilimitado
    'modify_functions': [
        "modify_folding",
    ]
}