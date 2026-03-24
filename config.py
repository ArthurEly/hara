# hara_config.py

# ==============================================================================
# CONFIGURAÇÕES DE TOPOLOGIA E QUANTIZAÇÃO
# ==============================================================================
# Mapeia o ID da topologia para a classe correspondente.
from cnns_classes import t1_quantizedCNN, t2_quantizedCNN, MobileNetWrapper

# --- Configurações de Performance de Classificação ---
CLASSIFICATION_CONSTRAINTS = {
    'enabled': True,  # Habilita ou desabilita a Fase 1
    'primary_metric': 'accuracy',  # pode ser 'accuracy', 'f1_score', 'precision', 'recall'
    'target_value': 0.99, # Ex: acurácia de 92%
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
        'method': 'percentage', 
        'step_percentage': 0.05 
    }
}

FINETUNING_CONFIG = {
    'enabled': True,
    'epochs': TRAINING_CONFIG['epochs_per_bitwidth'] // 2,
    'learning_rate': 1e-4,
    'patience_epochs': TRAINING_CONFIG['patience_epochs'] // 2,
    'max_pruning_finetuning_cycles': -1
}

TOPOLOGY_MAP = {
    "SAT6_T1": t1_quantizedCNN,
    "SAT6_T2": t2_quantizedCNN,
    "MOBILENET": MobileNetWrapper
}

# Define as topologias e os níveis de quantização a serem explorados.
TOPOLOGIES_TO_EXPLORE = [
    
    # --- TESTE 1: Impacto dos PESOS (W) ---
    # Mantém Ativações (A) em 1 bit e varia Pesos (W) de 1 a 8 bits.
    # O script irá gerar e testar: [1,1], [2,1], [3,1], ..., [8,1]
    {
        'id': "SAT6_T1",
        'tp_class': TOPOLOGY_MAP["SAT6_T1"],
        'quant_strategy': {
            'method': 'sweep',
            'sweep_target': 'weight',  # Varia 'w_start' até 'w_end'
            'start_bits': [1, 1],    # [w_start=1, a_start=1] (a_start é o bit de ativação fixo)
            'end_bits': [8, 8],      # [w_end=8, a_end=ignorado]
        },
        'quant_list': [ ]
    },

    # --- TESTE 2: Impacto das ATIVAÇÕES (A) ---
    # Mantém Pesos (W) em 1 bit e varia Ativações (A) de 1 a 8 bits.
    # O script irá gerar e testar: [1,1], [1,2], [1,3], ..., [1,8]
    {
        'id': "SAT6_T1", 
        'tp_class': TOPOLOGY_MAP["SAT6_T1"],
        'quant_strategy': {
            'method': 'sweep',
            'sweep_target': 'activation', # Varia 'a_start' até 'a_end'
            'start_bits': [1, 1],       # [w_start=1, a_start=1] (w_start é o bit de peso fixo)
            'end_bits': [8, 8],         # [w_end=ignorado, a_end=8]
        },
        'quant_list': [ ]
    },
    
    # Você pode adicionar blocos duplicados para a "SAT6_T2" se quiser testá-la também
    
    # --- TESTE 3: MobileNet ---
    {
        'id': "MOBILENET",
        'tp_class': TOPOLOGY_MAP["MOBILENET"],
        'quant_strategy': {
            'method': 'list',
            'quant_list': [
                [4, 4], # Test MobileNet with 4-bit weights and 4-bit activations
            ]
        }
    }
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