# test_model_optimizer.py
import unittest
from unittest.mock import Mock, MagicMock, patch, call
import torch
import sys
import os

# --- [FIX 1: CORREÇÃO DE PATH] ---
# Adiciona o diretório pai (onde estão os módulos como 'model_optimizer') ao sys.path
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, parent_dir)
# --- FIM DO FIX 1 ---

# Agora as importações da pasta pai funcionarão
from model_optimizer import ModelOptimizer

# Mockear a classe base do modelo (já que não queremos treinar um real)
class MockModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
    def parameters(self):
        return iter([torch.nn.Parameter(torch.randn(1, 1))])

# Mockear a classe de quantização que precisamos passar
class MockWeightQuant:
    pass

class TestModelOptimizer(unittest.TestCase):

    def setUp(self):
        # 1. Criar Mocks para todas as dependências
        self.mock_trainer = Mock()
        self.mock_evaluator = Mock()
        self.mock_pruner = Mock()
        self.mock_logger = Mock()

        # 2. Criar Configurações de Teste
        self.mock_training_cfg = {
            'epochs_per_bitwidth': 1, 'learning_rate': 1e-5, 'patience_epochs': 1,
            'pruning_enabled': False, 'pruning_strategy': {}
        }
        self.mock_class_constraints = {'enabled': True, 'primary_metric': 'accuracy', 'target_value': 0.9}
        self.mock_finetuning_cfg = {'enabled': False}
        
        # --- [FIX 2: CORREÇÃO DO CONSTRUTOR] ---
        # Cria o mock do 'quantizer_cfg' que estava faltando
        self.mock_quantizer_cfg = {
            'weight': MockWeightQuant 
        }
        # --- FIM DO FIX 2 ---

        # 3. Instanciar a Classe sob Teste (com mocks e o novo argumento)
        self.optimizer = ModelOptimizer(
            build_dir='test_build_dir',
            trainer=self.mock_trainer,
            evaluator=self.mock_evaluator,
            pruner=self.mock_pruner,
            logger=self.mock_logger,
            quantizer_cfg=self.mock_quantizer_cfg, # <-- Adicionado aqui
            training_cfg=self.mock_training_cfg,
            class_constraints=self.mock_class_constraints,
            finetuning_cfg=self.mock_finetuning_cfg
        )
        
        # 4. Mockear funções de sistema de arquivos
        self.patcher_save = patch('torch.save')
        self.patcher_open = patch('builtins.open', new_callable=unittest.mock.mock_open)
        self.patcher_yaml = patch('yaml.dump')
        self.patcher_arch = patch('model_optimizer.generate_arch_config')
        self.mock_save = self.patcher_save.start()
        self.mock_open = self.patcher_open.start()
        self.mock_yaml = self.patcher_yaml.start()
        self.mock_arch = self.patcher_arch.start()

    def tearDown(self):
        # Limpa os patches
        self.patcher_save.stop()
        self.patcher_open.stop()
        self.patcher_yaml.stop()
        self.patcher_arch.stop()
        # Remove o diretório pai do path para não sujar outros testes
        sys.path.pop(0)
        
    def test_01_generate_bit_width_sweep_both(self):
        """Testa se a estratégia 'sweep' 'both' gera a lista correta."""
        strategy = {'method': 'sweep', 'sweep_target': 'both', 'start_bits': [2, 2], 'end_bits': [4, 4]}
        expected = [[2, 2], [3, 3], [4, 4]]
        result = self.optimizer._generate_bit_width_tests(strategy)
        self.assertEqual(result, expected)

    def test_02_generate_bit_width_list(self):
        """Testa se a estratégia 'list' retorna a lista correta."""
        strategy = {'method': 'list', 'quant_list': [[4, 4], [8, 8]]}
        expected = [[4, 4], [8, 8]]
        result = self.optimizer._generate_bit_width_tests(strategy)
        self.assertEqual(result, expected)

    def test_03_find_optimal_model_stops_at_first_success(self):
        """
        Testa a lógica principal: deve parar no primeiro bit-width que
        atingir a meta de acurácia.
        """
        # Configurar Mocks
        mock_model_4bit = MockModel()
        mock_model_8bit = MockModel()
        
        # Simula o Trainer: retorna um modelo treinado
        self.mock_trainer.train.side_effect = [
            (mock_model_4bit, {}), # 1ª chamada (4bit)
            (mock_model_8bit, {})  # 2ª chamada (8bit)
        ]
        
        # Simula o Evaluator: 
        # A 1ª chamada (4bit) falha (0.8 < 0.9)
        # A 2ª chamada (8bit) passa (0.95 >= 0.9)
        self.mock_evaluator.evaluate.side_effect = [
            {'accuracy': 0.80}, # 1ª chamada
            {'accuracy': 0.95}  # 2ª chamada
        ]
        
        # Definir a topologia de teste
        test_topology = {
            'id': 'TEST_T1',
            'tp_class': MockModel, # Usa a classe MockModel
            'quant_strategy': {'method': 'list', 'quant_list': [[4, 4], [8, 8]]}
        }
        
        # Executar a função
        path, bits = self.optimizer.find_optimal_model(test_topology)
        
        # Verificar resultados
        self.assertEqual(bits, "8w8a") # Deve ter parado no 8bit
        
        # Deve ter chamado o treino 2 vezes (4bit e 8bit)
        self.assertEqual(self.mock_trainer.train.call_count, 2)
        
        # Deve ter chamado a avaliação 2 vezes
        self.assertEqual(self.mock_evaluator.evaluate.call_count, 2)
        
        # Deve ter salvado o modelo APENAS UMA VEZ (o de 8bit)
        self.mock_save.assert_called_once_with(mock_model_8bit.state_dict(), 'test_build_dir/pytorch_models/TEST_T1w8a8_final.pth')
        self.mock_logger.log_step.assert_has_calls([
            call(mock_model_4bit, 'TEST_T1', '4w4a', 'post-training', {'accuracy': 0.80}),
            call(mock_model_8bit, 'TEST_T1', '8w8a', 'post-training', {'accuracy': 0.95})
        ])

if __name__ == '__main__':
    # Isso permite rodar o teste diretamente da pasta /test
    # (ex: python3 test_model_optimizer.py)
    unittest.main()