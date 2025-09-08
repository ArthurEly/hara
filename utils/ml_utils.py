# utils/ml_utils.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import csv
import onnx
from onnx import helper
import shutil
import copy
import io
import brevitas.nn as qnn
import yaml
import importlib 
from config import TOPOLOGY_MAP
from collections import defaultdict

# Imports específicos de Brevitas e FINN/QONNX
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp

# Transformações Gerais
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs, RemoveUnusedTensors
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.insert_topk import InsertTopK

def get_model_size(model):
    """Calcula e retorna o número de parâmetros e o tamanho em MB de um modelo."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Calcula o tamanho em MB salvando o state_dict em um buffer de memória
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.tell() / (1024 * 1024)
    return num_params, size_mb

def generate_arch_config(model):
    """
    Inspeciona um modelo PyTorch/Brevitas (podado ou não) e gera um
    dicionário descrevendo sua arquitetura (o número de unidades de saída
    de cada camada podável).
    """
    config = {}
    # Itera sobre todas as camadas nomeadas do modelo
    for name, module in model.named_modules():
        # Se a camada for convolucional, salva o número de canais de saída
        if isinstance(module, qnn.QuantConv2d):
            # A chave será 'conv1_out', 'conv2_out', etc.
            config[f"{name}_out"] = module.out_channels
        # Se a camada for linear, salva o número de features de saída (neurônios)
        elif isinstance(module, qnn.QuantLinear):
            # A chave será 'fc1_out', etc.
            config[f"{name}_out"] = module.out_features
    return config

def load_pruned_model(model_pth_path):
    """
    Carrega um modelo podado a partir de um par de arquivos .yaml (arquitetura) e .pth (pesos).
    """
    # --- CORREÇÃO APLICADA ---
    # Remove a extensão do caminho de entrada para obter o nome base do arquivo.
    base_path, _ = os.path.splitext(model_pth_path)
    
    # Monta os caminhos corretos para os arquivos .yaml e .pth.
    yaml_path = f"{base_path}.yaml"
    pth_path = f"{base_path}.pth" # Garante que o caminho .pth está correto
    
    # 1. Carrega a configuração da arquitetura do arquivo YAML
    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)
        
    topology_id = metadata['topology_id']
    arch_config = metadata['arch_config']
    w_bits = metadata['bit_widths']['weight']
    a_bits = metadata['bit_widths']['activation']
            
    # 3. Obtém a classe do modelo e instancia com a config completa
    model_class = TOPOLOGY_MAP.get(topology_id)
    if not model_class:
        raise ValueError(f"ID de topologia '{topology_id}' não encontrado.")
            
    model = model_class(
        weight_bit_width=w_bits,
        act_bit_width=a_bits,
        arch_config=arch_config
    )
    
    # 4. Carrega os pesos do arquivo .pth
    state_dict = torch.load(pth_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    print(f"Modelo da topologia {topology_id} com {w_bits}w{a_bits}a carregado com sucesso.")
    return model

# Classe SatImgDataset precisa ser definida aqui para que o pickle.load funcione corretamente
class SatImgDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.transform = T.ToTensor()
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        x = self.transform(self.X[index])
        y = torch.FloatTensor(self.y[index])
        return {'x':x, 'y':y}


def load_dataloaders(batch_size, data_dir="./"):
    """ Carrega os datasets em formato pickle e retorna DataLoaders. """
    train_path = os.path.join(data_dir, "train_dataset.pkl")
    test_path = os.path.join(data_dir, "test_dataset.pkl")

    with open(train_path, 'rb') as f:
        dataset_train = pickle.load(f)
    with open(test_path, 'rb') as f:
        dataset_test = pickle.load(f)

    loader_train = DataLoader(dataset_train, batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size, shuffle=False)
    
    print(f"Dataset de treino carregado com {len(dataset_train)} amostras.")
    print(f"Dataset de teste carregado com {len(dataset_test)} amostras.")
    
    return loader_train, loader_test, len(dataset_test)

class Evaluator:
    """ Classe responsável por avaliar a performance de classificação de um modelo. """
    def __init__(self, test_loader, device='cpu'):
        self.loader = test_loader
        self.device = device

    def evaluate(self, model, constraints):
        """ Calcula acurácia, precisão, recall e f1-score. """
        model.to(self.device)
        model.eval()
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.loader:
                inputs = batch['x'].to(self.device)
                labels = batch['y'].to(self.device)
                
                outputs = model(inputs)
                
                predicted_classes = torch.max(outputs, 1)[1]
                real_classes = torch.max(labels, 1)[1]
                
                all_preds.extend(predicted_classes.cpu().numpy())
                all_labels.extend(real_classes.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        
        # Gera as outras métricas com o modo de média especificado na configuração
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, 
            all_preds, 
            average=constraints['metric_average_mode'],
            zero_division=0
        )
        
        report = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }
        
        # Se uma classe específica for priorizada, calcula as métricas para ela também
        if constraints['prioritize_class'] is not None:
            p_class, r_class, f1_class, _ = precision_recall_fscore_support(
                all_labels, all_preds, labels=[constraints['prioritize_class']], average=None, zero_division=0
            )
            report[f'precision_class_{constraints["prioritize_class"]}'] = p_class[0]
            report[f'recall_class_{constraints["prioritize_class"]}'] = r_class[0]
            report[f'f1_score_class_{constraints["prioritize_class"]}'] = f1_class[0]

        return report


class Trainer:
    # ... (método __init__ permanece o mesmo) ...
    def __init__(self, train_loader, test_loader, device='cpu'):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.evaluator = Evaluator(self.test_loader, device=self.device)

    # --- MÉTODO ATUALIZADO ---
    def train(self, model, epochs, lr, patience, constraints=None, stop_on_target_met=False):
        """
        Treina o modelo.
        O novo flag 'stop_on_target_met' controla se o treino deve parar
        assim que a meta de performance das 'constraints' for atingida.
        """
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_iterator = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Treino]")
            
            for batch in train_iterator:
                images = batch['x'].to(self.device)
                labels = batch['y'].to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                train_iterator.set_postfix({'loss': loss.item()})

            avg_train_loss = train_loss / len(self.train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in self.test_loader:
                    images = batch['x'].to(self.device)
                    labels = batch['y'].to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(self.test_loader)

            perf_report = self.evaluator.evaluate(model, constraints)
            primary_metric = constraints['primary_metric']
            current_performance = perf_report[primary_metric]

            print(
                f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val {primary_metric}: {current_performance:.4f}"
            )

            # --- [LÓGICA ATUALIZADA] ---
            # A parada antecipada por performance só acontece se o flag for True.
            if stop_on_target_met and constraints and current_performance >= constraints['target_value']:
                print(f"[✓] Meta de performance ({constraints['target_value']:.4f}) alcançada durante o fine-tuning! Parando.")
                return model, perf_report

            # Lógica de early stopping baseada em Val Loss (patience)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                print(" -> Melhor modelo encontrado. Salvando estado.")
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f" -> Early stopping após {patience} épocas sem melhora na perda de validação.")
                    if best_model_state:
                        model.load_state_dict(best_model_state)
                    break
        
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        final_report = self.evaluator.evaluate(model, constraints)
        return model, final_report

class Pruner:
    """
    Pruner genérico que recebe as configurações de quantização como parâmetro,
    tornando-o agnóstico à topologia do modelo.
    """
    
    def _log_model_size(self, model, stage_name):
        # ... (sem alterações)
        num_params, size_mb = get_model_size(model)
        print(f"[{stage_name}] Parâmetros: {num_params:,} | Tamanho Aprox.: {size_mb:.2f} MB")

    def _get_prunable_layers_info(self, model):
        # ... (sem alterações)
        prunable_layers = []
        for name, module in model.named_modules():
            # Procura por camadas Quantizadas da Brevitas
            if isinstance(module, (qnn.QuantConv2d, qnn.QuantLinear)):
                prunable_layers.append((name, module))
        return prunable_layers[:-1]

    # --- [MÉTODO ATUALIZADO] ---
    def _create_pruned_model(self, original_model, layer_to_prune_name, unit_to_prune_idx, quantizer_cfg):
        """
        Cria uma cópia do modelo com uma unidade removida, usando as classes de
        quantização fornecidas via 'quantizer_cfg'.
        """
        model = copy.deepcopy(original_model)
        # ... (lógica para encontrar layer_to_prune e next_layer é a mesma) ...
        all_layers = dict(model.named_modules())
        layer_to_prune = all_layers[layer_to_prune_name]
        next_layer = None
        layer_iterator = iter(model.named_modules())
        for name, _ in layer_iterator:
            if name == layer_to_prune_name:
                try:
                    while True:
                        next_layer_name, next_layer_module = next(layer_iterator)
                        if isinstance(next_layer_module, (qnn.QuantConv2d, qnn.QuantLinear)):
                            next_layer = next_layer_module
                            break
                except StopIteration: break
        if next_layer is None: raise RuntimeError(f"Não foi possível encontrar a camada seguinte à {layer_to_prune_name} para ajustar.")

        # Obtém a classe do quantizador de pesos a partir da configuração
        weight_quant_class = quantizer_cfg['weight']
        
        # ... (lógica para keep_mask e new_num_units é a mesma) ...
        if isinstance(layer_to_prune, qnn.QuantConv2d):
            num_units = layer_to_prune.out_channels
        elif isinstance(layer_to_prune, qnn.QuantLinear):
            num_units = layer_to_prune.out_features
        else: return original_model
        keep_mask = torch.ones(num_units, dtype=torch.bool); keep_mask[unit_to_prune_idx] = False
        keep_indices = torch.where(keep_mask)[0]; new_num_units = len(keep_indices)

        
        bit_width = int(layer_to_prune.weight_quant.bit_width().item())

        if isinstance(layer_to_prune, qnn.QuantConv2d):
            new_layer = qnn.QuantConv2d(
                # ... (parâmetros geométricos) ...
                in_channels=layer_to_prune.in_channels, out_channels=new_num_units,
                kernel_size=layer_to_prune.kernel_size, stride=layer_to_prune.stride,
                padding=layer_to_prune.padding, bias=layer_to_prune.bias is not None,
                weight_quant=weight_quant_class,
                weight_bit_width=bit_width
            )
            # ... (cópia de pesos) ...
            new_layer.weight.data = layer_to_prune.weight.data[keep_indices, :, :, :]
            if layer_to_prune.bias is not None: new_layer.bias.data = layer_to_prune.bias.data[keep_indices]
        
        elif isinstance(layer_to_prune, qnn.QuantLinear):
            new_layer = qnn.QuantLinear(
                # ... (parâmetros geométricos) ...
                in_features=layer_to_prune.in_features, out_features=new_num_units,
                bias=layer_to_prune.bias is not None,
                # --- [CORREÇÃO] Usa a classe da configuração ---
                weight_quant=weight_quant_class,
                weight_bit_width=bit_width
            )
            # ... (cópia de pesos) ...
            new_layer.weight.data = layer_to_prune.weight.data[keep_indices, :]
            if layer_to_prune.bias is not None: new_layer.bias.data = layer_to_prune.bias.data[keep_indices]

        next_layer_bit_width = int(next_layer.weight_quant.bit_width().item())

        # Ajusta a próxima camada
        if isinstance(next_layer, qnn.QuantConv2d):
            new_next_layer = qnn.QuantConv2d(
                # ... (parâmetros) ...
                in_channels=new_num_units, out_channels=next_layer.out_channels,
                kernel_size=next_layer.kernel_size, stride=next_layer.stride,
                padding=next_layer.padding, bias=next_layer.bias is not None,
                # --- [CORREÇÃO] Usa a classe da configuração ---
                weight_quant=weight_quant_class,
                weight_bit_width=next_layer_bit_width
            )
            # ... (cópia de pesos) ...
            new_next_layer.weight.data = next_layer.weight.data[:, keep_indices, :, :]
            if next_layer.bias is not None: new_next_layer.bias.data = next_layer.bias.data
        elif isinstance(next_layer, qnn.QuantLinear):
            # ... (lógica para calcular new_in_features) ...
            if isinstance(layer_to_prune, qnn.QuantConv2d):
                features_per_unit = next_layer.in_features // layer_to_prune.out_channels
                new_in_features = new_num_units * features_per_unit
                keep_features_mask_indices = []
                for i in range(layer_to_prune.out_channels):
                    if i in keep_indices.tolist():
                        start_idx = i * features_per_unit
                        keep_features_mask_indices.extend(range(start_idx, start_idx + features_per_unit))
            else: new_in_features = new_num_units; keep_features_mask_indices = keep_indices
            
            new_next_layer = qnn.QuantLinear(
                in_features=new_in_features, out_features=next_layer.out_features,
                bias=next_layer.bias is not None,
                # --- [CORREÇÃO] Usa a classe da configuração ---
                weight_quant=weight_quant_class,
                weight_bit_width=next_layer_bit_width
            )
            # ... (cópia de pesos) ...
            new_next_layer.weight.data = next_layer.weight.data[:, keep_features_mask_indices]
            if next_layer.bias is not None: new_next_layer.bias.data = next_layer.bias.data
        
        # ... (lógica de substituição de camadas) ...
        def set_nested_attr(obj, attr_string, value):
            parts = attr_string.split('.');
            for part in parts[:-1]: obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        set_nested_attr(model, layer_to_prune_name, new_layer); set_nested_attr(model, next_layer_name, new_next_layer)
        return model

    def _find_least_important_unit_l1(self, model):
        # ... (sem alterações)
        prunable_layers = self._get_prunable_layers_info(model)
        min_l1_norm = float('inf'); candidate_info = None
        for layer_name, layer_module in prunable_layers:
            min_units = 2
            if isinstance(layer_module, qnn.QuantConv2d) and layer_module.out_channels <= min_units: continue
            if isinstance(layer_module, qnn.QuantLinear) and layer_module.out_features <= min_units: continue
            if isinstance(layer_module, qnn.QuantConv2d):
                l1_norms = torch.sum(torch.abs(layer_module.weight.data), dim=(1, 2, 3))
            elif isinstance(layer_module, qnn.QuantLinear):
                l1_norms = torch.sum(torch.abs(layer_module.weight.data), dim=1)
            else: continue
            min_norm_in_layer, min_idx_in_layer = torch.min(l1_norms, dim=0)
            if min_norm_in_layer < min_l1_norm:
                min_l1_norm = min_norm_in_layer; candidate_info = {"layer_name": layer_name, "unit_idx": min_idx_in_layer.item()}
        return candidate_info

    def _get_all_units_ranked_by_l1(self, model):
        prunable_layers = self._get_prunable_layers_info(model)
        all_units = []
        for layer_name, layer_module in prunable_layers:
            if isinstance(layer_module, qnn.QuantConv2d):
                l1_norms = torch.sum(torch.abs(layer_module.weight.data), dim=(1, 2, 3))
                num_units = layer_module.out_channels
            elif isinstance(layer_module, qnn.QuantLinear):
                l1_norms = torch.sum(torch.abs(layer_module.weight.data), dim=1)
                num_units = layer_module.out_features
            for i in range(num_units):
                all_units.append({'layer_name': layer_name, 'unit_idx': i, 'l1_norm': l1_norms[i].item()})
        return sorted(all_units, key=lambda x: x['l1_norm'])

    # --- [MÉTODO ATUALIZADO] ---
    def prune_single_least_important_unit(self, model, quantizer_cfg):
        """
        Encontra e remove a única unidade menos importante do modelo.
        """
        self._log_model_size(model, "Antes da Poda")
        candidate = self._find_least_important_unit_l1(model)
        if candidate is None:
            print("\n[Pruning] Nenhuma camada pode ser mais podada.")
            return model
        print(f"  -> Alvo da poda: Unidade {candidate['unit_idx']} da camada '{candidate['layer_name']}'.")
        # Passa a configuração para o método de criação
        pruned_model = self._create_pruned_model(
            model, 
            candidate['layer_name'], 
            candidate['unit_idx'],
            quantizer_cfg
        )
        self._log_model_size(pruned_model, "Depois da Poda")
        return pruned_model
    
    # --- [MÉTODO ATUALIZADO com trava de segurança] ---
    def prune_by_percentage(self, model, quantizer_cfg, strategy):
        self._log_model_size(model, "Antes da Poda (% Lote)")
        
        # 1. Obtém o ranking global de todas as unidades
        ranked_units = self._get_all_units_ranked_by_l1(model)
        if not ranked_units:
            return model

        # 2. Determina quantas unidades podar
        step_percentage = strategy.get('step_percentage', 0.05)
        num_to_prune = int(len(ranked_units) * step_percentage)
        if num_to_prune == 0: num_to_prune = 1

        units_to_prune_candidates = ranked_units[:num_to_prune]
        
        # --- [NOVO] Mecanismo de Segurança ---
        # Garante que nenhuma camada seja completamente eliminada.
        
        # Conta o tamanho original de cada camada podável
        original_layer_sizes = {}
        for name, module in self._get_prunable_layers_info(model):
            if isinstance(module, qnn.QuantConv2d):
                original_layer_sizes[name] = module.out_channels
            elif isinstance(module, qnn.QuantLinear):
                original_layer_sizes[name] = module.out_features
        
        # Conta quantas unidades estão planejadas para poda em cada camada
        planned_prune_counts = defaultdict(int)
        for unit in units_to_prune_candidates:
            planned_prune_counts[unit['layer_name']] += 1
            
        # Filtra a lista de poda para aplicar a trava de segurança
        final_units_to_prune = []
        # Mantém um mínimo de 2 unidades por camada
        MIN_UNITS_LEFT = 2 
        
        # Reconstrói a contagem, respeitando o limite
        current_prune_counts = defaultdict(int)
        for unit in units_to_prune_candidates:
            layer_name = unit['layer_name']
            # Verifica se a poda desta unidade ainda deixaria o mínimo necessário
            if current_prune_counts[layer_name] < original_layer_sizes[layer_name] - MIN_UNITS_LEFT:
                final_units_to_prune.append(unit)
                current_prune_counts[layer_name] += 1
        
        if not final_units_to_prune:
            print("  -> Trava de segurança impediu a poda de qualquer unidade para não eliminar camadas. Fim da poda.")
            return model
            
        print(f"  -> Alvo inicial da poda: {len(units_to_prune_candidates)} unidades.")
        if len(final_units_to_prune) < len(units_to_prune_candidates):
            print(f"  -> Trava de segurança ativada. Podando {len(final_units_to_prune)} unidades para preservar a arquitetura.")
        # --- Fim do Mecanismo de Segurança ---

        # 3. Cria e executa o plano de poda com a lista final e segura
        pruning_plan = defaultdict(list)
        for unit in final_units_to_prune:
            pruning_plan[unit['layer_name']].append(unit['unit_idx'])
        
        pruned_model = model
        for layer_name, indices in pruning_plan.items():
            indices.sort(reverse=True)
            for unit_idx in indices:
                 pruned_model = self._create_pruned_model(pruned_model, layer_name, unit_idx, quantizer_cfg)
        
        self._log_model_size(pruned_model, "Depois da Poda (% Lote)")
        return pruned_model