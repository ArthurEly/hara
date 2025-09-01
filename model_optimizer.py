# model_optimizer.py

import os
import csv
from datetime import datetime
import torch
import copy
from utils.ml_utils import Trainer, Pruner, Evaluator, load_dataloaders, get_model_size, generate_arch_config
from config import TRAINING_CONFIG, CLASSIFICATION_CONSTRAINTS, FINETUNING_CONFIG
import yaml

class ModelOptimizer:
    """
    Orquestra o ciclo de treino, pruning e fine-tuning para otimizar o modelo.
    """
    def __init__(self, build_dir, topologies_config):
        self.build_dir = build_dir
        self.topologies_config = topologies_config
        self.training_config = TRAINING_CONFIG
        self.class_constraints = CLASSIFICATION_CONSTRAINTS
        self.finetuning_config = FINETUNING_CONFIG
        
        self.pytorch_models_dir = os.path.join(self.build_dir, "pytorch_models")
        os.makedirs(self.pytorch_models_dir, exist_ok=True)
        
        self.train_loader, self.test_loader, self.test_len = load_dataloaders(
            self.training_config['batch_size']
        )
        self.trainer = Trainer(self.train_loader, self.test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = Evaluator(self.test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.pruner = Pruner()
        self.training_summary_path = os.path.join(build_dir, "training_summary.csv")
        self._init_summary_file()

    def _init_summary_file(self):
        """ Cria o arquivo CSV de sumário com os novos cabeçalhos. """
        header = [
            'timestamp', 'topology_id', 'quant_bits', 'stage',
            'accuracy', 'f1_score', 'precision', 'recall',
            'num_params', 'size_mb'  # Novas colunas
        ]
        if self.class_constraints.get('prioritize_class') is not None:
            p_class = self.class_constraints['prioritize_class']
            header.extend([f'precision_class_{p_class}', f'recall_class_{p_class}', f'f1_score_class_{p_class}'])
        with open(self.training_summary_path, 'w', newline='') as f:
            csv.writer(f).writerow(header)

    def _log_results(self, model, topology_id, bit_width, stage, report):
        """ Adiciona uma linha de resultado, incluindo o tamanho do modelo. """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Calcula o tamanho do modelo atual
        num_params, size_mb = get_model_size(model)
        
        row = [
            now, topology_id, bit_width, stage,
            report.get('accuracy', 0), report.get('f1_score', 0),
            report.get('precision', 0), report.get('recall', 0),
            num_params, f"{size_mb:.4f}" # Adiciona os novos dados à linha
        ]
        if self.class_constraints.get('prioritize_class') is not None:
            # ... (lógica para classes prioritárias)
            p_class = self.class_constraints['prioritize_class']
            row.extend([report.get(f'precision_class_{p_class}', 0), report.get(f'recall_class_{p_class}', 0), report.get(f'f1_score_class_{p_class}', 0)])
        with open(self.training_summary_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
            
    def _meets_constraints(self, report):
        # ... (sem alterações)
        if not self.class_constraints['enabled']: return True
        metric = self.class_constraints['primary_metric']
        target = self.class_constraints['target_value']
        if metric not in report:
            print(f"[!] Métrica primária '{metric}' não encontrada no relatório de avaliação.")
            return False
        return report[metric] >= target

    def find_optimal_model(self, topology_info):
        """
        Método principal que orquestra a busca pelo modelo ótimo, interpretando
        estratégias de quantização complexas, podando e fazendo fine-tuning de
        forma cíclica até encontrar a melhor solução.
        """
        topology_id = topology_info['id']
        topology_class = topology_info['tp_class']
        quantizer_cfg = topology_info['quantizers']
        
        # --- LÓGICA ATUALIZADA: Gera a lista de larguras de bits a serem testadas ---
        strategy = topology_info.get('quant_strategy')
        if not strategy:
            raise ValueError(f"Estratégia de quantização ('quant_strategy') não definida para a topologia {topology_id}")

        bit_widths_to_try = []
        method = strategy.get('method', 'list')

        print(f"\nIniciando otimização para Topologia {topology_id} com estratégia: '{method}'")

        if method == 'list':
            bit_widths_to_try = topology_info.get('quant_list', [])
        elif method == 'sweep':
            target = strategy.get('sweep_target', 'both')
            w_start, a_start = strategy.get('start_bits', [2, 2])
            w_end, a_end = strategy.get('end_bits', [8, 8])
            
            if target == 'activation': # Varia ativação, peso fixo
                for a in range(a_start, a_end + 1):
                    bit_widths_to_try.append([w_start, a])
            elif target == 'weight': # Varia peso, ativação fixa
                for w in range(w_start, w_end + 1):
                    bit_widths_to_try.append([w, a_start])
            elif target == 'both': # Varia ambos juntos
                steps = max(w_end - w_start, a_end - a_start) + 1
                for i in range(steps):
                    w = min(w_start + i, w_end)
                    a = min(a_start + i, a_end)
                    bit_widths_to_try.append([w, a])
            else:
                raise ValueError(f"Valor de 'sweep_target' desconhecido: {target}")
        
        print(f"  -> Sequência de teste de bits [Pesos, Ativações]: {bit_widths_to_try}")

        if not bit_widths_to_try:
            print(f"[AVISO] Nenhuma largura de bit definida para a topologia {topology_id}. Pulando.")
            return None, None
        
        # O laço agora itera sobre pares de [w_bits, a_bits]
        for w_bits, a_bits in bit_widths_to_try:
            bit_width_str = f"{w_bits}w{a_bits}a"
            print(f"\n--- Fase 1: Tentando com {bit_width_str} para Topologia {topology_id} ---")
            
            # Instancia o modelo com os bit-widths e quantizadores corretos
            model = topology_class(
                weight_bit_width=w_bits,
                act_bit_width=a_bits,
                weight_quant_class=quantizer_cfg['weight'],
                act_quant_class=quantizer_cfg['activation']
            )
            
            trained_model, _ = self.trainer.train(
                model,
                epochs=self.training_config['epochs_per_bitwidth'],
                lr=self.training_config['learning_rate'],
                patience=self.training_config['patience_epochs'],
                constraints=self.class_constraints,
                stop_on_target_met=False
            )

            perf_report = self.evaluator.evaluate(trained_model, self.class_constraints)
            self._log_results(trained_model, topology_id, bit_width_str, "post-training", perf_report)

            if self._meets_constraints(perf_report):
                print(f"[✓] Modelo {bit_width_str} atingiu a meta: {perf_report[self.class_constraints['primary_metric']]:.4f}")
                print(f"  -> Encontrada a menor combinação de bits funcional. Iniciando otimização final.")
                
                final_model = trained_model
                
                if self.training_config['pruning_enabled']:
                    current_model = trained_model
                    last_successful_model = copy.deepcopy(trained_model)
                    max_cycles = self.finetuning_config['max_pruning_finetuning_cycles']
                    cycle_count = 0
                    is_infinite_mode = (max_cycles == -1)

                    while True:
                        if not is_infinite_mode and cycle_count >= max_cycles:
                            print("\n[INFO] Número máximo de ciclos de otimização atingido.")
                            break
                        
                        cycle_display = "Infinito" if is_infinite_mode else max_cycles
                        print(f"\n{'='*20} CICLO DE OTIMIZAÇÃO #{cycle_count+1}/{cycle_display} {'='*20}")
                        model_at_cycle_start = copy.deepcopy(current_model)
                        
                        pruning_strategy_cfg = self.training_config['pruning_strategy']
                        method = pruning_strategy_cfg.get('method', 'iterative')
                        print(f"\n--- Iniciando poda com a estratégia: '{method}' ---")
                        
                        pruned_model_for_ft = current_model # Modelo a ser usado no fine-tuning
                        
                        if method == 'iterative':
                            # Laço de poda contínua para o método 'iterative'
                            while True:
                                candidate = self.pruner.prune_single_least_important_unit(current_model, quantizer_cfg)
                                if sum(p.numel() for p in candidate.parameters()) >= sum(p.numel() for p in current_model.parameters()):
                                    pruned_model_for_ft = current_model
                                    break
                                report = self.evaluator.evaluate(candidate, self.class_constraints)
                                self._log_results(candidate, topology_id, bit_width_str, f"cycle_{cycle_count+1}_pruning", report)
                                if self._meets_constraints(report):
                                    current_model = candidate
                                else:
                                    pruned_model_for_ft = candidate
                                    break
                        
                        elif method == 'percentage':
                            # O método de porcentagem poda um lote e retorna o resultado
                            pruned_model_for_ft = self.pruner.prune_by_percentage(
                                current_model, quantizer_cfg, pruning_strategy_cfg
                            )
                        else:
                            raise ValueError(f"Estratégia de pruning desconhecida: {method}")
                        
                        current_model = pruned_model_for_ft
                        
                        if sum(p.numel() for p in current_model.parameters()) >= sum(p.numel() for p in model_at_cycle_start.parameters()):
                             print("\n[INFO] Nenhuma poda bem-sucedida neste ciclo. Otimização finalizada.")
                             final_model = model_at_cycle_start
                             break

                        if self.finetuning_config['enabled']:
                            print("\n--- Iniciando Fine-Tuning para tentar recuperar performance ---")
                            finetuned_model, _ = self.trainer.train(model=current_model, epochs=self.finetuning_config['epochs'], lr=self.finetuning_config['learning_rate'], patience=self.finetuning_config['patience_epochs'], constraints=self.class_constraints, stop_on_target_met=True)
                            finetuned_report = self.evaluator.evaluate(finetuned_model, self.class_constraints)
                            self._log_results(finetuned_model, topology_id, bit_width_str, f"cycle_{cycle_count+1}_post-finetuning", finetuned_report)
                            if self._meets_constraints(finetuned_report):
                                print(f"[✓] CICLO {cycle_count+1} SUCESSO! Performance recuperada.")
                                current_model = finetuned_model
                                last_successful_model = copy.deepcopy(finetuned_model)
                            else:
                                print(f"[✗] CICLO {cycle_count+1} FALHA. Performance não recuperada. Encerrando.")
                                final_model = last_successful_model
                                break
                        else:
                            final_model = current_model
                            break
                        
                        final_model = last_successful_model
                        cycle_count += 1
                
                # --- LÓGICA DE SALVAMENTO YAML + PTH ---
                print("\n--- Salvando modelo final otimizado ---")
                final_arch_config = generate_arch_config(final_model)
                metadata = {
                    'topology_id': topology_id,
                    'bit_widths': {'weight': w_bits, 'activation': a_bits},
                    'quantizer_classes': {'weight': quantizer_cfg['weight'].__name__, 'activation': quantizer_cfg['activation'].__name__},
                    'arch_config': final_arch_config,
                }
                base_filename = f"t{topology_id}w{w_bits}a{a_bits}_final"
                yaml_path = os.path.join(self.pytorch_models_dir, f"{base_filename}.yaml")
                pth_path = os.path.join(self.pytorch_models_dir, f"{base_filename}.pth")
                with open(yaml_path, 'w') as f:
                    yaml.dump(metadata, f, indent=4)
                print(f"Configuração da arquitetura salva em: {yaml_path}")
                torch.save(final_model.state_dict(), pth_path)
                print(f"Pesos do modelo salvos em: {pth_path}")

                return os.path.join(self.pytorch_models_dir, base_filename), bit_width_str

            else:
                print(f"[✗] Modelo {bit_width_str} NÃO atingiu a meta. Tentando a próxima combinação...")

        print(f"[✗] Nenhuma combinação de quantização para a Topologia {topology_id} atingiu a meta.")
        return None, None