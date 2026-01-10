# model_optimizer.py
import os
import torch
import copy
import yaml
from utils.ml_utils import generate_arch_config # Dependência do seu ml_utils.py

class ModelOptimizer:
    """
    Orquestra o ciclo de treino, pruning e fine-tuning.
    
    Esta versão refatorada recebe todas as suas dependências (Injeção de Dependência),
    tornando-a totalmente testável e desacoplada dos carregadores de dados
    e arquivos de configuração globais.
    """
    def __init__(self, 
                 build_dir, 
                 trainer, 
                 evaluator, 
                 pruner, 
                 logger, 
                 quantizer_cfg, # <-- Adicionado para corrigir o bug de poda
                 training_cfg, 
                 class_constraints, 
                 finetuning_cfg):
        
        # Dependências injetadas
        self.trainer = trainer
        self.evaluator = evaluator
        self.pruner = pruner
        self.logger = logger
        
        # Configurações injetadas
        self.quantizer_cfg = quantizer_cfg # Ex: {'weight': CommonWeightQuant}
        self.training_cfg = training_cfg
        self.class_constraints = class_constraints
        self.finetuning_cfg = finetuning_cfg

        # Gerenciamento de diretórios
        self.build_dir = build_dir
        self.pytorch_models_dir = os.path.join(self.build_dir, "pytorch_models")
        os.makedirs(self.pytorch_models_dir, exist_ok=True)


    def _meets_constraints(self, report):
        """Verifica se um relatório de performance atende às restrições."""
        if not self.class_constraints.get('enabled', True): 
            return True
        
        metric = self.class_constraints['primary_metric']
        target = self.class_constraints['target_value']
        
        if metric not in report:
            print(f"[!] Métrica primária '{metric}' não encontrada no relatório.")
            return False
        return report[metric] >= target

    def _generate_bit_width_tests(self, strategy):
        """Gera a lista de pares de [w, a] bits para testar."""
        bit_widths_to_try = []
        method = strategy.get('method', 'list')

        if method == 'list':
            bit_widths_to_try = strategy.get('quant_list', [])
        elif method == 'sweep':
            target = strategy.get('sweep_target', 'both')
            w_start, a_start = strategy.get('start_bits', [2, 2])
            w_end, a_end = strategy.get('end_bits', [8, 8])
            
            if target == 'activation':
                for a in range(a_start, a_end + 1): bit_widths_to_try.append([w_start, a])
            elif target == 'weight':
                for w in range(w_start, w_end + 1): bit_widths_to_try.append([w, a_start])
            elif target == 'both':
                steps = max(w_end - w_start, a_end - a_start) + 1
                for i in range(steps):
                    w = min(w_start + i, w_end)
                    a = min(a_start + i, a_end)
                    bit_widths_to_try.append([w, a])
            else:
                raise ValueError(f"Valor de 'sweep_target' desconhecido: {target}")
        
        return bit_widths_to_try

    def _save_model_artifacts(self, model, topology_id, w_bits, a_bits):
        """Gera o YAML de metadados e salva o .pth e .yaml finais."""
        print("\n--- Salvando modelo final otimizado ---")
        final_arch_config = generate_arch_config(model)
        metadata = {
            'model_source': 'hara_internal',
            'topology_id': topology_id,
            'bit_widths': {'weight': w_bits, 'activation': a_bits},                    
            'arch_config': final_arch_config,
        }
        base_filename = f"{topology_id}w{w_bits}a{a_bits}_final"
        yaml_path = os.path.join(self.pytorch_models_dir, f"{base_filename}.yaml")
        pth_path = os.path.join(self.pytorch_models_dir, f"{base_filename}.pth")
        
        with open(yaml_path, 'w') as f:
            yaml.dump(metadata, f, indent=4)
        print(f"Configuração da arquitetura salva em: {yaml_path}")
        
        torch.save(model.state_dict(), pth_path)
        print(f"Pesos do modelo salvos em: {pth_path}")
        
        # Retorna o "caminho base" (sem extensão) para o executor
        return os.path.join(self.pytorch_models_dir, base_filename)

    def _run_pruning_finetuning_cycle(self, trained_model, topology_id, bit_width_str):
        """
        Executa o ciclo completo de poda e fine-tuning.
        Esta é a lógica que foi omitida anteriormente, agora completa.
        """
        
        current_model = trained_model
        # last_successful_model é nosso "checkpoint" do último modelo que passou nas restrições
        last_successful_model = copy.deepcopy(trained_model)
        
        max_cycles = self.finetuning_cfg.get('max_pruning_finetuning_cycles', -1)
        cycle_count = 0
        is_infinite_mode = (max_cycles == -1)

        while True:
            if not is_infinite_mode and cycle_count >= max_cycles:
                print("\n[INFO] Número máximo de ciclos de otimização atingido.")
                break
            
            cycle_display = "Infinito" if is_infinite_mode else max_cycles
            print(f"\n{'='*20} CICLO DE OTIMIZAÇÃO #{cycle_count+1}/{cycle_display} {'='*20}")
            model_at_cycle_start = copy.deepcopy(current_model)
            
            # --- 1. Fase de Poda ---
            pruning_strategy_cfg = self.training_cfg['pruning_strategy']
            method = pruning_strategy_cfg.get('method', 'iterative')
            print(f"\n--- Iniciando poda com a estratégia: '{method}' ---")
            
            pruned_model = None
            if method == 'iterative':
                # O método 'iterative' poda uma unidade de cada vez e para
                # na primeira vez que a performance cair abaixo da meta.
                while True:
                    # Passa o quantizer_cfg, corrigindo o bug do código original
                    candidate = self.pruner.prune_single_least_important_unit(current_model, self.quantizer_cfg)
                    
                    # Verifica se a poda realmente removeu algo
                    if sum(p.numel() for p in candidate.parameters()) >= sum(p.numel() for p in current_model.parameters()):
                        print("\n[INFO] Pruner não conseguiu remover mais unidades. Parando poda iterativa.")
                        pruned_model = current_model # O modelo antes da última tentativa
                        break 
                        
                    report = self.evaluator.evaluate(candidate, self.class_constraints)
                    self.logger.log_step(candidate, topology_id, bit_width_str, f"cycle_{cycle_count+1}_pruning", report)
                    
                    if self._meets_constraints(report):
                        current_model = candidate # Sucesso, continue podando
                    else:
                        pruned_model = candidate # Falhou, use este modelo para fine-tuning
                        break
            
            elif method == 'percentage':
                # O método 'percentage' poda um lote de uma vez
                pruned_model = self.pruner.prune_by_percentage(
                    current_model, self.quantizer_cfg, pruning_strategy_cfg
                )
            else:
                raise ValueError(f"Estratégia de pruning desconhecida: {method}")

            # Atualiza o modelo atual para ser o modelo podado
            current_model = pruned_model
            
            # Verifica se a poda de fato aconteceu
            if sum(p.numel() for p in current_model.parameters()) >= sum(p.numel() for p in model_at_cycle_start.parameters()):
                 print("\n[INFO] Nenhuma poda bem-sucedida neste ciclo. Otimização finalizada.")
                 break # Sai do 'while True' principal

            # --- 2. Fase de Fine-Tuning (se habilitada) ---
            if self.finetuning_cfg['enabled']:
                print("\n--- Iniciando Fine-Tuning para tentar recuperar performance ---")
                finetuned_model, _ = self.trainer.train(
                    model=current_model, 
                    epochs=self.finetuning_cfg['epochs'], 
                    lr=self.finetuning_cfg['learning_rate'], 
                    patience=self.finetuning_cfg['patience_epochs'], 
                    constraints=self.class_constraints, 
                    stop_on_target_met=True
                )
                
                finetuned_report = self.evaluator.evaluate(finetuned_model, self.class_constraints)
                self.logger.log_step(finetuned_model, topology_id, bit_width_str, f"cycle_{cycle_count+1}_post-finetuning", finetuned_report)
                
                if self._meets_constraints(finetuned_report):
                    print(f"[✓] CICLO {cycle_count+1} SUCESSO! Performance recuperada.")
                    current_model = finetuned_model
                    last_successful_model = copy.deepcopy(finetuned_model) # Salva o novo "checkpoint"
                else:
                    print(f"[✗] CICLO {cycle_count+1} FALHA. Performance não recuperada. Encerrando.")
                    break # Sai do 'while True', revertendo para o 'last_successful_model'
            
            # --- 3. Fase de Avaliação (se fine-tuning estiver desabilitado) ---
            else:
                print("\n--- Fine-Tuning desabilitado, avaliando modelo podado ---")
                pruned_report = self.evaluator.evaluate(current_model, self.class_constraints)
                self.logger.log_step(current_model, topology_id, bit_width_str, f"cycle_{cycle_count+1}_post-pruning", pruned_report)
                
                if self._meets_constraints(pruned_report):
                    print(f"[✓] Modelo podado ainda atende às restrições.")
                    last_successful_model = copy.deepcopy(current_model) # Salva o novo "checkpoint"
                else:
                    print(f"[✗] Modelo podado não atende às restrições. Encerrando.")
                    break # Sai do 'while True', revertendo para o 'last_successful_model'
            
            cycle_count += 1
        
        # Retorna o último modelo bem-sucedido e que passou nas restrições
        return last_successful_model


    def find_optimal_model(self, topology_info):
        """
        Método principal que orquestra a busca pelo modelo ótimo.
        """
        topology_id = topology_info['id']
        topology_class = topology_info['tp_class']
        
        strategy = topology_info.get('quant_strategy')
        if not strategy:
            raise ValueError(f"Estratégia de quantização não definida para {topology_id}")

        bit_widths_to_try = self._generate_bit_width_tests(strategy)
        print(f"\nIniciando otimização para {topology_id} com estratégia: '{strategy.get('method')}'")
        print(f"  -> Sequência de teste [W, A]: {bit_widths_to_try}")

        if not bit_widths_to_try:
            print(f"[AVISO] Nenhuma largura de bit definida para {topology_id}. Pulando.")
            return None, None
        
        for w_bits, a_bits in bit_widths_to_try:
            bit_width_str = f"{w_bits}w{a_bits}a"
            print(f"\n--- Fase 1: Tentando com {bit_width_str} para {topology_id} ---")
            
            model = topology_class(weight_bit_width=w_bits, act_bit_width=a_bits)
            
            trained_model, _ = self.trainer.train(
                model,
                epochs=self.training_cfg['epochs_per_bitwidth'],
                lr=self.training_cfg['learning_rate'],
                patience=self.training_cfg['patience_epochs'],
                constraints=self.class_constraints,
                stop_on_target_met=False
            )

            perf_report = self.evaluator.evaluate(trained_model, self.class_constraints)
            self.logger.log_step(trained_model, topology_id, bit_width_str, "post-training", perf_report)

            if self._meets_constraints(perf_report):
                print(f"[✓] Modelo {bit_width_str} atingiu a meta: {perf_report[self.class_constraints['primary_metric']]:.4f}")
                
                final_model = trained_model
                
                if self.training_cfg.get('pruning_enabled', False):
                    print("  -> Iniciando ciclo de otimização (Pruning & Fine-Tuning)...")
                    final_model = self._run_pruning_finetuning_cycle(
                        trained_model, topology_id, bit_width_str
                    )
                else:
                    print("  -> Pruning desabilitado. Salvando modelo treinado.")
                
                # Salva os artefatos finais
                saved_model_base_path = self._save_model_artifacts(final_model, topology_id, w_bits, a_bits)
                
                # Para o loop de bit-width, pois encontramos o menor que funciona
                return saved_model_base_path, bit_width_str

            else:
                print(f"[✗] Modelo {bit_width_str} NÃO atingiu a meta. Próxima combinação...")

        print(f"[✗] Nenhuma combinação para {topology_id} atingiu a meta.")
        return None, None