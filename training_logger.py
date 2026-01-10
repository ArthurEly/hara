# training_logger.py
import os
import csv
from datetime import datetime
from utils.ml_utils import get_model_size

class TrainingSummaryLogger:
    """
    Classe dedicada a inicializar e escrever no arquivo de sumário CSV.
    """
    def __init__(self, file_path, class_constraints):
        self.summary_path = file_path
        self.class_constraints = class_constraints
        self._init_summary_file()

    def _init_summary_file(self):
        """ Cria o arquivo CSV de sumário com os cabeçalhos. """
        header = [
            'timestamp', 'topology_id', 'quant_bits', 'stage',
            'accuracy', 'f1_score', 'precision', 'recall',
            'num_params', 'size_mb'
        ]
        if self.class_constraints.get('prioritize_class') is not None:
            p_class = self.class_constraints['prioritize_class']
            header.extend([f'precision_class_{p_class}', f'recall_class_{p_class}', f'f1_score_class_{p_class}'])
        
        os.makedirs(os.path.dirname(self.summary_path), exist_ok=True)
        with open(self.summary_path, 'w', newline='') as f:
            csv.writer(f).writerow(header)

    def log_step(self, model, topology_id, bit_width_str, stage, report):
        """ Adiciona uma linha de resultado, incluindo o tamanho do modelo. """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        num_params, size_mb = get_model_size(model)
        
        row = [
            now, topology_id, bit_width_str, stage,
            report.get('accuracy', 0), report.get('f1_score', 0),
            report.get('precision', 0), report.get('recall', 0),
            num_params, f"{size_mb:.4f}"
        ]
        if self.class_constraints.get('prioritize_class') is not None:
            p_class = self.class_constraints['prioritize_class']
            row.extend([report.get(f'precision_class_{p_class}', 0), report.get(f'recall_class_{p_class}', 0), report.get(f'f1_score_class_{p_class}', 0)])
        
        with open(self.summary_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)