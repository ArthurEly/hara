"""
evaluate_all_builds.py - Validação do HARA v3 (Filtro MNIST/SAT6)
========================================================================
Lê os sumários de hardware já consolidados na raiz dos builds e compara
diretamente com as predições do modelo de Machine Learning, ignorando
datasets não homologados (ex: CIFAR10).
"""

import os
import json
import glob
import pandas as pd
from tqdm import tqdm
from multi_module_learner import MultiModuleLearner

# =============================================================================
# CONFIGURAÇÕES DE DIRETÓRIO E FILTROS
# =============================================================================
BASE_BUILDS_DIR = "/home/arthurely/Desktop/finn_chi2p/hara/exhaustive_hw_builds"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieval", "results", "trained_models")

# Filtro para isolar apenas os datasets desejados
ALLOWED_DATASETS = ["MNIST", "SAT6_T2"]

def calculate_mape(y_true, y_pred):
    """Calcula o Erro Percentual Absoluto Médio (MAPE)."""
    if y_true == 0:
        return 0.0 if y_pred == 0 else 100.0
    return abs((y_pred - y_true) / y_true) * 100.0

def main():
    print("[HARA Validator] Inicializando MultiModuleLearner...")
    learner = MultiModuleLearner(MODELS_DIR)
    
    if not learner.is_loaded():
        print("[!] Erro: Nenhum modelo carregado. Verifique a pasta de modelos.")
        return

    # Busca todos os sumários na raiz de cada bateria
    search_pattern = os.path.join(BASE_BUILDS_DIR, "**", "hardware_summary.csv")
    summary_files = glob.glob(search_pattern, recursive=True)
    
    print(f"[HARA Validator] Encontrados {len(summary_files)} arquivos 'hardware_summary.csv'.\n")
    
    results_list = []
    status = {"success": 0, "failed_hw": 0, "missing_onnx": 0, "ml_error": 0, "filtered_out": 0}

    for summary_path in summary_files:
        base_dir = os.path.dirname(summary_path)
        bateria_name = os.path.basename(base_dir)
        
        try:
            df_hw = pd.read_csv(summary_path)
        except Exception as e:
            print(f"[!] Erro ao ler {summary_path}: {e}")
            continue

        # --- FILTRO DE DATASET ---
        if not any(ds in bateria_name for ds in ALLOWED_DATASETS):
            status["filtered_out"] += len(df_hw)
            continue

        # Itera sobre cada build registrado no CSV
        for _, row in tqdm(df_hw.iterrows(), total=len(df_hw), desc=f"Avaliando {bateria_name}"):
            hw_name = str(row.get('hw_name', ''))
            hw_status = str(row.get('status', ''))
            
            # Só avalia o que de fato terminou a síntese com sucesso
            if hw_status != 'success':
                status["failed_hw"] += 1
                continue

            run_dir = os.path.join(base_dir, hw_name)
            onnx_path = os.path.join(run_dir, "intermediate_models", "step_generate_estimate_reports.onnx")
            config_path = os.path.join(run_dir, "final_hw_config.json")
            
            if not os.path.exists(onnx_path) or not os.path.exists(config_path):
                status["missing_onnx"] += 1
                continue

            # Carrega a configuração de folding
            with open(config_path, "r") as f:
                cfg = json.load(f)

            # --- 1. Predição do HARA ---
            try:
                preds = learner.predict(onnx_path, [cfg])[0]
            except Exception as e:
                print(f"\n[!] Erro no ML para {hw_name}: {e}")
                status["ml_error"] += 1
                continue

            # --- 2. Realidade do Vivado (Extraído do CSV) ---
            actual_luts = float(row.get('Total LUTs', 0))
            actual_ffs  = float(row.get('FFs', 0))
            actual_bram = float(row.get('BRAM (36k)', 0.0))
            actual_dsp  = float(row.get('DSP Blocks', 0))

            status["success"] += 1
            
            # --- 3. Registro dos Erros ---
            res_row = {
                "Build Name": f"{bateria_name}/{hw_name}",
                
                "Real_LUTs": actual_luts, 
                "Pred_LUTs": preds.get("Total LUTs", 0), 
                "Err%_LUTs": calculate_mape(actual_luts, preds.get("Total LUTs", 0)),
                
                "Real_FFs": actual_ffs, 
                "Pred_FFs": preds.get("FFs", 0), 
                "Err%_FFs": calculate_mape(actual_ffs, preds.get("FFs", 0)),
                
                "Real_BRAM": actual_bram, 
                "Pred_BRAM": preds.get("BRAM (36k)", 0.0), 
                "Err%_BRAM": calculate_mape(actual_bram, preds.get("BRAM (36k)", 0.0)),
                
                "Real_DSP": actual_dsp, 
                "Pred_DSP": preds.get("DSP Blocks", 0),
            }
            results_list.append(res_row)

    # =========================================================================
    # GERAÇÃO DO RELATÓRIO FINAL
    # =========================================================================
    print(f"\n\n{'='*55}")
    print("📊 DIAGNÓSTICO DA VARREDURA (MNIST / SAT6_T2):")
    print(f"  - Sucesso (Validados):             {status['success']}")
    print(f"  - Ignorados (Filtro CIFAR/Outros): {status['filtered_out']}")
    print(f"  - Ignorados (Síntese falhou):      {status['failed_hw']}")
    print(f"  - Ignorados (Falta ONNX/Config):   {status['missing_onnx']}")
    print(f"  - Ignorados (Erro no ML):          {status['ml_error']}")
    print(f"{'='*55}")

    if results_list:
        df_out = pd.DataFrame(results_list)
        csv_out = "hara_validation_results.csv"
        df_out.to_csv(csv_out, index=False)
        
        print(f"🏆 MÉDIA DE ERRO GLOBAL (MAPE) DOS {status['success']} BUILDS (SEM CIFAR):")
        print(f"    - Área de Lógica (LUTs): {df_out['Err%_LUTs'].mean():.2f}%")
        print(f"    - Registradores  (FFs) : {df_out['Err%_FFs'].mean():.2f}%")
        print(f"    - Memória        (BRAM): {df_out['Err%_BRAM'].mean():.2f}%")
        print(f"\nPlanilha detalhada salva em: {csv_out}")
    else:
        print("[!] Nenhum resultado foi validado com sucesso.")

if __name__ == "__main__":
    main()