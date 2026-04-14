"""
evaluate_all_builds.py
========================================================================
Valida o HARA em todos os builds consolidados, comparando os sumários de
hardware com as predições do MultiModuleLearner.

Atualizações desta versão:
- caminho de modelos alinhado com results/trained_models;
- relatório inclui DSP;
- mensagens explícitas sobre ausência dos artefatos auxiliares de FIFO.
"""

import glob
import json
import os

import pandas as pd
from tqdm import tqdm

from multi_module_learner import MultiModuleLearner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_BUILDS_DIR = "/home/arthurely/Desktop/finn_chi2p/hara/exhaustive_hw_builds"
MODELS_DIR = os.path.join(BASE_DIR, "retrieval", "results", "trained_models")
ALLOWED_DATASETS = ["MNIST_1W1A", "SAT6_T2W2"]


def calculate_mape(y_true, y_pred):
    if y_true == 0:
        return 0.0 if y_pred == 0 else 100.0
    return abs((y_pred - y_true) / y_true) * 100.0


def main():
    print("[HARA Validator] Inicializando MultiModuleLearner...")
    learner = MultiModuleLearner(MODELS_DIR)

    if not learner.is_loaded():
        print("[!] Erro: nenhum modelo carregado. Verifique results/trained_models.")
        return

    if not os.path.exists(os.path.join(MODELS_DIR, "StreamingFIFO_Classifier.pkl")):
        print("[!] Aviso: StreamingFIFO_Classifier.pkl não encontrado.")
    if not os.path.exists(os.path.join(MODELS_DIR, "SplitFIFO_area_model.pkl")):
        print("[!] Aviso: SplitFIFO_area_model.pkl não encontrado.")

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

        if not any(ds in bateria_name for ds in ALLOWED_DATASETS):
            status["filtered_out"] += len(df_hw)
            continue

        for _, row in tqdm(df_hw.iterrows(), total=len(df_hw), desc=f"Avaliando {bateria_name}"):
            hw_name = str(row.get("hw_name", ""))
            hw_status = str(row.get("status", ""))
            if hw_status != "success":
                status["failed_hw"] += 1
                continue

            run_dir = os.path.join(base_dir, hw_name)
            onnx_path = os.path.join(run_dir, "intermediate_models", "step_generate_estimate_reports.onnx")
            config_path = os.path.join(run_dir, "final_hw_config.json")
            if not os.path.exists(onnx_path) or not os.path.exists(config_path):
                status["missing_onnx"] += 1
                continue

            with open(config_path, "r") as f:
                cfg = json.load(f)

            # --- FILTRO DO USUÁRIO: Ignorar se tiver ram_style='auto' em FIFOs ---
            has_auto_fifo = False
            for k, v in cfg.items():
                if "StreamingFIFO" in k and isinstance(v, dict):
                    if v.get("ram_style", "auto") == "auto" or v.get("ram_style") == b"auto":
                        has_auto_fifo = True
                        break
            
            if has_auto_fifo:
                status["filtered_out"] += 1
                continue

            try:
                preds = learner.predict(onnx_path, [cfg])[0]
            except Exception as e:
                print(f"\n[!] Erro no ML para {bateria_name}/{hw_name}: {e}")
                status["ml_error"] += 1
                continue

            actual_luts = float(row.get("Total LUTs", 0))
            actual_ffs = float(row.get("FFs", 0))
            actual_bram = float(row.get("BRAM (36k)", 0.0))
            actual_dsp = float(row.get("DSP Blocks", 0))

            pred_luts = float(preds.get("Total LUTs", 0))
            pred_ffs = float(preds.get("FFs", 0))
            pred_bram = float(preds.get("BRAM (36k)", 0.0))
            pred_dsp = float(preds.get("DSP Blocks", 0))

            status["success"] += 1
            results_list.append(
                {
                    "Build Name": f"{bateria_name}/{hw_name}",
                    "Real_LUTs": actual_luts,
                    "Pred_LUTs": pred_luts,
                    "Err%_LUTs": calculate_mape(actual_luts, pred_luts),
                    "Real_FFs": actual_ffs,
                    "Pred_FFs": pred_ffs,
                    "Err%_FFs": calculate_mape(actual_ffs, pred_ffs),
                    "Real_BRAM": actual_bram,
                    "Pred_BRAM": pred_bram,
                    "Err%_BRAM": calculate_mape(actual_bram, pred_bram),
                    "Real_DSP": actual_dsp,
                    "Pred_DSP": pred_dsp,
                    "Err%_DSP": calculate_mape(actual_dsp, pred_dsp),
                }
            )

    print(f"\n\n{'=' * 60}")
    print("📊 DIAGNÓSTICO DA VARREDURA")
    print(f"  - Sucesso (validados):           {status['success']}")
    print(f"  - Ignorados (filtro dataset):    {status['filtered_out']}")
    print(f"  - Ignorados (síntese falhou):    {status['failed_hw']}")
    print(f"  - Ignorados (falta ONNX/config): {status['missing_onnx']}")
    print(f"  - Ignorados (erro no ML):        {status['ml_error']}")
    print(f"{'=' * 60}")

    if not results_list:
        print("[!] Nenhum build foi validado com sucesso.")
        return

    df_out = pd.DataFrame(results_list)
    csv_out = os.path.join(BASE_DIR, "hara_validation_results.csv")
    df_out.to_csv(csv_out, index=False)

    print(f"\n🏆 MÉDIA DE ERRO GLOBAL (MAPE) DOS {status['success']} BUILDS:")
    print(f"    - LUTs: {df_out['Err%_LUTs'].mean():.2f}%")
    print(f"    - FFs : {df_out['Err%_FFs'].mean():.2f}%")
    print(f"    - BRAM: {df_out['Err%_BRAM'].mean():.2f}%")
    print(f"    - DSP : {df_out['Err%_DSP'].mean():.2f}%")
    print(f"\nPlanilha detalhada salva em: {csv_out}")


if __name__ == "__main__":
    main()