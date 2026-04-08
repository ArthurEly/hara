#!/usr/bin/env python3
"""
Compara os resultados do modelo analítico (analytic_utils.py) 
com o modelo do FINN iterativo, usando as estimativas geradas.

Para rodar para todos os cenários da pasta fps_campaign_results:
    python3 compare_analytical_vs_finn.py

Para liberar espaço do disco limando os artefatos pesados gerados pelo FINN depois:
    python3 compare_analytical_vs_finn.py --clean

Requer que as pastas originais ainda tenham o 'intermediate_models/*.onnx'
(caso não possuam, ele avisa).
"""

import argparse
import json
import os
import shutil
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from utils.analytic_utils import FinnCycleEstimator
from utils.hw_utils import utils

def main():
    parser = argparse.ArgumentParser(description="Compara FPS analítico vs FINN.")
    parser.add_argument('--results-dir', type=str, default='./fps_campaign_results', 
                        help='Diretório base onde os resultados estão salvos.')
    parser.add_argument('--clean', action='store_true',
                        help='Remove diretórios pesados do FINN após a análise para liberar muito espaço no disco.')
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    if not base_dir.exists():
        print(f"[✗] Diretório {base_dir} não encontrado.")
        return

    subdirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    print(f"Encontrados {len(subdirs)} diretórios de resultados em {base_dir}")

    all_comparison_results = []
    failed_comparisons = []

    for d in subdirs:
        print(f"\n--- Processando {d.name} ---")
        csv_path = d / "fps_map.csv"
        if not csv_path.exists():
            print("  -> fps_map.csv não encontrado. Pulando.")
            continue

        # Procura por um ONNX intermediário do run1 (baseline) para carregar o modelo analítico
        onnx_candidates = list(d.glob("run1_baseline_folded/intermediate_models/step_generate_estimate_reports.onnx"))
        if not onnx_candidates:
            onnx_candidates = list(d.glob("run*/intermediate_models/*.onnx"))
            
        if not onnx_candidates:
            print("  -> ARQUIVO ONNX base não encontrado (talvez já tenha sido feito clean?). Impossível rodar modelo analítico.")
            continue
            
        onnx_path = onnx_candidates[0]
        print(f"  -> ONNX encontrado: {onnx_path.relative_to(d)}")

        # 1. Carrega o estimador analítico
        try:
            analyzer = FinnCycleEstimator(str(onnx_path), debug=False)
            cycle_formulas = analyzer.get_cycle_formulas()
        except Exception as e:
            print(f"  -> [✗] Falha ao iniciar modelo analítico: {e}")
            continue

        # 2. Lê os resultados do FINN
        df = pd.read_csv(csv_path)
        
        runs = []
        analytical_fps_list = []

        # 3. Para cada folding no CSV de FINN, calcular pelo modelo Analítico
        f_clock = 100e6  # 100 MHz
        for _, row in df.iterrows():
            run_id = row["run_id"]
            folding_config = json.loads(row["folding_config"])
            
            # Avalia ciclos de todas as camadas
            max_cycles = 0  # "bottleneck_cycles"
            for layer_name, data in cycle_formulas.items():
                formula = data['formula']
                
                cfg = folding_config.get(layer_name, {})
                defaults = folding_config.get("Defaults", {"PE": 1, "SIMD": 1})
                
                pe = cfg.get("PE", defaults.get("PE", 1))
                simd = cfg.get("SIMD", defaults.get("SIMD", 1))
                parallel_window = cfg.get("parallel_window", 0)

                current_params = {"PE": pe, "SIMD": simd}

                if "ConvolutionInputGenerator" in data.get("op_type", "") and parallel_window == 1:
                    # Lógica do analytic_utils para PW
                    # Sem recarregar o ONNX na unha, a fórmula é: (IFMChannels * dim_w * dim_h / SIMD) + 2
                    import onnx
                    from onnx import helper
                    m = onnx.load(str(onnx_path))
                    node = next((n for n in m.graph.node if n.name == layer_name), None)
                    ifm_dim_w, ifm_dim_h = 1, 1
                    if node:
                       for attr in node.attribute:
                           if attr.name == "IFMDim":
                               ifm_dim_w = helper.get_attribute_value(attr)[0]
                               ifm_dim_h = helper.get_attribute_value(attr)[1]
                    cycles = (data.get("IFMChannels", 1) * ifm_dim_w * ifm_dim_h / simd) + 2
                else:
                    cycles = analyzer._eval_formula(formula, current_params)
                
                if cycles > max_cycles:
                    max_cycles = cycles
            
            analyt_fps = f_clock / max_cycles if max_cycles > 0 else 0
            analytical_fps_list.append(analyt_fps)
            runs.append(run_id)

        df["analytical_fps"] = analytical_fps_list
        df["error_pct"] = ((df["analytical_fps"] - df["estimated_fps"]) / df["estimated_fps"] * 100).fillna(0)
        
        # Tracking para o sumário final
        max_err = df["error_pct"].abs().max()
        if max_err > 0.5: # Tolerância de 0.5% para micro arredondamentos matemáticos
            mismatches = len(df[df["error_pct"].abs() > 0.5])
            failed_comparisons.append({
                "model": d.name,
                "max_error_pct": max_err,
                "mismatches": mismatches
            })

        # Salva o CSV atualizado
        combined_csv = d / "fps_map_combined.csv"
        df.to_csv(combined_csv, index=False)
        print(f"  -> [✓] Arquivo {combined_csv.name} gravado com dados analíticos.")

        # Plot comparativo em escalas logarítmicas/lineares adaptado
        plt.figure(figsize=(10, 6))
        plt.plot(df["run_id"], df["estimated_fps"], marker='o', linestyle='-', label="FINN Estimativa (Hardware)", linewidth=2)
        plt.plot(df["run_id"], df["analytical_fps"], marker='s', linestyle='--', label="Modelo Analítico", linewidth=2)
        
        plt.title(f"Acurácia do Modelo Analítico vs FINN - {d.name.split('_202')[0]}")
        plt.xlabel("Iteração do Otimizador (Aumento de Paralelismo)")
        plt.ylabel("FPS (Frames por Segundo)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.yscale("log")  # Usando log, pois o FPS pode crescer de 1 a milhões
        
        plot_path = d / "fps_comparison.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"  -> [✓] Gráfico comparativo gerado: fps_comparison.png.")

        # Realizar o Clean caso Solicitado
        if args.clean:
            # Lista todas as subpastas "run..." de cada tentativa do FINN
            run_dirs = [r for r in d.iterdir() if r.is_dir() and r.name.startswith("run")]
            for rdir in run_dirs:
                utils.clean_build_artifacts(str(rdir))
            print("  -> [CLEAN] Diretórios pesados removidos com sucesso.")

    # =========================================================
    #                    PRINT DO SUMÁRIO FINAL
    # =========================================================
    print("\n" + "="*80)
    print("                  RESUMO FINAL DA VALIDAÇÃO ANALÍTICA                   ")
    print("="*80)
    
    if not failed_comparisons:
        print("\n[✓] SUCESSO TOTAL! Todas as estimativas analíticas bateram com o FINN perfeitamente.")
        print("    O modelo matemático está 100% calibrado e alinhado com o hardware gerado.")
    else:
        print(f"\n[!] AVISO: {len(failed_comparisons)} campanha(s) apresentaram divergência (>0.5% de erro) entre a fórmula matemática e o FINN:\n")
        for f in failed_comparisons:
            print(f"  -> {f['model']}: Erro máximo de {f['max_error_pct']:.2f}% ({f['mismatches']} iterações divergiram)")
        print("\nRevise a implementação das fórmulas ou as conexões do pipeline para corrigir o descompasso.")
        print("Dica: rode este script sem a flag --clean para manter o ONNX e facilitar a inspeção log-a-log.")
    print("\n")


if __name__ == '__main__':
    main()
