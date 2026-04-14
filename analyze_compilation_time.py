"""
analyze_compilation_time.py
Mapeia o tempo gasto em cada etapa da compilação do FINN e calcula
a escalabilidade do tempo de síntese por cada camada (MVAU) instanciada.
"""

import os
import json
import glob
import pandas as pd
import re

BASE_DIR = "/home/arthurely/Desktop/finn_chi2p/hara/exhaustive_hw_builds"

def extract_topology(session_name):
    """Extrai a topologia base do nome da pasta (ex: CIFAR10_1W1A_2026... -> CIFAR10_1W1A)"""
    match = re.search(r"^(.*?)_202\d", session_name)
    if match:
        return match.group(1)
    return session_name.split('/')[0]

def main():
    search_pattern = os.path.join(BASE_DIR, "*", "run*", "time_per_step.json")
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        print(f"[!] Nenhum arquivo time_per_step.json encontrado em {BASE_DIR}")
        return

    data = []
    
    for file_path in json_files:
        run_dir = os.path.dirname(file_path)
        parts = file_path.split(os.sep)
        session_name = parts[-3]
        run_name = parts[-2]
        topology = extract_topology(session_name)
        
        # 1. LER OS TEMPOS
        try:
            with open(file_path, 'r') as f:
                times = json.load(f)
        except Exception as e:
            print(f"Erro ao ler tempos {file_path}: {e}")
            continue

        # 2. LER O HARDWARE PARA CONTAR AS MVAUs
        hw_config_path = os.path.join(run_dir, "final_hw_config.json")
        mvau_count = 0
        if os.path.exists(hw_config_path):
            try:
                with open(hw_config_path, 'r') as f:
                    cfg = json.load(f)
                    # Conta quantas chaves no JSON têm "MVAU" no nome
                    mvau_count = sum(1 for k in cfg.keys() if "MVAU" in k)
            except Exception:
                pass
                
        # Se por acaso não achar, salva como NaN para não sujar a média
        if mvau_count == 0:
            mvau_count = pd.NA

        row = {"Topology": topology, "Session": session_name, "Run": run_name, "MVAU_Count": mvau_count}
        row.update(times)
        data.append(row)
                
    df = pd.DataFrame(data)
    
    # Isolar apenas as colunas de tempo
    step_columns = [col for col in df.columns if col.startswith("step_")]
    
    # Agrupar por topologia e tirar a média de cada step e contagem de MVAU
    grouped = df.groupby("Topology").agg({**{c: 'mean' for c in step_columns}, "MVAU_Count": 'mean'})
    
    # Somar o tempo total da build
    grouped["Total_Time_s"] = grouped[step_columns].sum(axis=1)
    grouped["Total_Time_min"] = grouped["Total_Time_s"] / 60.0
    
    # Calcular custo por MVAU
    grouped["Time_per_MVAU_s"] = grouped["Total_Time_s"] / grouped["MVAU_Count"]
    grouped["Time_per_MVAU_min"] = grouped["Time_per_MVAU_s"] / 60.0
    
    print("\n" + "="*95)
    print("⏱️  MÉDIA DE TEMPO DE COMPILAÇÃO E ESCALABILIDADE POR MVAU (FINN)")
    print("="*95)
    
    for topo, row in grouped.iterrows():
        total_s = row["Total_Time_s"]
        total_m = row["Total_Time_min"]
        mvaus = row["MVAU_Count"]
        time_per_mvau_m = row["Time_per_MVAU_min"]
        
        print(f"\n📌 Topologia: {topo:<15} | MVAUs: {mvaus:.1f}")
        print(f"   Tempo Total: {total_m:>5.1f} min  |  Custo de Compilação por Camada: ~{time_per_mvau_m:.1f} min/MVAU")
        print("-" * 95)
        
        # Ordenar os steps do mais demorado para o mais rápido
        steps_sorted = row[step_columns].sort_values(ascending=False)
        synthesis_time = 0
        
        for step_name, time_s in steps_sorted.items():
            if pd.isna(time_s): continue
            pct = (time_s / total_s) * 100 if total_s > 0 else 0
            
            # Destacar visualmente as etapas pesadas de síntese
            is_synth = "create_stitched_ip" in step_name or "synthesis" in step_name
            if is_synth:
                synthesis_time += time_s
                print(f" 🚨 {step_name:<32}: {time_s:>7.1f}s  |  {pct:>5.1f}%")
            else:
                print(f"    {step_name:<32}: {time_s:>7.1f}s  |  {pct:>5.1f}%")
        
        synth_pct = (synthesis_time / total_s) * 100 if total_s > 0 else 0
        print("-" * 95)
        print(f" ➔ Tempo gasto APENAS na Síntese Física: {synthesis_time/60:.1f} min ({synth_pct:.1f}%)")

if __name__ == "__main__":
    main()