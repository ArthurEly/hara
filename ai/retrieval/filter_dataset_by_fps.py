#!/usr/bin/env python3
import pandas as pd
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Filter FIFO dataset by FPS")
    parser.add_argument("--min_fps", type=float, default=500.0, help="Minimum FPS to keep")
    parser.add_argument("--input", type=str, default="ai/retrieval/results/fifo_depth/fifo_backpressure_dataset.csv")
    parser.add_argument("--output", type=str, default="ai/retrieval/results/fifo_depth/fifo_backpressure_dataset_filtered.csv")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    df = pd.read_csv(args.input)
    print(f"Initial dataset size: {len(df)} FIFOs")

    sessions_runs = df[['session', 'run_name']].drop_duplicates()
    print(f"Total session-run pairs to check: {len(sessions_runs)}")

    valid_keys = set()
    drop_count = 0

    for _, row in sessions_runs.iterrows():
        session = row['session']
        run = row['run_name']
        
        # Path real no sistema do usuário
        path = f"exhaustive_hw_builds/{session}/{run}/report/estimate_network_performance.json"
        
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    perf = json.load(f)
                    # USANDO A CHAVE CORRETA IDENTIFICADA NA ANÁLISE
                    fps = perf.get('estimated_throughput_fps', 0)
                    if fps >= args.min_fps:
                        valid_keys.add((session, run))
                    else:
                        drop_count += 1
            except Exception:
                drop_count += 1
        else:
            # Se não achar o report, dropa por segurança (ou mantém)
            # Como o objetivo é ter qualidade, vamos assumir que sem report não entra
            drop_count += 1

    # Filtrar o dataframe total
    df['keep'] = df.apply(lambda r: (r['session'], r['run_name']) in valid_keys, axis=1)
    df_filtered = df[df['keep']].drop(columns=['keep'])

    print(f"Dropped {drop_count} builds with FPS < {args.min_fps} or missing report.")
    print(f"Final dataset size: {len(df_filtered)} FIFOs (from {len(df)})")

    # Salvar
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_filtered.to_csv(args.output, index=False)
    print(f"Saved filtered dataset to: {args.output}")

if __name__ == "__main__":
    main()
