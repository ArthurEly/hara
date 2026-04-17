import csv
import os

# Fatores de Elasticidade (QoR) para o paper
# GAMMA = inf -> Accuracy-First (SEC puro, só pruna o que não couber)
# GAMMA = 8   -> Balanced QoR (Aceita perder 1% de acc se ganhar >8% FPS)
# GAMMA = 0   -> Max Throughput (Ignora acc, pega o maior FPS até o limite de 5% drop)
GAMMAS = [float('inf'), 8.0, 0.0]

def load_greedy_baseline(csv_path, fpga, targets):
    runs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'success':
                runs.append({
                    'run_id': row['hw_name'],
                    'dur': float(row['duration_in_seconds']),
                    'lut': float(row['Total LUTs']),
                    'ff': float(row['FFs']),
                    'bram': float(row['BRAM (36k)']),
                    'dsp': float(row['DSP Blocks']),
                    'fps': float(row['estimated_throughput_fps'])
                })

    results = {}
    for t in targets:
        max_lut = fpga['LUTs'] * (t / 100.0)
        max_ff = fpga['FFs'] * (t / 100.0)
        max_bram = fpga['BRAM'] * (t / 100.0)
        max_dsp = fpga['DSP'] * (t / 100.0)
        
        best_run = None
        best_idx = -1
        
        for i, r in enumerate(runs):
            if r['lut'] <= max_lut and r['ff'] <= max_ff and r['bram'] <= max_bram and r['dsp'] <= max_dsp:
                util = max(r['lut']/fpga['LUTs'], r['ff']/fpga['FFs'], r['bram']/fpga['BRAM'], r['dsp']/fpga['DSP'])
                if best_run is None or util > max(best_run['lut']/fpga['LUTs'], best_run['ff']/fpga['FFs'], best_run['bram']/fpga['BRAM'], best_run['dsp']/fpga['DSP']):
                    best_run = r
                    best_idx = i
        
        if best_run:
            total_time = sum(r['dur'] for r in runs[:best_idx+1])
            results[t] = {
                'fps': best_run['fps'],
                'builds': best_idx + 1,
                'time_h': total_time / 3600.0,
                'dsp': best_run['dsp']
            }
    return results

def load_predse_data(csv_path, gamma):
    hara_winners = {}
    if not os.path.exists(csv_path):
        return hara_winners
        
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            budget = int(float(row['budget_pct']))
            fps = float(row['fps'])
            drop = float(row['acc_drop']) # 0.0 para PREBUILT
            model = row['model']
            dsp = float(row['pred_DSP'])
            max_util_pred = float(row['max_util_pct'])
            
            # Tolerância de Machine Learning na área (+1.5%)
            if max_util_pred <= (budget + 1.5):
                
                # Restrição HARA: Drop máx tolerado = 5.0%
                if drop <= 5.0:
                    acc_retention = (100.0 - drop) / 100.0
                    
                    if gamma == float('inf'):
                        # Accuracy First (Lexicographic)
                        if budget not in hara_winners:
                            hara_winners[budget] = {'fps': fps, 'model': model, 'drop': drop, 'dsp': dsp}
                        else:
                            curr = hara_winners[budget]
                            if drop < curr['drop']:
                                hara_winners[budget] = {'fps': fps, 'model': model, 'drop': drop, 'dsp': dsp}
                            elif drop == curr['drop'] and fps > curr['fps']:
                                hara_winners[budget] = {'fps': fps, 'model': model, 'drop': drop, 'dsp': dsp}
                    else:
                        # Elastic QoR Score
                        score = fps * (acc_retention ** gamma)
                        if budget not in hara_winners:
                            hara_winners[budget] = {'fps': fps, 'model': model, 'drop': drop, 'dsp': dsp, 'score': score}
                        else:
                            curr = hara_winners[budget]
                            if score > curr['score']:
                                hara_winners[budget] = {'fps': fps, 'model': model, 'drop': drop, 'dsp': dsp, 'score': score}
                        
    return hara_winners

def format_cell(data):
    if not data:
        return "N/A"
    drop_str = f"(-{data['drop']:.0f}%)" if data['drop'] > 0 else "(0%)"
    return f"{data['fps']:.0f} FPS {drop_str}"

def main():
    greedy_csv = '/home/arthurely/Desktop/finn_chi2p/hara/exhaustive_hw_builds/SAT6_T2W2_2026-04-15_23-19-16/hardware_summary.csv'
    predse_csv = '/home/arthurely/Desktop/finn_chi2p/hara/models/SAT6_SEC/predse_summary.csv'
    
    fpga = {'LUTs': 53200, 'FFs': 106400, 'BRAM': 140, 'DSP': 220}
    targets = [10, 20, 30, 40]
    
    greedy_data = load_greedy_baseline(greedy_csv, fpga, targets)
    
    # Carrega os 3 perfis do HARA
    hara_inf = load_predse_data(predse_csv, float('inf'))
    hara_8   = load_predse_data(predse_csv, 8.0)
    hara_0   = load_predse_data(predse_csv, 0.0)
    
    print("="*120)
    print(" EXPERIMENTO SBCCI: COMPORTAMENTO ADAPTATIVO DO HARA (ELASTIC QoR vs GREEDY)")
    print("="*120)
    print(f"{'Budget':<7} | {'Greedy (Unpruned)':<26} | {'HARA γ=∞ (Accuracy-First)':<26} | {'HARA γ=8 (Balanced)':<26} | {'HARA γ=0 (Max FPS)'}")
    print("-" * 120)
    
    for t in targets:
        g = greedy_data.get(t)
        g_str = f"{g['fps']:.0f} FPS (0%)" if g else "FAILED"
        
        c_inf = format_cell(hara_inf.get(t))
        c_8   = format_cell(hara_8.get(t))
        c_0   = format_cell(hara_0.get(t))
        
        print(f"{t}% Area | {g_str:<26} | {c_inf:<26} | {c_8:<26} | {c_0}")
        
        # Sub-linha para os DSPs e Builds (apenas para ilustrar o ganho)
        if g:
            b_str = f"{g['builds']} builds / {g['dsp']:.0f} DSP"
        else:
            b_str = "-"
            
        dsp_inf = f"1 build  / {hara_inf[t]['dsp']:.0f} DSP" if t in hara_inf else "-"
        dsp_8   = f"1 build  / {hara_8[t]['dsp']:.0f} DSP" if t in hara_8 else "-"
        dsp_0   = f"1 build  / {hara_0[t]['dsp']:.0f} DSP" if t in hara_0 else "-"
        
        print(f"{'':<7} | {b_str:<26} | {dsp_inf:<26} | {dsp_8:<26} | {dsp_0}")
        print("-" * 120)

if __name__ == "__main__":
    main()