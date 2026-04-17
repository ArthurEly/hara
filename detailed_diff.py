import csv
import os
import re

def get_run_idx(name):
    # No hardware_summary.csv, vem como "run1_baseline_folded" -> extraímos o 1
    m = re.search(r'run(\d+)', str(name))
    return int(m.group(1)) if m else -1

def main():
    # Caminhos para os CSVs
    gt_csv = '/home/arthurely/Desktop/finn_chi2p/hara/exhaustive_hw_builds/SAT6_T2W2_2026-04-15_23-19-16/hardware_summary.csv'
    pred_csv = '/home/arthurely/Desktop/finn_chi2p/hara/models/SAT6_SEC/predse_SAT6_T2W2_PREBUILT.csv'

    if not os.path.exists(gt_csv) or not os.path.exists(pred_csv):
        print(f"[!] Arquivos CSV não encontrados. Verifique os caminhos.")
        print(f"GT: {os.path.exists(gt_csv)} | PRED: {os.path.exists(pred_csv)}")
        return

    # Carrega Ground Truth
    gt_runs = {}
    with open(gt_csv, 'r') as f:
        for r in csv.DictReader(f):
            if r['status'] == 'success':
                idx = get_run_idx(r['hw_name'])
                if idx != -1:
                    gt_runs[idx] = {
                        'lut': float(r['Total LUTs']),
                        'ff': float(r['FFs']),
                        'bram': float(r['BRAM (36k)']),
                        'dsp': float(r['DSP Blocks'])
                    }

    # Carrega Predições do PreDSE
    pred_runs = {}
    with open(pred_csv, 'r') as f:
        for r in csv.DictReader(f):
            # No predse_*.csv, a coluna se chama 'run_id' e já é um inteiro direto
            idx = int(float(r['run_id']))
            pred_runs[idx] = {
                'lut': float(r['pred_LUTs']),
                'ff': float(r['pred_FFs']),
                'bram': float(r['pred_BRAM']),
                'dsp': float(r['pred_DSP'])
            }

    # Limites da FPGA (Zynq 7020)
    fpga = {'LUTs': 53200, 'FFs': 106400, 'BRAM': 140, 'DSP': 220}

    print(f"\n{'='*145}")
    print(f" COMPARAÇÃO FINA: REAL (Vivado) vs PREDITO (HARA) — SAT6_T2W2_PREBUILT")
    print(f"{'='*145}")
    print(f"{'Run':<4} | {'LUT (Real / Pred / Err% / Util%)':<35} | {'FF (Real / Pred / Err% / Util%)':<35} | {'BRAM (Real / Pred / Err%)':<28} | {'DSP (Real/Pred)':<15} | {'10% FIT?'}")
    print("-" * 145)

    for idx in sorted(gt_runs.keys()):
        if idx not in pred_runs: 
            continue
        
        g = gt_runs[idx]
        p = pred_runs[idx]

        # Calcula o erro percentual e a utilização predita
        e_lut = (p['lut'] - g['lut']) / max(1, g['lut']) * 100
        u_lut = p['lut'] / fpga['LUTs'] * 100

        e_ff = (p['ff'] - g['ff']) / max(1, g['ff']) * 100
        u_ff = p['ff'] / fpga['FFs'] * 100

        e_bram = (p['bram'] - g['bram']) / max(1, g['bram']) * 100
        u_bram = p['bram'] / fpga['BRAM'] * 100

        e_dsp = (p['dsp'] - g['dsp']) / max(1, g['dsp']) * 100
        u_dsp = p['dsp'] / fpga['DSP'] * 100

        # Pega a utilização máxima predita (gargalo)
        max_util_p = max(u_lut, u_ff, u_bram, u_dsp)
        max_util_g = max(g['lut']/fpga['LUTs'], g['ff']/fpga['FFs'], g['bram']/fpga['BRAM'], g['dsp']/fpga['DSP']) * 100

        # Formata as strings
        lut_str = f"{g['lut']:.0f} / {p['lut']:.0f} / {e_lut:+.1f}% / {u_lut:.1f}%"
        ff_str  = f"{g['ff']:.0f} / {p['ff']:.0f} / {e_ff:+.1f}% / {u_ff:.1f}%"
        bram_str= f"{g['bram']:.1f} / {p['bram']:.1f} / {e_bram:+.1f}%"
        dsp_str = f"{g['dsp']:.0f} / {p['dsp']:.0f}"

        # Verifica se passou da borda dos 10.0%
        if max_util_g <= 10.0 and max_util_p > 10.0:
            fit_status = f"❌ FALSO NEG (P={max_util_p:.1f}%)"
        elif max_util_g > 10.0 and max_util_p <= 10.0:
            fit_status = f"⚠️ FALSO POS (P={max_util_p:.1f}%)"
        else:
            fit_status = f"{max_util_p:.1f}%"

        print(f"#{idx:<3} | {lut_str:<35} | {ff_str:<35} | {bram_str:<28} | {dsp_str:<15} | {fit_status}")
    print("-" * 145)

if __name__ == "__main__":
    main()