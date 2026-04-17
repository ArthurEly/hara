import sys, json, os
sys.path.insert(0, '.')
from ai.multi_module_learner import MultiModuleLearner, parse_util_rpt

# --- Configurações da Rede Prunada (DROP5) ---
learner = MultiModuleLearner('ai/retrieval/results/trained_models')
onnx_path = 'models/SAT6_SEC/final_optimized_drop5_model_estimate.onnx'
run_dir = 'hls_candidate_builds/synth_cand01_drop5_run9'
cfg_path = os.path.join("/home/arthurely/Desktop/finn_chi2p/hara/hls_candidate_builds/hls_cand01_drop5_run9", 'final_hw_config.json')
rpt_path = os.path.join(run_dir, 'stitched_ip', 'finn_design_partition_util.rpt')

# 1. Carrega Configurações e Depths
with open(cfg_path, 'r') as f:
    cfg = json.load(f)

depths = {k.replace('_rtl','').replace('_hls',''): v['depth'] 
          for k, v in cfg.items() if isinstance(v, dict) and 'depth' in v}

# 2. Faz a Predição
pred = learner.predict(onnx_path, [cfg], precomputed_depths=[depths])[0]
details = pred.get("_details", {})

# 3. Lê o Ground Truth do Vivado
gt_util = parse_util_rpt(rpt_path)
has_gt = bool(gt_util)

print(f"\n{'='*75}")
print(f" ANÁLISE DE ERRO: REDE PRUNADA (DROP5) - Predição vs Realidade")
print(f"{'='*75}")

if not has_gt:
    print(f"[!] Arquivo de Ground Truth não encontrado em:\n{rpt_path}")
    sys.exit(1)

hdr = f"{'Módulo':<35} | {'LUT (Real)':<10} | {'LUT (Pred)':<10} | {'Erro (%)'}"
print(hdr)
print("-" * 75)

import re
all_names = sorted(set(list(details.keys()) + list(gt_util.keys())))

total_real = 0
total_pred = 0

for name in all_names:
    p = details.get(name, {})
    gt = gt_util.get(name)

    # Tenta dar match com o nome sem sufixo se o FINN renomeou no RPT
    if gt is None:
        clean = re.sub(r'_(rtl|hls)_(\d+)$', r'_\2', name)
        gt = gt_util.get(clean)

    p_lut = int(round(p.get("Total LUT", p.get("Total LUTs", 0))))
    
    if gt:
        g_lut = gt["LUT"]
        total_real += g_lut
        total_pred += p_lut
        
        err = ((p_lut - g_lut) / max(1, g_lut)) * 100
        
        # Destaca erros absurdos (maiores que 20%)
        alert = " 🚨" if abs(err) > 20 and g_lut > 50 else ""
        print(f"{name:<35} | {g_lut:<10} | {p_lut:<10} | {err:>+6.1f}%{alert}")

print("-" * 75)
print(f"{'TOTAL ACUMULADO':<35} | {total_real:<10} | {total_pred:<10} | {((total_pred - total_real)/max(1, total_real))*100:>+6.1f}%")
print(f"{'='*75}\n")