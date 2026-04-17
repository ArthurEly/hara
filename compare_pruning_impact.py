import sys, json
sys.path.insert(0, '.')
from ai.multi_module_learner import MultiModuleLearner

def run_prediction(learner, onnx_path, cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    depths = {k.replace('_rtl','').replace('_hls',''): v['depth'] 
              for k, v in cfg.items() if isinstance(v, dict) and 'depth' in v}
    return learner.predict(onnx_path, [cfg], precomputed_depths=[depths])[0]

# --- Configurações ---
learner = MultiModuleLearner('ai/retrieval/results/trained_models')

# 1. Rede Original (Densa) - Run 1
onnx_dense = 'exhaustive_hw_builds/SAT6_T2W2_2026-04-10_19-04-21/run1_baseline_folded/intermediate_models/step_generate_estimate_reports.onnx'
cfg_dense = 'exhaustive_hw_builds/SAT6_T2W2_2026-04-10_19-04-21/run1_baseline_folded/final_hw_config.json'

# 2. Rede Prunada (DROP5) - Run 9
onnx_pruned = 'models/SAT6_SEC/final_optimized_drop5_model_estimate.onnx'
cfg_pruned = 'hls_candidate_builds/hls_cand01_drop5_run9/final_hw_config.json'

# --- Predições ---
pred_dense = run_prediction(learner, onnx_dense, cfg_dense)
pred_pruned = run_prediction(learner, onnx_pruned, cfg_pruned)

print(f"\n{'='*70}")
print(f" IMPACTO DO PRUNING NAS MVAUS (XGBoost Prediction)")
print(f"{'='*70}")
print(f"{'Módulo':<20} | {'LUTs (Rede Densa)':<20} | {'LUTs (Prunada DROP5)'}")
print(f"{'-'*70}")

total_mvau_dense = 0
total_mvau_pruned = 0

for i in range(5):
    name = f"MVAU_hls_{i}"
    
    # Pega as LUTs totais da MVAU (lidando com as variações de nomenclatura do dict)
    lut_dense = pred_dense['_details'].get(name, {}).get('Total LUT', pred_dense['_details'].get(name, {}).get('Total LUTs', 0))
    lut_pruned = pred_pruned['_details'].get(name, {}).get('Total LUT', pred_pruned['_details'].get(name, {}).get('Total LUTs', 0))
    
    total_mvau_dense += lut_dense
    total_mvau_pruned += lut_pruned
    
    # Calcula a redução
    if lut_dense > 0:
        reducao = ((lut_dense - lut_pruned) / lut_dense) * 100
        print(f"{name:<20} | {lut_dense:>10.0f} LUTs{'':<5} | {lut_pruned:>10.0f} LUTs (-{reducao:.1f}%)")
    else:
        print(f"{name:<20} | {'N/A':>10}{'':<10} | {'N/A':>10}")

print(f"{'-'*70}")
print(f"{'TOTAL (Só MVAUs)':<20} | {total_mvau_dense:>10.0f} LUTs{'':<5} | {total_mvau_pruned:>10.0f} LUTs (-{((total_mvau_dense-total_mvau_pruned)/total_mvau_dense)*100:.1f}%)")
print(f"{'='*70}\n")