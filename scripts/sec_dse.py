import os
import sys
import glob
import pandas as pd
import json
import re
import shutil

hara_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(hara_dir)

from orchestrator import HARAv2Orchestrator

TARGET_DIR = os.path.join(hara_dir, "models", "SAT6_SEC")
os.makedirs(TARGET_DIR, exist_ok=True)

A_STAR_BUDGETS = [0.10, 0.20, 0.30, 0.40]

def parse_accuracy_drop(file_path):
    match = re.search(r"drop(\d+)", file_path)
    if match: return int(match.group(1))
    elif "baseline" in file_path: return 0
    return 999 

def run_sec_dse_and_map():
    # 1. Move _estimate files from SAT6 to SAT6_SEC
    source_dir = os.path.join(hara_dir, "models", "SAT6")
    for fp in glob.glob(os.path.join(source_dir, "*_estimate.onnx")):
        shutil.move(fp, os.path.join(TARGET_DIR, os.path.basename(fp)))
        
    estimate_files = glob.glob(os.path.join(TARGET_DIR, "*_estimate.onnx"))
    estimate_files.sort()
    
    print("="*60)
    print(" HARA SEC (Strict Edge Constraint) Design Space Exploration")
    print("="*60)
    
    workspace = os.path.join(hara_dir, "configs", "sat6_sec", "orchestrator")
    os.makedirs(workspace, exist_ok=True)
    
    if not estimate_files:
        print("[!] No ONNX estimate files found in SAT6_SEC.")
        return
        
    models_path = os.path.join(hara_dir, "ai", "hardware_models")
    if not os.path.exists(models_path):
        models_path = os.path.join(hara_dir, "ai", "retrieval", "results", "trained_models")

    orchestrator = HARAv2Orchestrator(
        onnx_path=estimate_files[0],
        build_dir=workspace,
        target_fps=0,
        area_budget=1.0, 
        simulate=True,
        models_dir=models_path
    )
    if orchestrator.learner is None or not getattr(orchestrator.learner, "is_loaded", lambda: True)():
        print(f"[!] Failed to load XGBoost learner from {models_path}.")
        return

    # 2. GENERATE FPS MAPS
    print("\n--- Generating FPS Maps & Predicting Area for all topologies ---")
    all_mapped_configs = {} # map model name -> list of configs
    
    for onnx_path in estimate_files:
        model_name = os.path.basename(onnx_path).replace("_estimate.onnx", "")
        print(f" -> Mapping: {model_name}")
        
        # Uses FinnCycleEstimator inside
        map_configs = orchestrator.explorer._map_theoretical_design_space(onnx_path)
        
        if not map_configs:
            print(f"   [!] Failed to generate map for {model_name}.")
            continue
            
        foldings = [c["folding"] for c in map_configs]
        predictions = orchestrator.learner.predict(onnx_path, foldings)
        
        csv_data = []
        for i, c in enumerate(map_configs):
            pred = predictions[i]
            csv_data.append({
                "hw_name_base": f"{model_name}_try{i}",
                "estimated_fps": c["fps_estimated"],
                "Total LUTs": pred.get("Total LUTs", 0),
                "FFs": pred.get("FFs", 0),
                "BRAM (36k)": pred.get("BRAM (36k)", 0),
                "DSP Blocks": pred.get("DSP Blocks", 0),
                "folding_config": json.dumps(c["folding"])
            })
            c["pred"] = pred
            
        all_mapped_configs[onnx_path] = map_configs
            
        df = pd.DataFrame(csv_data)
        out_csv = os.path.join(TARGET_DIR, f"fps_map_{model_name}.csv")
        df.to_csv(out_csv, index=False)
        print(f"   => Saved map to {out_csv}")
        
    # 3. Sweep A* limits
    best_results = {A: None for A in A_STAR_BUDGETS}
    limits_max = orchestrator.fpga_limits
    
    print("\n" + "="*60)
    print(" EVALUATING SEC BUDGETS")
    print("="*60)
    for A in A_STAR_BUDGETS:
        print(f"\n[!] SWEEPING AREA BUDGET: {int(A*100)}%")
        
        best_fps_for_budget = 0
        best_candidate = None
        
        for onnx_path, map_configs in all_mapped_configs.items():
            drop = parse_accuracy_drop(onnx_path)
            if drop > 5: continue
            
            best_local_fps = 0
            best_local_cfg = None
            
            for c in map_configs:
                pred = c["pred"]
                fits = True
                max_pct = 0.0
                
                for res, limit in limits_max.items():
                    req = pred.get(res, 0)
                    available = limit * A
                    pct = req / limit if limit > 0 else 0
                    if req > available:
                        fits = False
                        break
                    max_pct = max(max_pct, pct)
                
                if fits and c["fps_estimated"] > best_local_fps:
                    best_local_fps = c["fps_estimated"]
                    best_local_cfg = c
                    best_local_cfg["max_pct"] = max_pct
                    
            if best_local_cfg and best_local_fps > best_fps_for_budget:
                best_fps_for_budget = best_local_fps
                best_candidate = {
                    "model": os.path.basename(onnx_path).replace("_estimate.onnx", ""),
                    "drop": drop,
                    "fps": best_local_fps,
                    "max_pct": best_local_cfg["max_pct"],
                    "pred": best_local_cfg["pred"],
                    "folding": best_local_cfg["folding"]
                }
                print(f"  -> {best_candidate['model']}: New Lead! {best_local_fps:.2f} FPS (Util: {best_local_cfg['max_pct']*100:.2f}%)")

        if best_candidate:
            best_results[A] = best_candidate
            print(f" => WINNER for {int(A*100)}% BUDGET: {best_candidate['model']} (FPS: {best_candidate['fps']:.2f})")
        else:
            print(f" => NO WINNER for {int(A*100)}% BUDGET (none fit).")

    print("\n" + "="*60)
    print(" SEC FINAL EXPLORATION RESULTS")
    print("="*60)
    for A in A_STAR_BUDGETS:
        res = best_results[A]
        if res:
            print(f"[{int(A*100)}% Budget] Model: {res['model']}")
            print(f"  -> Accuracy Drop: {res['drop']}%")
            print(f"  -> Perf: {res['fps']:.2f} FPS")
            print(f"  -> Max Hardware Utilization: {res['max_pct']*100:.2f}% / {A*100}% limit")
            print(f"  -> LUT Utilization: {res['pred'].get('Total LUTs', 0):.0f} / {limits_max['Total LUTs']*A:.0f}")
            print(f"  -> BRAM Utilization: {res['pred'].get('BRAM (36k)', 0):.1f} / {limits_max['BRAM (36k)']*A:.1f}")
        else:
            print(f"[{int(A*100)}% Budget] No valid accelerators available.")

if __name__ == "__main__":
    run_sec_dse_and_map()
