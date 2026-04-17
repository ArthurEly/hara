#!/usr/bin/env python3
"""
run_predse_sweep.py — PreDSE: Pre-Synthesis Design Space Exploration

For each model variant in fps_campaign_results, this script:
  1. Reads the fps_map.csv (with FPS + folding config per run)
  2. Loads the ONNX topology ONCE per model (from run0's estimate)
  3. Passes ALL foldings as a BATCH to the MultiModuleLearner
  4. Outputs a complete CSV: FPS × Predicted LUT/FF/BRAM/DSP per config
  5. Finds the first-shot (best FPS within area budget) for each A*

Usage (inside FINN Docker):
    python3 scripts/run_predse_sweep.py
"""

import os
import sys
import csv
import json
import glob
import re

hara_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(hara_dir)

from ai.multi_module_learner import MultiModuleLearner

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CAMPAIGN_DIR = os.path.join(hara_dir, "fps_campaign_results")
OUTPUT_DIR = os.path.join(hara_dir, "models", "SAT6_SEC")
MODELS_DIR = os.path.join(hara_dir, "ai", "retrieval", "results", "trained_models")
if not os.path.isdir(MODELS_DIR):
    MODELS_DIR = os.path.join(hara_dir, "ai", "hardware_models")

FPGA_LIMITS = {
    "Total LUTs": 53200,
    "FFs": 106400,
    "BRAM (36k)": 140,
    "DSP Blocks": 220,
}

A_STAR_BUDGETS = [0.10, 0.20, 0.30, 0.40]

ACC_DROP_MAP = {
    "PREBUILT": 0, "DROP1": 1, "DROP2": 2,
    "DROP3": 3, "DROP4": 4, "DROP5": 5,
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def find_campaign_dirs():
    pattern = os.path.join(CAMPAIGN_DIR, "SAT6_T2W2_*")
    return sorted(glob.glob(pattern))


def extract_model_label(dirname):
    basename = os.path.basename(dirname)
    m = re.match(r"(SAT6_T2W2_\w+?)_\d{8}_\d{6}", basename)
    return m.group(1) if m else basename


def get_acc_drop(label):
    for key, val in ACC_DROP_MAP.items():
        if key in label:
            return val
    return 0


def parse_fps_map(csv_path):
    """Parse fps_map.csv into list of (run_id, fps, folding_dict)."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = int(row["run_id"])
            fps = float(row["estimated_fps"])
            folding = json.loads(row["folding_config"])
            rows.append((run_id, fps, folding))
    return rows


def find_cycles_json(campaign_dir, run_id):
    """Find estimate_layer_cycles.json for a given run_id (run{id}_* pattern)."""
    for entry in sorted(os.listdir(campaign_dir)):
        if entry.startswith(f"run{run_id}_") or entry == f"run{run_id}":
            p = os.path.join(campaign_dir, entry, "report", "estimate_layer_cycles.json")
            if os.path.exists(p):
                return p
    return None


def load_cycles_per_folding(campaign_dir, fps_entries):
    """Load estimate_layer_cycles.json for each run in fps_entries.
    Returns list[dict], one cycles dict per folding. Empty dict = fallback to formula."""
    cache = {}
    result = []
    n_found = 0
    for run_id, _, _ in fps_entries:
        if run_id not in cache:
            path = find_cycles_json(campaign_dir, run_id)
            if path:
                with open(path) as f:
                    cache[run_id] = json.load(f)
                n_found += 1
            else:
                cache[run_id] = {}
        result.append(cache[run_id])
    if n_found:
        print(f"  -> estimate_layer_cycles.json carregado para {n_found}/{len(cache)} runs")
    else:
        print(f"  -> AVISO: nenhum estimate_layer_cycles.json encontrado — usando fórmula MH×MW")
    return result


def find_canonical_onnx(campaign_dir):
    """Find ONE representative ONNX for the entire model.
    Uses run0_get_initial_fold's estimate ONNX (the first build)."""
    candidates = [
        os.path.join(campaign_dir, "run0_get_initial_fold", "intermediate_models", "step_generate_estimate_reports.onnx"),
        os.path.join(campaign_dir, "run1_baseline_folded", "intermediate_models", "step_generate_estimate_reports.onnx"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Fallback: find any available one
    for entry in sorted(os.listdir(campaign_dir)):
        p = os.path.join(campaign_dir, entry, "intermediate_models", "step_generate_estimate_reports.onnx")
        if os.path.exists(p):
            return p
    return None


def utilization_pct(pred, resource_key):
    val = pred.get(resource_key, 0)
    limit = FPGA_LIMITS.get(resource_key, 1)
    return (val / limit) * 100.0 if limit > 0 else 0.0


def max_utilization(pred):
    return max(
        utilization_pct(pred, "Total LUTs"),
        utilization_pct(pred, "FFs"),
        utilization_pct(pred, "BRAM (36k)"),
        utilization_pct(pred, "DSP Blocks"),
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_predse():
    print("=" * 60)
    print(" PreDSE — Pre-Synthesis Design Space Exploration")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load learner ONCE
    print(f"\n[1] Loading MultiModuleLearner from {MODELS_DIR}")
    learner = MultiModuleLearner(MODELS_DIR)
    if not learner.is_loaded():
        print("[!] No models loaded. Exiting.")
        return
    print(f"    ✓ {len(learner._models)} specialist models loaded.\n")

    # Find campaign dirs
    campaign_dirs = find_campaign_dirs()
    if not campaign_dirs:
        print(f"[!] No campaign directories found in {CAMPAIGN_DIR}")
        return
    
    print(f"[2] Found {len(campaign_dirs)} model variants:")
    for d in campaign_dirs:
        print(f"    - {os.path.basename(d)}")

    # Process each model variant
    all_first_shots = []
    
    for campaign_path in campaign_dirs:
        label = extract_model_label(campaign_path)
        acc_drop = get_acc_drop(label)
        
        csv_path = os.path.join(campaign_path, "fps_map.csv")
        if not os.path.exists(csv_path):
            print(f"\n[!] Skipping {label}: no fps_map.csv found")
            continue
        
        print(f"\n{'='*60}")
        print(f" Processing: {label} (Accuracy Drop: {acc_drop}%)")
        print(f"{'='*60}")
        
        # Find the canonical ONNX for this model (load ONCE!)
        onnx_path = find_canonical_onnx(campaign_path)
        if not onnx_path:
            print(f"  [!] No ONNX found for {label}, skipping")
            continue
        print(f"  -> Using ONNX: {os.path.basename(os.path.dirname(os.path.dirname(onnx_path)))}/...")

        # Parse ALL folding configs
        fps_entries = parse_fps_map(csv_path)
        print(f"  -> {len(fps_entries)} folding configurations to evaluate")
        
        # Extract all foldings for BATCH prediction
        all_foldings = [folding for _, _, folding in fps_entries]

        # --- HYPOTHETICAL HARDWARE INJECTION ---
        # Force all MVAU layers to use DSP and BRAM for area study
        for folding in all_foldings:
            for layer_name, cfg in folding.items():
                if "MVAU" in layer_name:
                    cfg["resType"] = "dsp"
                    cfg["ram_style"] = "block"

        # --- STEP A: PREDICT FIFO DEPTHS (before area prediction) ---
        print(f"  -> [Step A] Predicting FIFO depths for {len(all_foldings)} folding configs...")
        all_cycles = load_cycles_per_folding(campaign_path, fps_entries)
        all_fifo_depths = learner.predict_fifo_depths_batch(onnx_path, all_foldings,
                                                            cycles_per_folding=all_cycles)
        if all_fifo_depths:
            depth_vals = list(all_fifo_depths[0].values())
            if depth_vals:
                non_trivial = sum(1 for d in depth_vals if d > 2)
                print(f"  -> Folding-0: {len(depth_vals)} FIFOs | {non_trivial} com depth>2 "
                      f"| max={max(depth_vals)} | median={sorted(depth_vals)[len(depth_vals)//2]}")

        # --- STEP B: PREDICT AREA using the pre-computed depths ---
        print(f"  -> [Step B] Running batch area prediction ({len(all_foldings)} configs with MVAU=DSP+BRAM)...")
        predictions = learner.predict(onnx_path, all_foldings, precomputed_depths=all_fifo_depths)
        
        # Build results
        results = []
        for i, (run_id, fps, _) in enumerate(fps_entries):
            pred = predictions[i] if i < len(predictions) else {}
            
            luts = pred.get("Total LUTs", 0)
            ffs = pred.get("FFs", 0)
            bram = pred.get("BRAM (36k)", 0)
            dsp = pred.get("DSP Blocks", 0)
            max_util = max_utilization(pred)
            
            results.append({
                "run_id": run_id,
                "estimated_fps": round(fps, 2),
                "pred_LUTs": int(luts),
                "logic_LUTs": int(pred.get("Logic LUTs", 0)),
                "lutrams": int(pred.get("LUTRAMs", 0)),
                "srls": int(pred.get("SRLs", 0)),
                "pred_FFs": int(ffs),
                "pred_BRAM": round(bram, 1),
                "pred_DSP": int(dsp),
                "LUT_util_pct": round(utilization_pct(pred, "Total LUTs"), 2),
                "FF_util_pct": round(utilization_pct(pred, "FFs"), 2),
                "BRAM_util_pct": round(utilization_pct(pred, "BRAM (36k)"), 2),
                "DSP_util_pct": round(utilization_pct(pred, "DSP Blocks"), 2),
                "max_util_pct": round(max_util, 2),
                "acc_drop": acc_drop,
                "model": label,
                "fifo_depths": pred.get("fifo_depths", {})
            })
        
        if not results:
            print(f"  [!] No valid results for {label}")
            continue
        
        # Save per-model CSV
        out_csv = os.path.join(OUTPUT_DIR, f"predse_{label}.csv")
        fieldnames = list(results[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"  ✓ Saved {len(results)} predictions to {out_csv}")
        
        # Find first-shot for each budget
        for a_star in A_STAR_BUDGETS:
            budget_pct = a_star * 100
            feasible = [r for r in results if r["max_util_pct"] <= budget_pct]
            
            if feasible:
                best = max(feasible, key=lambda x: x["estimated_fps"])
                all_first_shots.append({
                    "budget_pct": budget_pct,
                    "model": label,
                    "acc_drop": acc_drop,
                    "first_shot_run": best["run_id"],
                    "fps": best["estimated_fps"],
                    "max_util_pct": best["max_util_pct"],
                    "pred_LUTs": best["pred_LUTs"],
                    "logic_LUTs": best["logic_LUTs"],
                    "lutrams": best["lutrams"],
                    "srls": best["srls"],
                    "pred_FFs": best["pred_FFs"],
                    "pred_BRAM": best["pred_BRAM"],
                    "pred_DSP": best["pred_DSP"],
                    "onnx_path": onnx_path,
                    "folding_config": next(f for rid, f in zip([r[0] for r in fps_entries], all_foldings) if rid == best["run_id"]),
                    "fifo_depths": best["fifo_depths"]
                })
                print(f"  [A*={budget_pct:.0f}%] First-shot: run#{best['run_id']} "
                      f"→ {best['estimated_fps']:.0f} FPS @ {best['max_util_pct']:.1f}% util "
                      f"(LUT:{best['pred_LUTs']} [Log:{best['logic_LUTs']} RAM:{best['lutrams']} SRL:{best['srls']}] FF:{best['pred_FFs']} BRAM:{best['pred_BRAM']} DSP:{best['pred_DSP']})")
            else:
                print(f"  [A*={budget_pct:.0f}%] No feasible configuration!")

    # Save summary
    if all_first_shots:
        summary_csv = os.path.join(OUTPUT_DIR, "predse_summary.csv")
        fieldnames = list(all_first_shots[0].keys())
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_first_shots)
        
        print(f"\n{'='*60}")
        print(f" SUMMARY saved to: {summary_csv}")
        print(f"{'='*60}")

        # Save FIFO depths for debugging
        depths_dir = os.path.join(OUTPUT_DIR, "fifo_depths_debug")
        os.makedirs(depths_dir, exist_ok=True)
        print(f" DEBUG: Verificando depths para {len(all_first_shots)} configurações Pareto...")
        for entry in all_first_shots:
            if entry.get("fifo_depths"):
                depth_file = os.path.join(depths_dir, f"depths_{entry['model']}_{int(entry['budget_pct'])}pct.json")
                with open(depth_file, "w") as f:
                    json.dump(entry["fifo_depths"], f, indent=4)
                print(f"   -> Salvo: {os.path.basename(depth_file)} ({len(entry['fifo_depths'])} FIFOs)")
            else:
                print(f"   -> AVISO: {entry['model']}@{int(entry['budget_pct'])}% não possui fifo_depths!")
        
        # Print the Pareto winners per budget
        print("\n PARETO-OPTIMAL FIRST-SHOTS (max FPS within acc_drop <= 5%):")
        print("-" * 110)
        print(f"{'Budget':>8}  {'Model':<28} {'Drop':>5} {'FPS':>10} {'Max Util':>10} {'LUTs':>8} {'FFs':>8} {'BRAM':>6} {'DSP':>5}")
        print("-" * 110)
        for a_star in A_STAR_BUDGETS:
            budget_pct = a_star * 100
            candidates = [r for r in all_first_shots 
                         if r["budget_pct"] == budget_pct and r["acc_drop"] <= 5]
            
            baseline = next((r for r in candidates if r["acc_drop"] == 0), None)
            winner = max(candidates, key=lambda x: x["fps"]) if candidates else None

            if baseline:
                print(f"{budget_pct:>7.0f}%  {baseline['model']:<28} {baseline['acc_drop']:>4}% "
                      f"{baseline['fps']:>10.0f} {baseline['max_util_pct']:>9.1f}% "
                      f"{baseline['pred_LUTs']:>8} ({baseline['logic_LUTs']}/{baseline['lutrams']}/{baseline['srls']}) "
                      f"{baseline['pred_FFs']:>8} {baseline['pred_BRAM']:>5.1f} {baseline['pred_DSP']:>5} (Baseline)")
            
            if winner and winner != baseline:
                print(f"{budget_pct:>7.0f}%  {winner['model']:<28} {winner['acc_drop']:>4}% "
                      f"{winner['fps']:>10.0f} {winner['max_util_pct']:>9.1f}% "
                      f"{winner['pred_LUTs']:>8} ({winner['logic_LUTs']}/{winner['lutrams']}/{winner['srls']}) "
                      f"{winner['pred_FFs']:>8} {winner['pred_BRAM']:>5.1f} {winner['pred_DSP']:>5} (Winner)")
            
            if candidates: print("-" * 110)


if __name__ == "__main__":
    run_predse()
