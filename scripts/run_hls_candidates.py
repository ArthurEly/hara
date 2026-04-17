#!/usr/bin/env python3
"""
run_hls_candidates.py — HARA HLS Candidate Refinement Pipeline

Flow:
  1. Read predse_summary.csv
  2. Select top-N candidates using GAMMA scoring across all budgets/gammas
  3. For each unique candidate:
     a. Run FINN: steps up to step_set_fifo_depths (HLS IP gen + RTL-sim depths)
     b. Read real FIFO depths from final_hw_config.json
     c. Re-run HARA area prediction with real depths
     d. Re-score with GAMMA
  4. Print final ranking → winner proceeds to full build

GAMMA semantics (from compare_hara_greedy.py):
  γ=∞   → Accuracy-First  (lexicographic: min drop first, then max FPS)
  γ=8   → Balanced QoR   (trade 1% accuracy only if FPS gain > 8%)
  γ=0   → Max Throughput  (ignore accuracy up to 5% drop limit)

Usage (inside FINN Docker):
    python3 scripts/run_hls_candidates.py
    python3 scripts/run_hls_candidates.py --top-n 5 --gamma 8.0 --budget 30
    python3 scripts/run_hls_candidates.py --dry-run   # select only, no build
"""

import ast
import os
import sys
import csv
import json
import argparse
import subprocess

hara_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, hara_dir)

from ai.multi_module_learner import MultiModuleLearner

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(hara_dir, "ai", "retrieval", "results", "trained_models")
if not os.path.isdir(MODELS_DIR):
    MODELS_DIR = os.path.join(hara_dir, "ai", "hardware_models")

SAT6_SEC_DIR = os.path.join(hara_dir, "models", "SAT6_SEC")
OUTPUT_DIR   = SAT6_SEC_DIR
BUILD_DIR    = os.path.join(hara_dir, "hls_candidate_builds")

FPGA_PART   = "xc7z020clg400-1"
FPGA_LIMITS = {
    "Total LUTs": 53200,
    "FFs":        106400,
    "BRAM (36k)": 140,
    "DSP Blocks": 220,
}

# Gammas to consider during candidate selection
GAMMAS = [float('inf'), 8.0, 0.0]

# All area budgets used in predse_summary
BUDGET_PCTS = [10, 20, 30, 40]

# Hard constraint on accuracy drop
MAX_ACC_DROP = 5.0

# ML tolerance when checking if a candidate fits the budget (pp)
ML_TOLERANCE = 1.5

# FINN steps through step_set_fifo_depths (gets real FIFO depths via RTL sim)
HLS_DEPTH_STEPS = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    "step_streamline",
    "step_convert_to_hw",
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
]

# Model label → ONNX filename in SAT6_SEC_DIR
ONNX_MAP = {
    "SAT6_T2W2_PREBUILT": "sat6_t2_baseline_estimate.onnx",
    "SAT6_T2W2_DROP1":    "final_optimized_drop1_model_estimate.onnx",
    "SAT6_T2W2_DROP2":    "final_optimized_drop2_model_estimate.onnx",
    "SAT6_T2W2_DROP3":    "final_optimized_drop3_model_estimate.onnx",
    "SAT6_T2W2_DROP4":    "final_optimized_drop4_model_estimate.onnx",
    "SAT6_T2W2_DROP5":    "final_optimized_drop5_model_estimate.onnx",
}


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def gamma_score(fps: float, drop_pct: float, gamma: float):
    """Elastic QoR score.  γ=∞ returns a tuple for lexicographic sort."""
    acc = (100.0 - drop_pct) / 100.0
    if gamma == float("inf"):
        return (-drop_pct, fps)          # tuple: minimise drop, then maximise fps
    return fps * (acc ** gamma)


def utilization_pct(pred: dict, key: str) -> float:
    return pred.get(key, 0) / max(FPGA_LIMITS.get(key, 1), 1) * 100.0


def max_utilization(pred: dict) -> float:
    return max(
        utilization_pct(pred, "Total LUTs"),
        utilization_pct(pred, "FFs"),
        utilization_pct(pred, "BRAM (36k)"),
        utilization_pct(pred, "DSP Blocks"),
    )


# ---------------------------------------------------------------------------
# CANDIDATE SELECTION
# ---------------------------------------------------------------------------

def _parse_cell(s):
    s = (s or "{}").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)


def _row_to_candidate(row: dict, budget: int, gamma: float = 8.0) -> dict:
    fps  = float(row["fps"])
    drop = float(row["acc_drop"])
    balanced = gamma_score(fps, drop, gamma)
    return {
        "model":          row["model"],
        "run_id":         row.get("first_shot_run", "0"),
        "fps":            fps,
        "acc_drop":       drop,
        "budget_pct":     budget,
        "folding_config": _parse_cell(row.get("folding_config")),
        "fifo_depths":    _parse_cell(row.get("fifo_depths")),
        "pred_lut":       int(float(row.get("pred_LUTs", 0))),
        "pred_ff":        int(float(row.get("pred_FFs",  0))),
        "pred_bram":      float(row.get("pred_BRAM", 0)),
        "pred_dsp":       int(float(row.get("pred_DSP",  0))),
        "pred_util_pct":  float(row.get("max_util_pct", 0)),
        "_balanced":      balanced,
        "_is_fallback":   False,
    }


# Margin below budget to qualify as "conservative fallback"
FALLBACK_MARGIN = 5.0


def select_top_candidates(summary_csv: str, top_n: int = 3,
                          filter_budget: int = 0,
                          gamma: float = 8.0) -> list[dict]:
    """
    Return top-N candidates ranked by QoR(gamma) plus one conservative
    fallback per requested budget.

    Conservative fallback = highest QoR among rows with
        max_util_pct < budget - FALLBACK_MARGIN
    (only added if not already in the top-N list).
    """
    if not os.path.exists(summary_csv):
        print(f"[!] Summary not found: {summary_csv}")
        return []

    rows = []
    with open(summary_csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    budgets = [b for b in BUDGET_PCTS if (not filter_budget or b == filter_budget)]
    pool: dict[tuple, dict] = {}   # key=(model, run_id) → best entry

    for budget in budgets:
        for row in rows:
            if int(float(row["budget_pct"])) != budget:
                continue
            drop = float(row["acc_drop"])
            if drop > MAX_ACC_DROP:
                continue
            util = float(row["max_util_pct"])
            if util > budget + ML_TOLERANCE:
                continue

            cand = _row_to_candidate(row, budget, gamma=gamma)
            key  = (cand["model"], cand["run_id"])
            if key not in pool or cand["_balanced"] > pool[key]["_balanced"]:
                pool[key] = cand

    ranked = sorted(pool.values(), key=lambda c: c["_balanced"], reverse=True)
    top    = ranked[:top_n]
    top_keys = {(c["model"], c["run_id"]) for c in top}

    # Conservative fallback: for each budget, best QoR with extra headroom
    for budget in budgets:
        fb_best = None
        for row in rows:
            if int(float(row["budget_pct"])) != budget:
                continue
            drop = float(row["acc_drop"])
            if drop > MAX_ACC_DROP:
                continue
            util = float(row["max_util_pct"])
            if util > budget - FALLBACK_MARGIN:          # tighter threshold
                continue
            cand = _row_to_candidate(row, budget, gamma=gamma)
            key  = (cand["model"], cand["run_id"])
            if key in top_keys:
                continue
            if fb_best is None or cand["_balanced"] > fb_best["_balanced"]:
                fb_best = cand

        if fb_best:
            fb_best["_is_fallback"] = True
            top.append(fb_best)
            top_keys.add((fb_best["model"], fb_best["run_id"]))

    return top


# ---------------------------------------------------------------------------
# HLS BUILD
# ---------------------------------------------------------------------------

def find_onnx(model_label: str) -> str | None:
    fn = ONNX_MAP.get(model_label)
    if not fn:
        # Fuzzy fallback: match drop/prebuilt suffix
        import re
        m = re.search(r"(PREBUILT|DROP\d+)", model_label.upper())
        if m:
            tag = m.group(1)
            fn = next((v for k, v in ONNX_MAP.items() if tag in k), None)
    if not fn:
        return None
    path = os.path.join(SAT6_SEC_DIR, fn)
    return path if os.path.exists(path) else None


def run_hls_build(candidate: dict, build_idx: int) -> tuple[str | None, bool]:
    """
    Invoke run_build.py up to step_set_fifo_depths for one candidate.
    Returns (run_dir, success).
    """
    model    = candidate["model"]
    run_id   = candidate["run_id"]
    tag      = model.replace("SAT6_T2W2_", "").lower()
    hw_name  = f"hls_cand{build_idx:02d}_{tag}_run{run_id}"

    onnx_path = find_onnx(model)
    if not onnx_path:
        print(f"  [!] ONNX not found for '{model}'")
        return None, False

    os.makedirs(BUILD_DIR, exist_ok=True)

    # Write folding config to a temporary JSON that run_build.py can read
    folding_path = os.path.join(BUILD_DIR, f"folding_{hw_name}.json")
    with open(folding_path, "w") as f:
        json.dump(candidate["folding_config"], f, indent=2)

    run_dir  = os.path.join(BUILD_DIR, hw_name)
    log_path = os.path.join(BUILD_DIR, f"{hw_name}.log")

    print(f"\n  [HLS build {build_idx}] {hw_name}")
    print(f"    ONNX    : {os.path.basename(onnx_path)}")
    print(f"    Folding : {folding_path}")
    print(f"    Log     : {log_path}")

    cmd = [
        "python3", os.path.join(hara_dir, "run_build.py"),
        "--model_path",   onnx_path,
        "--build_dir",    BUILD_DIR,
        "--hw_name",      hw_name,
        "--steps",        json.dumps(HLS_DEPTH_STEPS),
        "--fpga-part",    FPGA_PART,
        "--folding_file", folding_path,
    ]

    try:
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=7200)
        success = proc.returncode == 0
        if not success:
            print(f"  [!] Build failed (exit {proc.returncode}) — see {log_path}")
    except subprocess.TimeoutExpired:
        print(f"  [!] Build timed out (2 h) — see {log_path}")
        success = False
    except Exception as exc:
        print(f"  [!] Build exception: {exc}")
        success = False

    return run_dir, success


def read_real_depths(run_dir: str) -> dict | None:
    """Extract FIFO depths from final_hw_config.json written by step_set_fifo_depths."""
    path = os.path.join(run_dir, "final_hw_config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        cfg = json.load(f)
    depths = {
        k.replace("_rtl", ""): v["depth"]
        for k, v in cfg.items()
        if isinstance(v, dict) and "depth" in v
    }
    return depths or None


def read_cycles(run_dir: str) -> dict:
    path = os.path.join(run_dir, "report", "estimate_layer_cycles.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HARA HLS Candidate Refinement")
    parser.add_argument("--top-n",   type=int,   default=3,
                        help="Number of candidates to build (default 3)")
    parser.add_argument("--gamma",   type=float, default=8.0,
                        help="GAMMA for final ranking  [inf=accuracy-first, 8=balanced, 0=max-fps]")
    parser.add_argument("--budget",  type=int,   default=0,
                        help="Restrict to one area budget %% (0 = consider all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Select candidates but do not launch HLS builds")
    parser.add_argument("--summary", type=str,
                        default=os.path.join(OUTPUT_DIR, "predse_summary.csv"),
                        help="Path to predse_summary.csv")
    args = parser.parse_args()

    print("=" * 70)
    print(" HARA — HLS Candidate Refinement Pipeline")
    print(f"   γ={args.gamma}  top-N={args.top_n}  "
          f"budget={'all' if args.budget == 0 else str(args.budget) + '%'}")
    print("=" * 70)

    # [1] Load HARA learner
    print(f"\n[1] Loading MultiModuleLearner from {MODELS_DIR}")
    learner = MultiModuleLearner(MODELS_DIR)
    if not learner.is_loaded():
        print("[!] No models loaded. Exiting.")
        return
    print(f"    ✓ {len(learner._models)} specialist models loaded.")

    # [2] Select candidates
    print(f"\n[2] Selecting top-{args.top_n} candidates from:\n    {args.summary}")
    candidates = select_top_candidates(args.summary, top_n=args.top_n,
                                       filter_budget=args.budget,
                                       gamma=args.gamma)
    if not candidates:
        print("[!] No candidates found. Run run_predse_sweep.py first.")
        return

    gamma_lbl = f"γ={args.gamma:.0f}" if args.gamma != float("inf") else "γ=∞"
    hdr = (f"  {'#':<3} {'Type':<10} {'Model':<28} {'Run':>5} {'FPS':>8} {'Drop':>6} {'Budget':>7}  "
           f"{'LUT(est)':>9} {'Util%':>6} {'BRAM':>6} {'DSP':>5}  {f'QoR({gamma_lbl})':>12}")
    print(f"\n  Candidates (ranked by {gamma_lbl}):")
    print(hdr)
    print("  " + "-" * 110)
    for i, c in enumerate(candidates):
        tag = "FALLBACK" if c.get("_is_fallback") else "primary"
        qor = c["_balanced"]
        print(f"  {i+1:<3} {tag:<10} {c['model']:<28} {c['run_id']:>5} {c['fps']:>8.0f} "
              f"{c['acc_drop']:>5.1f}%  {c['budget_pct']:>6}%  "
              f"{c['pred_lut']:>9} {c['pred_util_pct']:>5.1f}% "
              f"{c['pred_bram']:>6.1f} {c['pred_dsp']:>5}  {qor:>10.1f}")

    if args.dry_run:
        # Per-budget breakdown with all gammas
        print(f"\n  ── Per-budget winners (all gammas) ──")
        gamma_labels = {float('inf'): "γ=∞ (acc-first)", 8.0: "γ=8 (balanced)", 0.0: "γ=0 (max-fps)"}
        all_rows = []
        with open(args.summary) as f:
            all_rows = list(csv.DictReader(f))

        for budget in BUDGET_PCTS:
            print(f"\n  Budget {budget}%:")
            hdr2 = (f"    {'Gamma':<18} {'Model':<28} {'Run':>5} {'FPS':>8} "
                    f"{'Drop':>6}  {'QoR':>10}  {'LUT':>7} {'BRAM':>6} {'DSP':>5}")
            print(hdr2)
            print("    " + "-" * 95)
            for gamma in GAMMAS:
                best = None
                best_score = None
                for row in all_rows:
                    if int(float(row["budget_pct"])) != budget:
                        continue
                    drop = float(row["acc_drop"])
                    if drop > MAX_ACC_DROP:
                        continue
                    if float(row["max_util_pct"]) > budget + ML_TOLERANCE:
                        continue
                    fps   = float(row["fps"])
                    score = gamma_score(fps, drop, gamma)
                    if best_score is None or score > best_score:
                        best_score = score
                        best = row
                if best:
                    fps  = float(best["fps"])
                    drop = float(best["acc_drop"])
                    qor8 = fps * ((100.0 - drop) / 100.0) ** 8.0
                    lbl  = gamma_labels[gamma]
                    print(f"    {lbl:<18} {best['model']:<28} {best['first_shot_run']:>5} "
                          f"{fps:>8.0f} {drop:>5.1f}%  {qor8:>10.1f}  "
                          f"{int(float(best['pred_LUTs'])):>7} "
                          f"{float(best['pred_BRAM']):>6.1f} "
                          f"{int(float(best['pred_DSP'])):>5}")
                else:
                    print(f"    {gamma_labels[gamma]:<18} {'—':>28}")

        print("\n[DRY RUN] Skipping HLS builds.")
        return

    # [3] Run HLS builds
    print(f"\n[3] Running HLS builds (up to step_set_fifo_depths)…")
    print(f"    Build dir: {BUILD_DIR}")
    results = []

    for idx, cand in enumerate(candidates):
        print(f"\n  ── Candidate {idx+1}/{len(candidates)}: "
              f"{cand['model']}  run#{cand['run_id']} ──")

        run_dir, ok = run_hls_build(cand, idx + 1)

        # Defaults: fall back to PreDSE estimates if build fails
        refined = {
            "lut":  cand["pred_lut"],
            "ff":   cand["pred_ff"],
            "bram": cand["pred_bram"],
            "dsp":  cand["pred_dsp"],
        }

        real_depths = None
        if ok and run_dir:
            real_depths = read_real_depths(run_dir)
            cycles      = read_cycles(run_dir)

            if real_depths:
                nnt = sum(1 for d in real_depths.values() if d > 2)
                print(f"  -> Real depths: {len(real_depths)} FIFOs, "
                      f"{nnt} non-trivial, max={max(real_depths.values())}")
            else:
                print("  -> step_set_fifo_depths did not write depths — "
                      "using PreDSE estimates")
                cycles = {}

            # Re-predict with real depths
            onnx_path = find_onnx(cand["model"])
            if onnx_path:
                try:
                    pred_list = learner.predict(
                        onnx_path,
                        [cand["folding_config"]],
                        precomputed_depths=[real_depths] if real_depths else None,
                        cycles_cfg_list=[cycles]         if cycles      else None,
                    )
                    p = pred_list[0] if pred_list else {}
                    refined = {
                        "lut":  p.get("Total LUTs", refined["lut"]),
                        "ff":   p.get("FFs",         refined["ff"]),
                        "bram": p.get("BRAM (36k)",  refined["bram"]),
                        "dsp":  p.get("DSP Blocks",  refined["dsp"]),
                    }
                    print(f"  -> Refined HARA: "
                          f"LUT={refined['lut']}  FF={refined['ff']}  "
                          f"BRAM={refined['bram']:.1f}  DSP={refined['dsp']}")
                except Exception as exc:
                    print(f"  [!] HARA re-prediction failed: {exc}")

        results.append({
            **cand,
            "hls_ok":       ok,
            "run_dir":      run_dir or "",
            "real_depths":  real_depths,
            "refined_lut":  refined["lut"],
            "refined_ff":   refined["ff"],
            "refined_bram": refined["bram"],
            "refined_dsp":  refined["dsp"],
        })

    # [4] Final ranking with fallback chain
    def refined_pred_dict(c):
        return {
            "Total LUTs": c["refined_lut"], "FFs": c["refined_ff"],
            "BRAM (36k)": c["refined_bram"], "DSP Blocks": c["refined_dsp"],
        }

    def is_feasible(c):
        return max_utilization(refined_pred_dict(c)) <= c["budget_pct"] + ML_TOLERANCE

    def refined_score(c):
        if not is_feasible(c):
            return (-1.0,) if args.gamma == float("inf") else -1.0
        return gamma_score(c["fps"], c["acc_drop"], args.gamma)

    # Sort: feasible first (by QoR), then infeasible
    feasible_results   = sorted([r for r in results if is_feasible(r)],
                                 key=lambda c: gamma_score(c["fps"], c["acc_drop"], args.gamma),
                                 reverse=True)
    infeasible_results = [r for r in results if not is_feasible(r)]
    ranked_results = feasible_results + infeasible_results

    print(f"\n[4] Final ranking  (γ={args.gamma})")
    print("=" * 122)
    hdr2 = (f"  {'#':<3} {'Type':<10} {'Model':<28} {'Run':>5} {'FPS':>8} {'Drop':>6}  "
            f"{'LUT(est)':>9} {'LUT(ref)':>9}  "
            f"{'FF(ref)':>8} {'BRAM(ref)':>9} {'DSP(ref)':>8}  "
            f"{'MaxUtil':>8}  {'HLS':>5}")
    print(hdr2)
    print("  " + "-" * 119)

    winner = None
    for i, c in enumerate(ranked_results):
        pred  = refined_pred_dict(c)
        util  = max_utilization(pred)
        feas  = util <= c["budget_pct"] + ML_TOLERANCE
        tag   = "FALLBACK" if c.get("_is_fallback") else "primary"
        label = ""
        if feas and winner is None:
            winner = c
            label  = " ← WINNER"

        print(f"  {i+1:<3} {tag:<10} {c['model']:<28} {c['run_id']:>5} {c['fps']:>8.0f} "
              f"{c['acc_drop']:>5.1f}%  "
              f"{c['pred_lut']:>9} {c['refined_lut']:>9}  "
              f"{c['refined_ff']:>8} {c['refined_bram']:>9.1f} {c['refined_dsp']:>8}  "
              f"{util:>7.1f}%  "
              f"{'OK' if c.get('hls_ok') else 'FAIL':>5}"
              f"{label}")

    # Fallback chain decision log
    primary_feasible  = [r for r in feasible_results if not r.get("_is_fallback")]
    fallback_feasible = [r for r in feasible_results if r.get("_is_fallback")]

    if primary_feasible:
        print(f"\n  Decision: primary candidate fits budget → using it directly.")
    elif fallback_feasible:
        print(f"\n  Decision: primary over-budget; conservative fallback fits → using fallback.")
    elif not winner:
        # ── Reverse greedy: scan predse_summary from best QoR downward ──
        print(f"\n  Decision: all candidates over-budget → triggering reverse greedy search…")
        budget_target = args.budget if args.budget else (
            ranked_results[0]["budget_pct"] if ranked_results else BUDGET_PCTS[0])
        summary_csv = args.summary

        rg_rows = []
        with open(summary_csv) as f:
            for row in csv.DictReader(f):
                if int(float(row["budget_pct"])) != budget_target:
                    continue
                drop = float(row["acc_drop"])
                if drop > MAX_ACC_DROP:
                    continue
                fps  = float(row["fps"])
                util = float(row["max_util_pct"])
                # Only candidates predicted safely within budget (no ML tolerance here)
                if util > budget_target:
                    continue
                balanced = gamma_score(fps, drop, args.gamma)
                rg_rows.append((balanced, row))

        rg_rows.sort(key=lambda x: x[0], reverse=True)

        already_built = {(r["model"], r["run_id"]) for r in results}
        rg_winner = None
        for _, row in rg_rows:
            key = (row["model"], row.get("first_shot_run", "0"))
            if key in already_built:
                continue
            rg_cand = _row_to_candidate(row, budget_target, gamma=args.gamma)
            rg_cand["_is_fallback"] = True
            print(f"\n  [Reverse greedy] Building: {rg_cand['model']}  "
                  f"run#{rg_cand['run_id']}  util={rg_cand['pred_util_pct']:.1f}%  "
                  f"QoR={rg_cand['_balanced']:.1f}")

            rg_dir, rg_ok = run_hls_build(rg_cand, len(results) + 1)
            if not rg_ok:
                print("  [!] Reverse greedy build failed — trying next.")
                continue

            rg_depths = read_real_depths(rg_dir) if rg_dir else None
            rg_onnx   = find_onnx(rg_cand["model"])
            rg_refined = {"lut": rg_cand["pred_lut"], "ff": rg_cand["pred_ff"],
                          "bram": rg_cand["pred_bram"], "dsp": rg_cand["pred_dsp"]}
            if rg_onnx and rg_dir:
                try:
                    p = learner.predict(rg_onnx, [rg_cand["folding_config"]],
                                        precomputed_depths=[rg_depths] if rg_depths else None)[0]
                    rg_refined = {"lut": p.get("Total LUTs", rg_refined["lut"]),
                                  "ff":  p.get("FFs", rg_refined["ff"]),
                                  "bram": p.get("BRAM (36k)", rg_refined["bram"]),
                                  "dsp":  p.get("DSP Blocks", rg_refined["dsp"])}
                except Exception as exc:
                    print(f"  [!] HARA re-prediction failed: {exc}")

            rg_cand.update({"hls_ok": rg_ok, "run_dir": rg_dir or "",
                             "real_depths": rg_depths,
                             "refined_lut": rg_refined["lut"],
                             "refined_ff":  rg_refined["ff"],
                             "refined_bram": rg_refined["bram"],
                             "refined_dsp":  rg_refined["dsp"]})
            results.append(rg_cand)

            rg_util = max_utilization({"Total LUTs": rg_refined["lut"],
                                       "FFs": rg_refined["ff"],
                                       "BRAM (36k)": rg_refined["bram"],
                                       "DSP Blocks": rg_refined["dsp"]})
            if rg_util <= budget_target + ML_TOLERANCE:
                rg_winner = rg_cand
                winner    = rg_cand
                print(f"  [Reverse greedy] ✓ Found feasible candidate! util={rg_util:.1f}%")
                break
            else:
                print(f"  [Reverse greedy] Still over budget ({rg_util:.1f}%) — continuing.")

        if not rg_winner:
            print("  [Reverse greedy] Exhausted all options. No feasible candidate found.")

    # Winner summary
    if winner:
        pred = refined_pred_dict(winner)
        util = max_utilization(pred)
        tag  = "FALLBACK" if winner.get("_is_fallback") else "primary"
        print(f"\n  ✓ WINNER ({tag}): {winner['model']}  run#{winner['run_id']}")
        print(f"    Build dir : {winner.get('run_dir', 'N/A')}")
        print(f"    FPS       : {winner['fps']:.0f}")
        print(f"    Acc drop  : {winner['acc_drop']:.1f}%")
        print(f"    Resources : LUT={pred['Total LUTs']}  FF={pred['FFs']}  "
              f"BRAM={pred['BRAM (36k)']:.1f}  DSP={pred['DSP Blocks']}")
        print(f"    Max util  : {util:.1f}%  (budget={winner['budget_pct']}%)")
        print(f"\n  Next steps:")
        print(f"    1. Continue FINN build from {winner.get('run_dir', '<run_dir>')}")
        print(f"       (step_create_stitched_ip → step_out_of_context_synthesis)")
        print(f"    2. Or call run_build.py with remaining steps + --build_dir {BUILD_DIR}")

        winner_path = os.path.join(OUTPUT_DIR, "hls_winner.json")
        save = {k: v for k, v in winner.items()
                if k not in ("real_depths", "_balanced")}
        with open(winner_path, "w") as f:
            json.dump(save, f, indent=2, default=str)
        print(f"\n  Winner info → {winner_path}")
    else:
        print("\n  [!] No feasible winner found. Consider relaxing the area budget.")

    # Save full results CSV
    csv_keys = [
        "model", "run_id", "fps", "acc_drop", "budget_pct",
        "pred_lut", "pred_ff", "pred_bram", "pred_dsp",
        "refined_lut", "refined_ff", "refined_bram", "refined_dsp",
        "hls_ok", "run_dir",
    ]
    results_csv = os.path.join(OUTPUT_DIR, "hls_candidates_results.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  Results CSV → {results_csv}")


if __name__ == "__main__":
    main()
