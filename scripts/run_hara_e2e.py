#!/usr/bin/env python3
"""
run_hara_e2e.py — HARA End-to-End Pipeline

Orchestrates the complete HARA flow for a given budget:

  Stage 1 — HLS Candidate Refinement  (run_hls_candidates.py logic)
    1a. Select top-N candidates + conservative fallback via GAMMA scoring
    1b. Run FINN up to step_set_fifo_depths for each candidate
    1c. Re-predict area with real FIFO depths
    1d. Pick winner (fallback chain → reverse greedy if needed)

  Stage 2 — Full Synthesis  (run_full_build.py logic)
    2a. Run step_create_stitched_ip + step_out_of_context_synthesis on winner

Usage:
    python3 scripts/run_hara_e2e.py --budget 10
    python3 scripts/run_hara_e2e.py --budget 30 --top-n 5 --gamma 8.0
    python3 scripts/run_hara_e2e.py --budget 10 --stage1-only   # stop after picking winner
    python3 scripts/run_hara_e2e.py --budget 10 --stage2-only   # synthesise existing winner
    python3 scripts/run_hara_e2e.py --budget 10 --dry-run       # plan only, no builds
"""

import argparse
import json
import os
import sys
import time

hara_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, hara_dir)

# Reuse all helpers from the two sub-scripts
from scripts.run_hls_candidates import (
    select_top_candidates,
    run_hls_build,
    read_real_depths,
    read_cycles,
    find_onnx,
    gamma_score,
    max_utilization,
    BUDGET_PCTS,
    FPGA_PART,
    FPGA_LIMITS,
    ML_TOLERANCE,
    MAX_ACC_DROP,
    OUTPUT_DIR,
    BUILD_DIR,
    MODELS_DIR,
    _row_to_candidate,
    _parse_cell,
)
from scripts.run_full_build import (
    run_synthesis,
    parse_partition_util_rpt,
    report_budget_check,
    load_winner,
)
from ai.multi_module_learner import MultiModuleLearner

import csv
import scripts.run_hls_candidates as hls_cands # <-- Adicione esta linha!
SAT6_SEC_DIR = os.path.join(hara_dir, "models", "SAT6_SEC")
WINNER_PATH  = os.path.join(SAT6_SEC_DIR, "hls_winner.json")


# ---------------------------------------------------------------------------
# Stage 1: HLS candidate selection + refinement
# ---------------------------------------------------------------------------

def stage1(learner: MultiModuleLearner, summary_csv: str,
           top_n: int, budget: int, gamma_final: float,
           dry_run: bool) -> dict | None:

    print(f"\n{'='*70}")
    print(" Stage 1 — HLS Candidate Refinement")
    print(f"{'='*70}")

    candidates = select_top_candidates(summary_csv, top_n=top_n,
                                       filter_budget=budget, gamma=gamma_final)
    if not candidates:
        print("[!] No candidates found. Run run_predse_sweep.py first.")
        return None

    gamma_lbl = f"γ={gamma_final:.0f}" if gamma_final != float("inf") else "γ=∞"
    hdr = (f"  {'#':<3} {'Type':<10} {'Model':<28} {'Run':>5} {'FPS':>8} "
           f"{'Drop':>6} {'Util%':>6}  {f'QoR({gamma_lbl})':>12}")
    print(f"\n  Candidates (ranked by {gamma_lbl}):")
    print(hdr)
    print("  " + "-" * 80)
    for i, c in enumerate(candidates):
        tag = "FALLBACK" if c.get("_is_fallback") else "primary"
        print(f"  {i+1:<3} {tag:<10} {c['model']:<28} {c['run_id']:>5} "
              f"{c['fps']:>8.0f} {c['acc_drop']:>5.1f}%  "
              f"{c['pred_util_pct']:>5.1f}%  {c['_balanced']:>10.1f}")

    if dry_run:
        print("\n  [DRY RUN] Skipping HLS builds.")
        return None

    # Build each candidate
    results = []
    for idx, cand in enumerate(candidates):
        print(f"\n  ── Candidate {idx+1}/{len(candidates)}: "
              f"{cand['model']}  run#{cand['run_id']} ──")

        run_dir, ok = run_hls_build(cand, idx + 1)

        refined = {"lut": cand["pred_lut"], "ff": cand["pred_ff"],
                   "bram": cand["pred_bram"], "dsp": cand["pred_dsp"]}

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

            onnx_path = find_onnx(cand["model"])
            if onnx_path:
                try:
                    p = learner.predict(
                        onnx_path, [cand["folding_config"]],
                        precomputed_depths=[real_depths] if real_depths else None,
                        cycles_cfg_list=[cycles]         if cycles      else None,
                    )[0]
                    refined = {"lut": p.get("Total LUTs", refined["lut"]),
                               "ff":  p.get("FFs",        refined["ff"]),
                               "bram": p.get("BRAM (36k)", refined["bram"]),
                               "dsp":  p.get("DSP Blocks", refined["dsp"])}
                    print(f"  -> Refined HARA: "
                          f"LUT={refined['lut']}  FF={refined['ff']}  "
                          f"BRAM={refined['bram']:.1f}  DSP={refined['dsp']}")
                except Exception as exc:
                    print(f"  [!] HARA re-prediction failed: {exc}")

        results.append({**cand, "hls_ok": ok, "run_dir": run_dir or "",
                        "real_depths": real_depths,
                        "refined_lut": refined["lut"], "refined_ff": refined["ff"],
                        "refined_bram": refined["bram"], "refined_dsp": refined["dsp"]})

    # Winner selection with fallback chain
    winner = _pick_winner(results, learner, summary_csv, budget, gamma_final)
    return winner


def _refined_pred_dict(c):
    return {"Total LUTs": c["refined_lut"], "FFs": c["refined_ff"],
            "BRAM (36k)": c["refined_bram"], "DSP Blocks": c["refined_dsp"]}


def _is_feasible(c):
    return max_utilization(_refined_pred_dict(c)) <= c["budget_pct"] + ML_TOLERANCE


def _pick_winner(results: list, learner: MultiModuleLearner,
                 summary_csv: str, budget: int, gamma_final: float) -> dict | None:

    feasible   = sorted([r for r in results if _is_feasible(r)],
                        key=lambda c: gamma_score(c["fps"], c["acc_drop"], gamma_final),
                        reverse=True)
    infeasible = [r for r in results if not _is_feasible(r)]

    print(f"\n  Ranking after refinement  (γ={gamma_final}):")
    hdr = (f"  {'#':<3} {'Type':<10} {'Model':<28} {'Run':>5} "
           f"{'FPS':>8} {'Drop':>6}  {'LUT(ref)':>9} {'MaxUtil':>8}  Status")
    print(hdr)
    print("  " + "-" * 95)

    winner = None
    for i, c in enumerate(feasible + infeasible):
        util  = max_utilization(_refined_pred_dict(c))
        feas  = _is_feasible(c)
        tag   = "FALLBACK" if c.get("_is_fallback") else "primary"
        label = " ← WINNER" if (feas and winner is None) else ""
        if feas and winner is None:
            winner = c
        status = "OK" if c.get("hls_ok") else "FAIL"
        print(f"  {i+1:<3} {tag:<10} {c['model']:<28} {c['run_id']:>5} "
              f"{c['fps']:>8.0f} {c['acc_drop']:>5.1f}%  "
              f"{c['refined_lut']:>9}  {util:>7.1f}%  {status}{label}")

    primary_ok  = any(not r.get("_is_fallback") and _is_feasible(r) for r in results)
    fallback_ok = any(r.get("_is_fallback")     and _is_feasible(r) for r in results)

    if primary_ok:
        print(f"\n  Decision: primary candidate fits → winner selected.")
    elif fallback_ok:
        print(f"\n  Decision: primary over-budget; conservative fallback fits → using fallback.")
    elif winner is None:
        print(f"\n  Decision: all candidates over-budget → reverse greedy search…")
        winner = _reverse_greedy(results, learner, summary_csv, budget, gamma_final)

    return winner


def _reverse_greedy(existing: list, learner: MultiModuleLearner,
                    summary_csv: str, budget: int, gamma_final: float) -> dict | None:
    already_built = {(r["model"], r["run_id"]) for r in existing}

    rg_rows = []
    with open(summary_csv) as f:
        for row in csv.DictReader(f):
            if int(float(row["budget_pct"])) != budget:
                continue
            if float(row["acc_drop"]) > MAX_ACC_DROP:
                continue
            # Only rows predicted safely within budget (no ML tolerance)
            if float(row["max_util_pct"]) > budget:
                continue
            cand = _row_to_candidate(row, budget)
            key  = (cand["model"], cand["run_id"])
            if key in already_built:
                continue
            rg_rows.append((cand["_balanced"], cand))

    rg_rows.sort(key=lambda x: x[0], reverse=True)
    build_idx = len(existing) + 1

    for _, cand in rg_rows:
        cand["_is_fallback"] = True
        print(f"\n  [Reverse greedy] Trying: {cand['model']}  run#{cand['run_id']}  "
              f"util={cand['pred_util_pct']:.1f}%  QoR={cand['_balanced']:.1f}")

        run_dir, ok = run_hls_build(cand, build_idx)
        build_idx += 1
        if not ok:
            print("  [!] Build failed — next.")
            continue

        real_depths = read_real_depths(run_dir) if run_dir else None
        cycles      = read_cycles(run_dir)      if run_dir else {}
        refined     = {"lut": cand["pred_lut"], "ff": cand["pred_ff"],
                       "bram": cand["pred_bram"], "dsp": cand["pred_dsp"]}

        onnx_path = find_onnx(cand["model"])
        if onnx_path:
            try:
                p = learner.predict(onnx_path, [cand["folding_config"]],
                                    precomputed_depths=[real_depths] if real_depths else None,
                                    cycles_cfg_list=[cycles]         if cycles      else None)[0]
                refined = {"lut": p.get("Total LUTs", refined["lut"]),
                           "ff":  p.get("FFs",        refined["ff"]),
                           "bram": p.get("BRAM (36k)", refined["bram"]),
                           "dsp":  p.get("DSP Blocks", refined["dsp"])}
            except Exception as exc:
                print(f"  [!] HARA re-prediction failed: {exc}")

        cand.update({"hls_ok": ok, "run_dir": run_dir or "",
                     "real_depths": real_depths,
                     "refined_lut": refined["lut"], "refined_ff": refined["ff"],
                     "refined_bram": refined["bram"], "refined_dsp": refined["dsp"]})

        util = max_utilization(_refined_pred_dict(cand))
        if util <= budget + ML_TOLERANCE:
            print(f"  [Reverse greedy] ✓ Feasible! util={util:.1f}%")
            return cand
        print(f"  [Reverse greedy] Still over-budget ({util:.1f}%) — continuing.")

    print("  [Reverse greedy] No feasible candidate found.")
    return None


# ---------------------------------------------------------------------------
# Stage 2: Full synthesis
# ---------------------------------------------------------------------------

def stage2(winner: dict, dry_run: bool) -> bool:
    print(f"\n{'='*70}")
    print(" Stage 2 — Full Synthesis (stitched IP + out-of-context)")
    print(f"{'='*70}")
    return run_synthesis(winner, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HARA End-to-End Pipeline")
    parser.add_argument("--budget",      type=int,   required=True,
                        choices=BUDGET_PCTS, help="Area budget %%")
    parser.add_argument("--top-n",       type=int,   default=3,
                        help="HLS candidates to evaluate (default 3)")
    parser.add_argument("--gamma",       type=float, default=8.0,
                        help="GAMMA for final winner selection (default 8.0)")
    parser.add_argument("--summary",     type=str,
                        default=os.path.join(SAT6_SEC_DIR, "predse_summary.csv"),
                        help="Path to predse_summary.csv")
    parser.add_argument("--winner-out",  type=str, default=WINNER_PATH,
                        help="Where to save hls_winner.json")
    parser.add_argument("--stage1-only", action="store_true",
                        help="Run Stage 1 only (no synthesis)")
    parser.add_argument("--stage2-only", action="store_true",
                        help="Run Stage 2 only (load existing winner)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print plan without executing any builds")
    args = parser.parse_args()

    t_start = time.time()

    # --- NOVO: Cria uma pasta específica para esta run! ---
    run_folder_name = f"e2e_budget{args.budget}_gamma{args.gamma}"
    new_build_dir = os.path.join(hara_dir, "hls_candidate_builds", run_folder_name)
    os.makedirs(new_build_dir, exist_ok=True)
    # Redireciona o script de candidatos para usar essa nova pasta
    hls_cands.BUILD_DIR = new_build_dir
    # ------------------------------------------------------

    print("=" * 70)
    print(" HARA — End-to-End Pipeline")
    print(f"   budget={args.budget}%  top-N={args.top_n}  γ={args.gamma}")
    print("=" * 70)

    winner = None

    if args.stage2_only:
        winner = load_winner(args.winner_out)
        print(f"\n  Loaded winner: {winner.get('model')}  run#{winner.get('run_id')}")
    else:
        # Load learner once for both stages
        print(f"\n[0] Loading MultiModuleLearner…")
        learner = MultiModuleLearner(MODELS_DIR)
        if not learner.is_loaded():
            print("[!] No models loaded. Exiting.")
            return

        winner = stage1(learner, args.summary,
                        top_n=args.top_n, budget=args.budget,
                        gamma_final=args.gamma, dry_run=args.dry_run)

        if winner and not args.dry_run:
            # Persist winner
            save = {k: v for k, v in winner.items()
                    if k not in ("real_depths", "_balanced")}
            with open(args.winner_out, "w") as f:
                json.dump(save, f, indent=2, default=str)
            print(f"\n  Winner saved → {args.winner_out}")

    if args.stage1_only or args.dry_run:
        if winner:
            print(f"\n  Winner: {winner.get('model')}  run#{winner.get('run_id')}  "
                  f"FPS={winner.get('fps', 0):.0f}  drop={winner.get('acc_drop', 0):.1f}%")
        elapsed = (time.time() - t_start) / 60.0
        print(f"\n  Total time: {elapsed:.1f} min")
        return

    if not winner:
        print("\n[!] No winner from Stage 1. Aborting.")
        return

    ok = stage2(winner, dry_run=args.dry_run)

    elapsed = (time.time() - t_start) / 3600.0
    print(f"\n{'='*70}")
    if ok:
        print(f" Pipeline complete in {elapsed:.2f}h")
        print(f" Winner  : {winner['model']}  run#{winner['run_id']}")
        
        # Deduz o nome da pasta de síntese baseada na pasta HLS
        hls_run_dir = winner.get("run_dir", "")
        hls_hw_name = os.path.basename(hls_run_dir)
        synth_hw_name = hls_hw_name.replace("hls_cand", "synth_cand", 1)
        synth_run_dir = os.path.join(os.path.dirname(hls_run_dir), synth_hw_name)
        
        print(f" Build   : {synth_run_dir}")
        
        synth = parse_partition_util_rpt(synth_run_dir)
        if synth:
            print(f"\n Post-synthesis utilisation:")
            fits = report_budget_check(synth, args.budget)
            
            pred_lut = winner.get("refined_lut", winner.get("pred_lut", 0))
            err_lut  = abs(synth["Total LUTs"] - pred_lut) / max(synth["Total LUTs"], 1) * 100
            print(f"\n  HARA pred LUTs : {pred_lut}  →  Real: {synth['Total LUTs']}  "
                  f"(err {err_lut:.1f}%)")
            print(f"\n  {'[✓] FITS BUDGET' if fits else '[!] OVER BUDGET'} ({args.budget}%)")
        else:
            print(f"  [!] finn_design_partition_util.rpt not found in {synth_run_dir}")
    else:
        print(f" Pipeline finished with errors after {elapsed:.2f}h")
    print("=" * 70)


if __name__ == "__main__":
    main()
