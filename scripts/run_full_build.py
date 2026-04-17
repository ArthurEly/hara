#!/usr/bin/env python3
"""
run_full_build.py — HARA Full Synthesis Build

Reads hls_winner.json (produced by run_hls_candidates.py) and runs the
remaining FINN steps on the winning candidate's already-built directory:

    step_create_stitched_ip → step_out_of_context_synthesis

The HLS artifacts (step_hw_ipgen, step_set_fifo_depths) produced during
candidate selection are reused — no re-compilation needed.

Usage:
    python3 scripts/run_full_build.py
    python3 scripts/run_full_build.py --winner path/to/hls_winner.json
    python3 scripts/run_full_build.py --winner path/to/hls_winner.json --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time

hara_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, hara_dir)

# ---------------------------------------------------------------------------
# Steps that come AFTER step_set_fifo_depths
# ---------------------------------------------------------------------------
SYNTHESIS_STEPS = [
    "step_create_stitched_ip",
    "step_out_of_context_synthesis",
    # "step_synthesize_bitfile",
    # "step_make_pynq_driver",
    # "step_deployment_package",
]

FPGA_PART    = "xc7z020clg400-1"
SAT6_SEC_DIR = os.path.join(hara_dir, "models", "SAT6_SEC")

# ONNX map (same as run_hls_candidates.py)
ONNX_MAP = {
    "SAT6_T2W2_PREBUILT": "sat6_t2_baseline_estimate.onnx",
    "SAT6_T2W2_DROP1":    "final_optimized_drop1_model_estimate.onnx",
    "SAT6_T2W2_DROP2":    "final_optimized_drop2_model_estimate.onnx",
    "SAT6_T2W2_DROP3":    "final_optimized_drop3_model_estimate.onnx",
    "SAT6_T2W2_DROP4":    "final_optimized_drop4_model_estimate.onnx",
    "SAT6_T2W2_DROP5":    "final_optimized_drop5_model_estimate.onnx",
}


def find_onnx(model_label: str) -> str | None:
    fn = ONNX_MAP.get(model_label)
    if not fn:
        import re
        m = re.search(r"(PREBUILT|DROP\d+)", model_label.upper())
        if m:
            fn = next((v for k, v in ONNX_MAP.items() if m.group(1) in k), None)
    if not fn:
        return None
    path = os.path.join(SAT6_SEC_DIR, fn)
    return path if os.path.exists(path) else None


def load_winner(winner_path: str) -> dict:
    if not os.path.exists(winner_path):
        print(f"[!] Winner file not found: {winner_path}")
        sys.exit(1)
    with open(winner_path) as f:
        return json.load(f)


def check_hls_artifacts(run_dir: str) -> bool:
    """Verify that step_set_fifo_depths completed successfully."""
    required = [
        os.path.join(run_dir, "final_hw_config.json"),
        os.path.join(run_dir, "intermediate_models", "step_set_fifo_depths.onnx"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"  [!] Missing HLS artifact: {p}")
        return False
    return True


def run_synthesis(winner: dict, dry_run: bool = False) -> bool:
    hls_run_dir = winner.get("run_dir", "")
    model       = winner.get("model", "")
    run_id      = winner.get("run_id", "0")
    hls_hw_name = os.path.basename(hls_run_dir)
    build_dir   = os.path.dirname(hls_run_dir)

    # Synthesis gets its own directory (synth_*) to avoid overwriting HLS artifacts
    synth_hw_name = hls_hw_name.replace("hls_cand", "synth_cand", 1)
    synth_run_dir = os.path.join(build_dir, synth_hw_name)

    print(f"\n  Model      : {model}  run#{run_id}")
    print(f"  HLS dir    : {hls_run_dir}")
    print(f"  Synth dir  : {synth_run_dir}")
    print(f"  Steps      : {' → '.join(SYNTHESIS_STEPS)}")

    if not hls_run_dir or not os.path.isdir(hls_run_dir):
        print(f"  [!] HLS run_dir does not exist: {hls_run_dir}")
        return False

    if not check_hls_artifacts(hls_run_dir):
        print("  [!] HLS artifacts incomplete — re-run run_hls_candidates.py first.")
        return False

    # Use step_set_fifo_depths.onnx checkpoint from the HLS dir as model input
    onnx_path = os.path.join(hls_run_dir, "intermediate_models", "step_set_fifo_depths.onnx")
    print(f"  Checkpoint : {onnx_path}")

    # Folding file from the HLS build (same config, different dir)
    folding_path = os.path.join(build_dir, f"folding_{hls_hw_name}.json")
    if not os.path.exists(folding_path):
        folding_cfg = winner.get("folding_config", {})
        if isinstance(folding_cfg, str):
            import ast
            folding_cfg = ast.literal_eval(folding_cfg)
        with open(folding_path, "w") as f:
            json.dump(folding_cfg, f, indent=2)
        print(f"  [i] Folding config reconstructed → {folding_path}")

    log_path = os.path.join(build_dir, f"{synth_hw_name}.log")
    print(f"  Log        : {log_path}")

    cmd = [
        "python3", os.path.join(hara_dir, "run_build.py"),
        "--model_path",   onnx_path,
        "--build_dir",    build_dir,
        "--hw_name",      synth_hw_name,
        "--steps",        json.dumps(SYNTHESIS_STEPS),
        "--fpga-part",    FPGA_PART,
        "--folding_file", folding_path,
    ]

    if dry_run:
        print(f"\n  [DRY RUN] Would execute:\n    {' '.join(cmd)}")
        return True

    t0 = time.time()
    log_path = os.path.join(build_dir, f"{synth_hw_name}.log")
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=14400)
        elapsed = (time.time() - t0) / 3600.0
        ok = proc.returncode == 0
        if ok:
            print(f"  ✓ Synthesis complete in {elapsed:.2f}h")
        else:
            print(f"  [!] Synthesis failed (exit {proc.returncode}) after {elapsed:.2f}h — see {log_path}")
        return ok
    except subprocess.TimeoutExpired:
        print(f"  [!] Synthesis timed out (4 h) — see {log_path}")
        return False
    except Exception as exc:
        print(f"  [!] Exception: {exc}")
        return False


FPGA_LIMITS = {"Total LUTs": 53200, "FFs": 106400, "BRAM (36k)": 140, "DSP Blocks": 220}


def parse_partition_util_rpt(synth_run_dir: str) -> dict | None:
    """
    Parse finn_design_partition_util.rpt — the authoritative GT utilization.
    Reads the finn_design_i row (Total LUTs, Logic LUTs, LUTRAMs, SRLs, FFs, RAMB36, RAMB18, DSP).
    BRAM (36k) equivalent = RAMB36 + RAMB18/2.
    """
    rpt = os.path.join(synth_run_dir, "stitched_ip", "finn_design_partition_util.rpt")
    if not os.path.exists(rpt):
        return None

    with open(rpt) as f:
        for line in f:
            if "finn_design_i" not in line or "finn_design |" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            # Expected columns: Instance, Module, Total LUTs, Logic LUTs, LUTRAMs, SRLs, FFs, RAMB36, RAMB18, DSP Blocks
            try:
                total_luts  = int(parts[3])
                logic_luts  = int(parts[4])
                lutrams     = int(parts[5])
                srls        = int(parts[6])
                ffs         = int(parts[7])
                ramb36      = int(parts[8])
                ramb18      = int(parts[9])
                dsp         = int(parts[10])
                bram_36k_eq = ramb36 + ramb18 / 2.0
                return {
                    "Total LUTs": total_luts,
                    "Logic LUTs": logic_luts,
                    "LUTRAMs":    lutrams,
                    "SRLs":       srls,
                    "FFs":        ffs,
                    "BRAM (36k)": bram_36k_eq,
                    "RAMB36":     ramb36,
                    "RAMB18":     ramb18,
                    "DSP Blocks": dsp,
                }
            except (IndexError, ValueError):
                continue
    return None


def report_budget_check(synth: dict, budget_pct: int) -> bool:
    """Print budget check table and return True if design fits."""
    limits = FPGA_LIMITS
    print(f"\n  {'Resource':<12} {'Real':>7} {'Budget':>7} {'FPGA':>7} {'Util%':>7}  Fit?")
    print("  " + "-" * 50)
    fits = True
    for key in ["Total LUTs", "FFs", "BRAM (36k)", "DSP Blocks"]:
        real    = synth.get(key, 0)
        total   = limits[key]
        budget  = total * budget_pct / 100.0
        pct     = real / total * 100.0
        ok      = real <= budget
        fits    = fits and ok
        mark    = "✓" if ok else "✗ OVER"
        print(f"  {key:<12} {real:>7.1f} {budget:>7.0f} {total:>7} {pct:>6.1f}%  {mark}")
    return fits


def main():
    parser = argparse.ArgumentParser(description="HARA Full Synthesis Build")
    parser.add_argument("--winner", type=str,
                        default=os.path.join(SAT6_SEC_DIR, "hls_winner.json"),
                        help="Path to hls_winner.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print command without executing")
    args = parser.parse_args()

    print("=" * 70)
    print(" HARA — Full Synthesis Build")
    print("=" * 70)

    winner = load_winner(args.winner)

    print(f"\n  Winner loaded from: {args.winner}")
    print(f"  Model    : {winner.get('model')}  run#{winner.get('run_id')}")
    print(f"  FPS      : {winner.get('fps', '?'):.0f}" if isinstance(winner.get('fps'), float)
          else f"  FPS      : {winner.get('fps', '?')}")
    print(f"  Acc drop : {winner.get('acc_drop', 0):.1f}%")
    r_lut  = winner.get("refined_lut",  winner.get("pred_lut",  "?"))
    r_bram = winner.get("refined_bram", winner.get("pred_bram", "?"))
    r_dsp  = winner.get("refined_dsp",  winner.get("pred_dsp",  "?"))
    print(f"  Est. res.: LUT={r_lut}  BRAM={r_bram}  DSP={r_dsp}")

    hls_run_dir   = winner.get("run_dir", "")
    hls_hw_name   = os.path.basename(hls_run_dir)
    synth_hw_name = hls_hw_name.replace("hls_cand", "synth_cand", 1)
    synth_run_dir = os.path.join(os.path.dirname(hls_run_dir), synth_hw_name)
    budget_pct    = int(winner.get("budget_pct", 10))

    # If synth dir already exists with rpt, skip build and just report
    existing_synth = parse_partition_util_rpt(synth_run_dir)
    if existing_synth and not args.dry_run:
        print(f"\n  [i] Synthesis already complete — reading existing report.")
        ok = True
    else:
        ok = run_synthesis(winner, dry_run=args.dry_run)

    if ok and not args.dry_run:
        synth = parse_partition_util_rpt(synth_run_dir)
        if synth:
            fits = report_budget_check(synth, budget_pct)
            pred_lut = winner.get("refined_lut", winner.get("pred_lut", 0))
            err_lut  = abs(synth["Total LUTs"] - pred_lut) / max(synth["Total LUTs"], 1) * 100
            print(f"\n  HARA pred LUTs : {pred_lut}  →  Real: {synth['Total LUTs']}  "
                  f"(err {err_lut:.1f}%)")
            print(f"\n  {'FITS BUDGET' if fits else 'OVER BUDGET'} ({budget_pct}%)")
        else:
            print(f"\n  [!] finn_design_partition_util.rpt not found in {synth_run_dir}")

        print(f"\n  Build artefacts: {synth_run_dir}")


if __name__ == "__main__":
    main()
