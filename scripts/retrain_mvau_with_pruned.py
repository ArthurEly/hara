#!/usr/bin/env python3
"""
retrain_mvau_with_pruned.py — Retrain MVAU_SAT6_T2W2 with pruned-model anchor builds

Flow:
  1. For each DROP model (0–5) that doesn't yet have a synthesis build,
     run FINN up to step_out_of_context_synthesis with PE=SIMD=1 folding.
  2. Parse per-module GT from finn_design_partition_util.rpt using
     get_exhaustive_area_results logic.
  3. Append new rows to the MVAU training CSV with high sample_weight.
  4. Retrain MVAU_SAT6_T2W2 XGBoost specialist and save updated .pkl.

Why PE=SIMD=1?
  The PreDSE sweep at tight budgets (10–20%) always lands on PE=SIMD=1
  for the SAT6 model family. A single build per DROP level is sufficient
  to anchor the model at the correct (MH, MW) operating point.

Usage:
    python3 scripts/retrain_mvau_with_pruned.py
    python3 scripts/retrain_mvau_with_pruned.py --skip-builds   # retrain only
    python3 scripts/retrain_mvau_with_pruned.py --dry-run
"""

import argparse
import ast
import json
import os
import pickle
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

hara_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, hara_dir)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAT6_SEC_DIR   = os.path.join(hara_dir, "models", "SAT6_SEC")
BUILD_DIR      = os.path.join(hara_dir, "hls_candidate_builds", "pruned_anchors")
RETRIEVAL_DIR  = os.path.join(hara_dir, "ai", "retrieval")
SPLITTED_DIR   = os.path.join(RETRIEVAL_DIR, "results", "splitted")
MODELS_DIR     = os.path.join(RETRIEVAL_DIR, "results", "trained_models")
MVAU_CSV       = os.path.join(SPLITTED_DIR, "exhaustive_MVAU_area_attrs.csv")
MVAU_MODEL_PKL = os.path.join(MODELS_DIR, "MVAU_SAT6_T2W2.pkl")
MVAU_CSV_AUG   = os.path.join(SPLITTED_DIR, "exhaustive_MVAU_area_attrs_augmented.csv")

FPGA_PART = "xc7z020clg400-1"

ONNX_MAP = {
    "DROP0": "sat6_t2_baseline_estimate.onnx",
    "DROP1": "final_optimized_drop1_model_estimate.onnx",
    "DROP2": "final_optimized_drop2_model_estimate.onnx",
    "DROP3": "final_optimized_drop3_model_estimate.onnx",
    "DROP4": "final_optimized_drop4_model_estimate.onnx",
    "DROP5": "final_optimized_drop5_model_estimate.onnx",
}

# Sample weight multiplier for new pruned-model anchor points
ANCHOR_WEIGHT = 10.0

# Full build steps needed to get partition_util.rpt
FULL_STEPS = [
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
    "step_create_stitched_ip",
    "step_out_of_context_synthesis",
]

TARGET_COLS = ["Total LUT", "Total FFs", "BRAM (36k eq.)", "DSP Blocks"]

# ---------------------------------------------------------------------------
# PE=SIMD=1 folding — minimal parallelism, valid for all DROP topologies
# ---------------------------------------------------------------------------

def pe_simd_1_folding() -> dict:
    """Returns a FINN folding config that forces PE=SIMD=1 on all layers."""
    return {"Defaults": {"PE": 1, "SIMD": 1}}


# ---------------------------------------------------------------------------
# Build one DROP model at PE=SIMD=1
# ---------------------------------------------------------------------------

def build_drop(drop_tag: str, dry_run: bool = False) -> tuple[str | None, bool]:
    onnx_fn = ONNX_MAP.get(drop_tag)
    if not onnx_fn:
        print(f"  [!] Unknown tag: {drop_tag}")
        return None, False

    onnx_path = os.path.join(SAT6_SEC_DIR, onnx_fn)
    if not os.path.exists(onnx_path):
        print(f"  [!] ONNX not found: {onnx_path}")
        return None, False

    hw_name      = f"anchor_{drop_tag.lower()}_pe1_simd1"
    run_dir      = os.path.join(BUILD_DIR, hw_name)
    folding_path = os.path.join(BUILD_DIR, f"folding_{hw_name}.json")
    log_path     = os.path.join(BUILD_DIR, f"{hw_name}.log")

    # Skip if already complete
    rpt = os.path.join(run_dir, "stitched_ip", "finn_design_partition_util.rpt")
    if os.path.exists(rpt):
        print(f"  [i] {drop_tag}: already built — skipping.")
        return run_dir, True

    os.makedirs(BUILD_DIR, exist_ok=True)
    with open(folding_path, "w") as f:
        json.dump(pe_simd_1_folding(), f, indent=2)

    cmd = [
        "python3", os.path.join(hara_dir, "run_build.py"),
        "--model_path",   onnx_path,
        "--build_dir",    BUILD_DIR,
        "--hw_name",      hw_name,
        "--steps",        json.dumps(FULL_STEPS),
        "--fpga-part",    FPGA_PART,
        "--folding_file", folding_path,
    ]

    print(f"  [Build] {drop_tag} → {hw_name}")
    print(f"          log: {log_path}")

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return run_dir, True

    t0 = time.time()
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=14400)
        elapsed = (time.time() - t0) / 3600.0
        ok = proc.returncode == 0
        status = "✓" if ok else f"FAILED (exit {proc.returncode})"
        print(f"  [{status}] {drop_tag} in {elapsed:.2f}h")
        return run_dir, ok
    except subprocess.TimeoutExpired:
        print(f"  [!] {drop_tag} timed out")
        return run_dir, False
    except Exception as e:
        print(f"  [!] {drop_tag} exception: {e}")
        return run_dir, False


# ---------------------------------------------------------------------------
# Parse per-module GT from partition_util.rpt + ONNX attrs
# ---------------------------------------------------------------------------

def parse_partition_rpt(rpt_path: str) -> dict[str, dict]:
    """
    Parse finn_design_partition_util.rpt using dynamic header detection.
    Returns {instance_name: {Total LUT, Total FFs, BRAM (36k eq.), DSP Blocks}}
    for direct children of finn_design_i. FIFO chains aggregated.
    (Mirrors parse_util_rpt from multi_module_learner.py)
    """
    import re as _re
    results = {}
    if not os.path.exists(rpt_path):
        return results

    in_utilization_table = False
    header_indices: dict[str, int] = {}
    col_headers: list[str] = []
    found_finn_design_i = False
    finn_design_i_indent = -1

    with open(rpt_path, encoding="utf-8") as f:
        content = f.readlines()

    for line_raw in content:
        line_s = line_raw.strip()

        if not line_s.startswith("|") and "1. Utilization by Hierarchy" in line_s:
            in_utilization_table = True
            continue
        if not in_utilization_table:
            continue
        if not line_s.startswith("|"):
            if found_finn_design_i:
                break
            continue

        # Detect header row dynamically
        if not col_headers and "Instance" in line_s and "Module" in line_s:
            temp = [h.strip() for h in line_s.split("|") if h.strip()]
            if temp and temp[0] == "Instance" and temp[1] == "Module":
                col_headers = temp
                for hdr in ["Total LUTs", "Logic LUTs", "LUTRAMs", "SRLs",
                             "FFs", "RAMB36", "RAMB18", "DSP Blocks"]:
                    if hdr in col_headers:
                        header_indices[hdr] = col_headers.index(hdr)
            continue

        if not header_indices:
            continue

        try:
            first_cell_raw = line_raw.split("|", 2)[1]
        except IndexError:
            continue

        indent        = len(first_cell_raw) - len(first_cell_raw.lstrip(" "))
        instance_name = first_cell_raw.strip()
        data_parts    = [p.strip() for p in line_s.split("|")[1:-1]]

        if len(data_parts) != len(col_headers):
            continue

        if not found_finn_design_i:
            if instance_name == "finn_design_i":
                found_finn_design_i = True
                finn_design_i_indent = indent
            continue

        if indent <= finn_design_i_indent:
            found_finn_design_i = False
            continue

        if indent != finn_design_i_indent + 2:
            continue

        if instance_name.startswith("("):
            continue

        try:
            def _i(h): return int(data_parts[header_indices[h]])   if h in header_indices else 0
            def _f(h): return float(data_parts[header_indices[h]]) if h in header_indices else 0.0
            total_lut = _i("Total LUTs")
            ffs       = _i("FFs")
            dsp       = _i("DSP Blocks")
            bram_eq   = _f("RAMB36") + _f("RAMB18") * 0.5
        except (ValueError, KeyError):
            continue

        # Aggregate FIFO chains: StreamingFIFO_rtl_N_K → StreamingFIFO_N
        m = _re.match(r"StreamingFIFO_(?:rtl|hls)_(\d+)(?:_\d+)?$", instance_name)
        key = f"StreamingFIFO_{m.group(1)}" if m else instance_name

        if key not in results:
            results[key] = {"Total LUT": 0, "Total FFs": 0, "BRAM (36k eq.)": 0.0, "DSP Blocks": 0}
        results[key]["Total LUT"]      += total_lut
        results[key]["Total FFs"]      += ffs
        results[key]["BRAM (36k eq.)"] += bram_eq
        results[key]["DSP Blocks"]     += dsp

    return results


def load_onnx_node_attrs(onnx_path: str) -> dict[str, dict]:
    """Returns {node_name: {attr_name: value, op_type: ...}} for all nodes."""
    import onnx as ox
    from onnx import helper as oxh
    result = {}
    if not os.path.exists(onnx_path):
        return result
    model = ox.load(onnx_path)
    for node in model.graph.node:
        attrs = {"op_type": node.op_type}
        for a in node.attribute:
            try:
                v = oxh.get_attribute_value(a)
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                attrs[a.name] = v
            except Exception:
                pass
        result[node.name] = attrs
    return result


def build_mvau_rows(run_dir: str, drop_tag: str) -> list[dict]:
    """
    Extract per-MVAU GT rows for the training CSV from a completed build.
    run_dir may be the synth dir (has the RPT) or the HLS dir (has ONNX+config).
    We search both the given dir and its hls_cand* sibling for each artifact.
    """
    import re as _re

    def find_artifact(candidates: list[str]) -> str:
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]  # return first (missing) so caller gets a clear path

    build_base = os.path.dirname(run_dir)
    hw_name    = os.path.basename(run_dir)

    # Sibling dir: synth_cand ↔ hls_cand
    sibling = None
    if hw_name.startswith("synth_cand"):
        sibling = os.path.join(build_base, hw_name.replace("synth_cand", "hls_cand", 1))
    elif hw_name.startswith("hls_cand"):
        sibling = os.path.join(build_base, hw_name.replace("hls_cand", "synth_cand", 1))

    dirs = [run_dir] + ([sibling] if sibling else [])

    rpt_path  = find_artifact([os.path.join(d, "stitched_ip",
                                "finn_design_partition_util.rpt") for d in dirs])
    onnx_path = find_artifact([os.path.join(d, "intermediate_models",
                                "step_generate_estimate_reports.onnx") for d in dirs])
    cfg_path  = find_artifact([os.path.join(d, "final_hw_config.json") for d in dirs])

    print(f"    RPT  : {rpt_path}  exists={os.path.exists(rpt_path)}")
    print(f"    ONNX : {onnx_path}  exists={os.path.exists(onnx_path)}")
    print(f"    CFG  : {cfg_path}  exists={os.path.exists(cfg_path)}")

    gt      = parse_partition_rpt(rpt_path)
    print(f"    GT keys: {list(gt.keys())}")
    attrs   = load_onnx_node_attrs(onnx_path)
    mvau_nodes = [n for n in attrs if "MVAU" in attrs[n].get("op_type","") or "MVAU" in n]
    print(f"    ONNX MVAU nodes: {mvau_nodes}")
    cfg     = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}

    model_id = f"SAT6_T2W2_{drop_tag}"  # e.g. SAT6_T2W2_DROP5
    rows = []

    for node_name, node_attrs in attrs.items():
        op_type = node_attrs.get("op_type", "")
        if "MVAU" not in op_type and "MVAU" not in node_name:
            continue

        # Match GT: node names in RPT use underscore convention
        rpt_key = node_name  # direct match
        if rpt_key not in gt:
            # Try without suffix
            rpt_key = node_name.rsplit("_", 1)[0] if "_" in node_name else node_name
        if rpt_key not in gt:
            continue

        area = gt[rpt_key]
        folding_cfg = cfg.get(node_name, {})

        row = {
            "model_id":   model_id,
            "session":    f"anchor_{drop_tag.lower()}",
            "timestamp":  "",
            "run_name":   f"anchor_pe1_simd1",
            "run_number": 1,
            "is_baseline": 1,
            "fixed_ram_style": folding_cfg.get("ram_style", "block"),
            "fixed_resType":   folding_cfg.get("resType", "dsp"),
            "Submodule Instance": node_name,
            "base_name":  "MVAU",
            "isRTL": 0,
            "isHLS": 1,
            "layer_idx": 0,
            # MVAU-specific attrs
            "MH":   node_attrs.get("MH", 0),
            "MW":   node_attrs.get("MW", 0),
            "PE":   folding_cfg.get("PE", node_attrs.get("PE", 1)),
            "SIMD": folding_cfg.get("SIMD", node_attrs.get("SIMD", 1)),
            "inputDataType":  node_attrs.get("inputDataType", ""),
            "weightDataType": node_attrs.get("weightDataType", ""),
            "outputDataType": node_attrs.get("outputDataType", ""),
            "ram_style":  folding_cfg.get("ram_style", "block"),
            "resType":    folding_cfg.get("resType", "dsp"),
            "mem_mode":   node_attrs.get("mem_mode", "internal_decoupled"),
            "binaryXnorMode": node_attrs.get("binaryXnorMode", 0),
            # targets
            "Total LUT":      area["Total LUT"],
            "Total FFs":      area["Total FFs"],
            "BRAM (36k eq.)": area["BRAM (36k eq.)"],
            "DSP Blocks":     area["DSP Blocks"],
            # anchor weight marker
            "_anchor_weight": ANCHOR_WEIGHT,
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Retrain MVAU_SAT6_T2W2
# ---------------------------------------------------------------------------

def retrain_mvau(new_rows: list[dict], anchor_weight: float = ANCHOR_WEIGHT):
    if not os.path.exists(MVAU_CSV):
        print(f"[!] Base training CSV not found: {MVAU_CSV}")
        return False

    df_base = pd.read_csv(MVAU_CSV)
    # Filter to SAT6_T2W2 only (same as train_xgboost.py prepare_module_df)
    df_sat6 = df_base[df_base["model_id"] == "SAT6_T2W2"].copy()
    print(f"  Base SAT6_T2W2 samples : {len(df_sat6)}")

    df_new = pd.DataFrame(new_rows)
    print(f"  New anchor samples     : {len(df_new)}")

    # Align columns — add missing base cols to df_new as NaN, preserve _anchor_weight
    anchor_weights_new = df_new["_anchor_weight"].values if "_anchor_weight" in df_new.columns \
                         else np.full(len(df_new), anchor_weight)
    for col in df_sat6.columns:
        if col not in df_new.columns:
            df_new[col] = np.nan
    df_new = df_new[[c for c in df_sat6.columns if c in df_new.columns]]

    df_aug = pd.concat([df_sat6, df_new], ignore_index=True)
    df_aug.to_csv(MVAU_CSV_AUG, index=False)
    print(f"  Augmented CSV saved    : {MVAU_CSV_AUG}")

    # Build sample_weight vector: base rows=1, anchor rows=anchor_weight
    sample_weights = np.concatenate([
        np.ones(len(df_sat6), dtype=np.float32),
        anchor_weights_new.astype(np.float32),
    ])

    # --- prepare features (mirrors train_xgboost.py prepare_module_df) ---
    LEAKAGE = ["model_id", "session", "timestamp", "run_name", "run_number",
               "is_baseline", "fixed_ram_style", "fixed_resType",
               "Submodule Instance", "base_name", "layer_idx", "Hardware config",
               "_anchor_weight"]
    EXTRA_DROPS = ["cycles_estimate", "estimated_cycles", "op_type",
                   "runtime_writeable_weights", "mem_mode"]
    BITWIDTH_COLS = ["inputDataType", "weightDataType", "outputDataType"]

    def extract_bitwidth(val):
        if pd.isna(val):
            return 1
        s = str(val)
        import re
        m = re.search(r"\d+", s)
        return int(m.group()) if m else 1

    df = df_aug.drop(columns=[c for c in LEAKAGE if c in df_aug.columns], errors="ignore")

    for col in BITWIDTH_COLS:
        if col in df.columns:
            df.insert(df.columns.get_loc(col) + 1,
                      f"{col} (bits)", df[col].apply(extract_bitwidth))
            df.drop(columns=[col], inplace=True)

    # OHE ram_style, resType
    for cat_col in ["ram_style", "resType"]:
        if cat_col not in df.columns:
            continue
        dummies = pd.get_dummies(df[cat_col], prefix=cat_col, dtype=int)
        df.drop(columns=[cat_col], inplace=True)
        df = pd.concat([df, dummies], axis=1)

    # mac_complexity engineered feature
    in_bits = pd.to_numeric(df.get("inputDataType (bits)", 1), errors="coerce").fillna(1)
    w_bits  = pd.to_numeric(df.get("weightDataType (bits)", 1), errors="coerce").fillna(1)
    pe      = pd.to_numeric(df.get("PE", 1), errors="coerce").fillna(1)
    simd    = pd.to_numeric(df.get("SIMD", 1), errors="coerce").fillna(1)
    df["mac_complexity"] = in_bits * w_bits * pe * simd

    # list-like columns → scalar
    for col in df.columns:
        if df[col].dtype == object:
            def _clean(v):
                if pd.isna(v):
                    return np.nan
                s = str(v).strip()
                if s.startswith("["):
                    try:
                        lst = ast.literal_eval(s)
                        if isinstance(lst, (list, tuple)):
                            return float(np.prod(lst)) if lst else np.nan
                    except Exception:
                        pass
                try:
                    return float(s)
                except Exception:
                    return np.nan
            df[col] = df[col].apply(_clean)

    # log1p targets
    for t in TARGET_COLS:
        if t in df.columns:
            df[t] = np.log1p(pd.to_numeric(df[t], errors="coerce").fillna(0))

    df.drop(columns=[c for c in EXTRA_DROPS if c in df.columns], inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Drop zero-variance columns (but protect key features)
    PROTECTED = ["inWidth", "depth", "bit_capacity", "mac_complexity",
                 "weightDataType (bits)", "inputDataType (bits)"]
    low_var = [c for c in df.columns if df[c].nunique() <= 1
               and c not in TARGET_COLS and c not in PROTECTED]
    df.drop(columns=low_var, inplace=True)

    missing = [t for t in TARGET_COLS if t not in df.columns]
    if missing:
        print(f"[!] Missing target columns: {missing}")
        return False

    y = df[TARGET_COLS].values.astype(np.float32)
    X = df.drop(columns=TARGET_COLS).values.astype(np.float32)
    feature_names = [c for c in df.columns if c not in TARGET_COLS]

    print(f"\n  Features : {X.shape[1]}")
    print(f"  Samples  : {X.shape[0]}  "
          f"(base={int((sample_weights == 1).sum())}  "
          f"anchors={int((sample_weights > 1).sum())})")

    xgb_params = dict(n_estimators=350, max_depth=7, learning_rate=0.05,
                      random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(XGBRegressor(**xgb_params))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, sample_weight=sample_weights)

    # Quick in-sample MAPE on anchor rows (real scale)
    anchor_mask = sample_weights > 1
    if anchor_mask.any():
        y_pred_anchor = model.predict(X[anchor_mask])
        print(f"\n  In-sample error on anchor rows (real scale):")
        for i, t in enumerate(TARGET_COLS):
            real = np.expm1(y[anchor_mask, i])
            pred = np.expm1(y_pred_anchor[:, i])
            mape = np.mean(np.abs(pred - real) / np.clip(real, 1, None)) * 100
            print(f"    {t:<22}: MAPE={mape:.1f}%")

    # Save updated model
    backup = MVAU_MODEL_PKL + ".bak"
    if os.path.exists(MVAU_MODEL_PKL):
        import shutil
        shutil.copy2(MVAU_MODEL_PKL, backup)
        print(f"\n  Backup saved → {backup}")

    with open(MVAU_MODEL_PKL, "wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names}, f)
    print(f"  Updated model saved → {MVAU_MODEL_PKL}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Retrain MVAU with pruned-model anchors")
    parser.add_argument("--skip-builds", action="store_true",
                        help="Skip FINN builds, use existing build dirs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show build commands without executing")
    parser.add_argument("--drops", type=str, default="0,1,2,3,4,5",
                        help="Comma-separated DROP indices to build (default: 0,1,2,3,4,5)")
    parser.add_argument("--weight", type=float, default=ANCHOR_WEIGHT,
                        help=f"Sample weight for anchor rows (default: {ANCHOR_WEIGHT})")
    parser.add_argument("--extra-dirs", type=str, default="",
                        help="Extra 'TAG:path' pairs for existing builds, comma-separated. "
                             "E.g. DROP5:/path/to/synth_cand01_drop5_run9")
    args = parser.parse_args()

    anchor_weight = args.weight
    drop_tags = [f"DROP{i}" for i in map(int, args.drops.split(","))]

    print("=" * 70)
    print(" MVAU Anchor Retraining — pruned model calibration")
    print(f"   drops={drop_tags}  anchor_weight={anchor_weight}")
    print("=" * 70)

    # Parse any manually-specified existing build dirs
    extra_dirs: dict[str, str] = {}
    if args.extra_dirs:
        for pair in args.extra_dirs.split(","):
            if ":" in pair:
                tag, path = pair.split(":", 1)
                extra_dirs[tag.strip()] = path.strip()

    # [1] Build or locate each DROP model
    built_dirs: dict[str, str] = {}

    # Pre-populate with explicitly provided dirs
    for tag, path in extra_dirs.items():
        rpt = os.path.join(path, "stitched_ip", "finn_design_partition_util.rpt")
        if os.path.exists(rpt):
            built_dirs[tag] = path
            print(f"  ✓ {tag} (manual): {path}")
        else:
            print(f"  [!] {tag} (manual): RPT missing at {rpt}")

    if args.skip_builds:
        print("\n[1] Skipping builds — scanning existing dirs…")
        for tag in drop_tags:
            if tag in built_dirs:
                continue  # already provided via --extra-dirs
            hw_name = f"anchor_{tag.lower()}_pe1_simd1"
            run_dir = os.path.join(BUILD_DIR, hw_name)
            rpt     = os.path.join(run_dir, "stitched_ip", "finn_design_partition_util.rpt")
            if os.path.exists(rpt):
                built_dirs[tag] = run_dir
                print(f"  ✓ {tag}: {run_dir}")
            else:
                print(f"  [!] {tag}: RPT missing at {rpt}")
    else:
        print(f"\n[1] Building {len(drop_tags)} DROP models at PE=SIMD=1…")
        for tag in drop_tags:
            if tag in built_dirs:
                print(f"\n  ── {tag} (already provided, skipping build) ──")
                continue
            print(f"\n  ── {tag} ──")
            run_dir, ok = build_drop(tag, dry_run=args.dry_run)
            if ok and run_dir:
                built_dirs[tag] = run_dir

    if args.dry_run:
        print("\n[DRY RUN] Stopping before GT extraction.")
        return

    if not built_dirs:
        print("[!] No completed builds found. Aborting.")
        return

    # [2] Extract per-module GT
    print(f"\n[2] Extracting per-MVAU GT from {len(built_dirs)} builds…")
    all_new_rows: list[dict] = []
    for tag, run_dir in built_dirs.items():
        rows = build_mvau_rows(run_dir, tag)
        print(f"  {tag}: {len(rows)} MVAU rows extracted")
        all_new_rows.extend(rows)

    if not all_new_rows:
        print("[!] No GT rows extracted. Check build completeness.")
        return

    # [3] Retrain
    print(f"\n[3] Retraining MVAU_SAT6_T2W2 with {len(all_new_rows)} anchor rows…")
    for r in all_new_rows:
        r["_anchor_weight"] = anchor_weight
    ok = retrain_mvau(all_new_rows, anchor_weight=anchor_weight)
    if ok:
        print("\n  Done. To validate, run:")
        print("    python3 ai/evaluate_all_builds.py")
    else:
        print("\n  [!] Retraining failed.")


if __name__ == "__main__":
    main()
