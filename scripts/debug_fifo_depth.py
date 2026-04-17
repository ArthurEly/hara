#!/usr/bin/env python3
"""
debug_fifo_depth.py — Diagnóstico completo do pipeline de predição de FIFO depth.

Mostra para cada FIFO:
  1. Nome do nó, produtor e consumidor
  2. Se os nomes batem com as chaves do folding config
  3. PE/SIMD extraídos, MH/MW dos nós
  4. Valores de cycle_ratio, speed_ratio e demais features
  5. Probabilidade do Stage-1 classifier e depth final

Uso (dentro do container FINN):
    python3 scripts/debug_fifo_depth.py [campaign_dir]

Exemplo:
    python3 scripts/debug_fifo_depth.py \
        fps_campaign_results/SAT6_T2W2_PREBUILT_20260416_060222
"""

import os
import sys
import csv
import json
import pickle
import numpy as np
import warnings
import re

hara_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, hara_dir)
sys.path.insert(0, os.path.join(hara_dir, "ai"))

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from onnx import helper

MODELS_DIR = os.path.join(hara_dir, "ai", "retrieval", "results", "trained_models")

TOPOLOGY_TO_SPECIALIST = {
    "SAT6":   "StreamingFIFO_depth_SAT6_T2",
    "MNIST":  "StreamingFIFO_depth_MNIST_TFC",
    "CIFAR10": "StreamingFIFO_depth_CIFAR10_CNV",
}


# ---------------------------------------------------------------------------
# Helpers (duplicados de predict_fifo_depths.py para isolamento)
# ---------------------------------------------------------------------------

def get_node_attr(node, attr_name, default=1):
    if node is None:
        return default
    for a in node.attribute:
        if a.name == attr_name:
            try:
                return helper.get_attribute_value(a)
            except Exception:
                return a.i if a.type == 2 else default
    return default


def get_folding_attr(name, folding_cfg, attr_name, default=1):
    if name in folding_cfg:
        return folding_cfg[name].get(attr_name, default)
    match = re.search(r'^(.*)_(\d+)$', name)
    if match:
        base, idx = match.groups()
        for backend in ["_rtl_", "_hls_", "_vivado_"]:
            alt = f"{base}{backend}{idx}"
            if alt in folding_cfg:
                return folding_cfg[alt].get(attr_name, default)
    return default


def extract_bits(dtype_str):
    if not dtype_str:
        return 8
    s = str(dtype_str).upper()
    if any(x in s for x in ["BINARY", "BIPOLAR", "INT1", "UINT1"]):
        return 1
    nums = re.findall(r'\d+', s)
    return int(nums[0]) if nums else 8


def find_canonical_onnx(campaign_dir):
    candidates = [
        os.path.join(campaign_dir, "run0_get_initial_fold", "intermediate_models",
                     "step_generate_estimate_reports.onnx"),
        os.path.join(campaign_dir, "run1_baseline_folded", "intermediate_models",
                     "step_generate_estimate_reports.onnx"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for entry in sorted(os.listdir(campaign_dir)):
        p = os.path.join(campaign_dir, entry, "intermediate_models",
                         "step_generate_estimate_reports.onnx")
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Debug core
# ---------------------------------------------------------------------------

def debug_one_fifo(node, model, folding_cfg, m_data, folding_index=0):
    """Prints a full diagnostic row for a single FIFO node."""
    prod_map = {}
    cons_map = {}
    for n in model.graph.node:
        for t in n.output:
            prod_map[t] = n
        for t in n.input:
            cons_map[t] = n

    p_node = prod_map.get(node.input[0])
    c_node = cons_map.get(node.output[0])

    p_name = p_node.name if p_node else "<<Input>>"
    p_op   = p_node.op_type if p_node else "<<Input>>"
    c_name = c_node.name if c_node else "<<Output>>"
    c_op   = c_node.op_type if c_node else "<<Output>>"

    p_in_folding = p_name in folding_cfg
    c_in_folding = c_name in folding_cfg

    # Spatial attrs
    p_mh = get_node_attr(p_node, "MH", 1)
    p_mw = get_node_attr(p_node, "MW", 1)
    c_mh = get_node_attr(c_node, "MH", 1)
    c_mw = get_node_attr(c_node, "MW", 1)

    p_pe   = get_folding_attr(p_name, folding_cfg, "PE",   1)
    p_simd = get_folding_attr(p_name, folding_cfg, "SIMD", 1)
    c_pe   = get_folding_attr(c_name, folding_cfg, "PE",   1)
    c_simd = get_folding_attr(c_name, folding_cfg, "SIMD", 1)

    p_cycles = max(1, (p_mh * p_mw) // max(1, p_pe * p_simd))
    c_cycles = max(1, (c_mh * c_mw) // max(1, c_pe * c_simd))

    p_throughput = float(p_pe)   / max(1, p_cycles)
    c_throughput = float(c_simd) / max(1, c_cycles)
    cycle_ratio  = float(p_cycles) / max(1, c_cycles)
    speed_ratio  = p_throughput / max(1e-12, c_throughput)

    try:
        shape = model.get_tensor_shape(node.input[0])
        tensor_volume   = int(np.prod(shape))
        dtype           = model.get_tensor_datatype(node.input[0])
        bits            = extract_bits(str(dtype))
        tensor_spatial  = shape[1] * shape[2] if len(shape) > 2 else 1
    except Exception:
        tensor_volume  = 1
        bits           = 8
        tensor_spatial = 1
        shape          = [1, 1, 1, 1]

    # Stage-1 prediction (if model available)
    has_v2 = "stage1_classifier" in m_data
    stage1_proba = None
    final_depth  = None

    if has_v2:
        raw_data = {
            "dataType_bits": float(bits),
            "tensor_volume": float(tensor_volume),
            "tensor_spatial": float(tensor_spatial),
            "tensor_H": float(shape[1] if len(shape) > 1 else 1),
            "tensor_W": float(shape[2] if len(shape) > 2 else 1),
            "tensor_C": float(shape[3] if len(shape) > 3 else bits),
            "produtor_PE": float(p_pe),
            "produtor_cycles": float(p_cycles),
            "p_throughput": float(p_throughput),
            "consumidor_SIMD": float(c_simd),
            "consumidor_cycles": float(c_cycles),
            "c_throughput": float(c_throughput),
            "cycle_ratio": float(cycle_ratio),
            "speed_ratio": float(speed_ratio),
            "parallelism_mismatch": abs(p_pe - c_simd),
            "p_transfers": float(tensor_volume),
            "c_transfers": float(tensor_volume),
            "chain_length": 1.0,
            "produtor_op": p_op,
            "consumidor_op": c_op,
            "op_pair": f"{p_op}→{c_op}",
            "ram_style": "auto",
            "impl_style": "rtl",
        }
        for col in ["cycle_ratio", "tensor_volume", "consumidor_cycles", "chain_length"]:
            raw_data[f"log_{col}"] = np.log1p(raw_data.get(col, 0))
        raw_data["log_speed_ratio"]  = np.log1p(speed_ratio)
        raw_data["chain_x_cycles"]   = np.log1p(1.0 * c_cycles)
        raw_data["spatial_x_cycle"]  = np.log1p(tensor_spatial * cycle_ratio)
        raw_data["is_spatial_large"] = 1 if tensor_spatial > 256 else 0

        feats = []
        for f in m_data["feature_names"]:
            if f in raw_data:
                feats.append(float(raw_data[f]))
            elif f.startswith("produtor_op_") and f.endswith(p_op):
                feats.append(1.0)
            elif f.startswith("consumidor_op_") and f.endswith(c_op):
                feats.append(1.0)
            elif f.startswith("op_pair_") and f.endswith(raw_data["op_pair"]):
                feats.append(1.0)
            elif f.startswith("ram_style_") and f.endswith("auto"):
                feats.append(1.0)
            elif f.startswith("impl_style_") and f.endswith("rtl"):
                feats.append(1.0)
            else:
                feats.append(0.0)

        X = np.array(feats, dtype=np.float32).reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stage1_proba = m_data["stage1_classifier"].predict_proba(X)[:, 1][0]
            threshold    = m_data.get("stage1_threshold", 0.55)
            if stage1_proba >= threshold:
                final_depth = 2
            else:
                log_pred    = m_data["stage2_regressor"].predict(X)[0]
                final_depth = int(max(2, np.round(np.expm1(log_pred))))

        # Check for all-zero features (key diagnostic)
        named_feats = dict(zip(m_data["feature_names"], feats))
        zero_count  = sum(1 for v in feats if v == 0.0)

    print(f"\n  FIFO: {node.name}")
    print(f"    Produtor  : {p_name} ({p_op})  [no folding? {'NAO' if p_in_folding else 'SIM — PROBLEMA!'}]")
    print(f"    Consumidor: {c_name} ({c_op})  [no folding? {'NAO' if c_in_folding else 'SIM — PROBLEMA!'}]")
    print(f"    MH/MW (prod): {p_mh}/{p_mw}   MH/MW (cons): {c_mh}/{c_mw}")
    print(f"    PE={p_pe} SIMD_prod={p_simd}   c_PE={c_pe} c_SIMD={c_simd}")
    print(f"    p_cycles={p_cycles}  c_cycles={c_cycles}")
    print(f"    cycle_ratio={cycle_ratio:.3f}  speed_ratio={speed_ratio:.3f}")
    print(f"    tensor: shape={shape}  bits={bits}  volume={tensor_volume}  spatial={tensor_spatial}")
    if stage1_proba is not None:
        flag = "  ← SEMPRE depth=2!" if stage1_proba >= threshold and cycle_ratio <= 2 else ""
        print(f"    Stage-1 proba(depth==2)={stage1_proba:.4f}  threshold={threshold:.2f}{flag}")
        print(f"    Predicted depth: {final_depth}")
        non_zero = [(k, v) for k, v in named_feats.items() if v != 0.0]
        print(f"    Features não-zero: {len(non_zero)}/{len(feats)}  (zeros={zero_count})")
        if non_zero:
            print(f"    Top features ativas: {non_zero[:8]}")


def run_debug(campaign_dir, folding_index=0, n_fifos=None):
    print("=" * 70)
    print(f" HARA — Debug FIFO Depth Prediction")
    print(f" Campaign: {os.path.basename(campaign_dir)}")
    print("=" * 70)

    onnx_path = find_canonical_onnx(campaign_dir)
    if not onnx_path:
        print("[!] ONNX não encontrado.")
        return
    print(f"\nONNX: {onnx_path}")

    csv_path = os.path.join(campaign_dir, "fps_map.csv")
    if not os.path.exists(csv_path):
        print(f"[!] fps_map.csv não encontrado.")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if folding_index >= len(rows):
        folding_index = 0
    folding_cfg = json.loads(rows[folding_index]["folding_config"])

    print(f"\n--- Folding config #{folding_index} ({len(folding_cfg)} chaves) ---")
    for k, v in folding_cfg.items():
        print(f"  {k:<45} -> {v}")

    # Load depth specialist
    specialist_path = None
    for topo_key, model_name in TOPOLOGY_TO_SPECIALIST.items():
        if topo_key in os.path.basename(campaign_dir).upper():
            p = os.path.join(MODELS_DIR, f"{model_name}_model.pkl")
            if os.path.exists(p):
                specialist_path = p
                break
    if not specialist_path:
        unified = os.path.join(MODELS_DIR, "StreamingFIFO_depth_UNIFIED_model.pkl")
        if os.path.exists(unified):
            specialist_path = unified

    m_data = {}
    if specialist_path:
        with open(specialist_path, "rb") as f:
            m_data = pickle.load(f)
        print(f"\nEspecialista carregado: {os.path.basename(specialist_path)}")
        print(f"  Features esperadas: {len(m_data['feature_names'])}")
        print(f"  Tem Stage-1: {'stage1_classifier' in m_data}")
    else:
        print("\n[!] Nenhum especialista de depth encontrado.")

    # Expand ONNX
    print("\n--- Expandindo topologia... ---")
    model = ModelWrapper(onnx_path)
    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    fifo_nodes = [n for n in model.graph.node if "FIFO" in n.op_type.upper()]
    print(f"Total FIFOs na topologia expandida: {len(fifo_nodes)}")

    # Check name matching
    all_node_names = [n.name for n in model.graph.node]
    folding_keys   = set(folding_cfg.keys()) - {"Defaults"}
    matched = set(all_node_names) & folding_keys
    unmatched_nodes = [n for n in all_node_names if n and n not in folding_keys
                       and "StreamingFIFO" not in n and "DWC" not in n]

    print(f"\n--- Name matching summary ---")
    print(f"  Nós no ONNX expandido: {len(all_node_names)}")
    print(f"  Chaves no folding:     {len(folding_keys)}")
    print(f"  Nomes que batem:       {len(matched)}")
    if unmatched_nodes:
        print(f"  Nós SEM match no folding (não-FIFO/DWC):")
        for nm in unmatched_nodes:
            print(f"    {nm}")

    if n_fifos is not None:
        fifo_nodes = fifo_nodes[:n_fifos]

    print(f"\n{'='*70}")
    print(f" Debug por FIFO (folding #{folding_index})")
    print(f"{'='*70}")
    for node in fifo_nodes:
        debug_one_fifo(node, model, folding_cfg, m_data, folding_index)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: first SAT6_T2W2 campaign found
        base = os.path.join(hara_dir, "fps_campaign_results")
        candidates = sorted([
            os.path.join(base, d) for d in os.listdir(base)
            if "SAT6_T2W2" in d and os.path.isdir(os.path.join(base, d))
        ])
        if not candidates:
            print("Uso: python3 debug_fifo_depth.py <campaign_dir> [folding_index] [n_fifos]")
            sys.exit(1)
        campaign = candidates[0]
    else:
        campaign = sys.argv[1]
        if not os.path.isabs(campaign):
            campaign = os.path.join(os.getcwd(), campaign)

    folding_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n_fifos     = int(sys.argv[3]) if len(sys.argv) > 3 else None

    run_debug(campaign, folding_index=folding_idx, n_fifos=n_fifos)
