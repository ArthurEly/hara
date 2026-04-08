"""
get_exhaustive_fifo_depths.py

Extrai o dataset de treino para o modelo de predição de FIFO depth.

Para cada StreamingFIFO_rtl_N em cada build, constrói um vetor de features
com informações das camadas UPSTREAM e DOWNSTREAM adjacentes no pipeline.

Fonte dos dados:
  - final_hw_config.json  → topologia (ordem pipeline) + depth (target)
  - run_optimized.json    → PE / SIMD de cada camada (folding)
  - step_generate_estimate_reports.onnx → MH, MW, IFMChannels, bitwidths, etc.

Saída:
  results/fifo_depth/exhaustive_fifo_depths.csv
"""

import os
import re
import json
import glob
import numpy as np
import pandas as pd
from collections import OrderedDict

import onnx
from onnx import helper

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

EXHAUSTIVE_BUILDS_DIR = "/home/arthurely/Desktop/finn_chi2p/hara/exhaustive_hw_builds"

# Treino: MNIST + SAT6_T2 | Excluir CIFAR10 (usado como teste)
FILTER_MODEL_IDS = ["MNIST", "SAT6_T2"]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "fifo_depth")

ONNX_FILENAME   = "step_generate_estimate_reports.onnx"
ONNX_SUBDIR     = "intermediate_models"
FOLDING_FILENAME = "final_hw_config.json"
FOLDING_REQ_FILENAME = None  # run_N_optimized.json — calculado dinamicamente

# Atributos ONNX úteis por op_type
ONNX_ATTRS_WANTED = [
    "MH", "MW", "SIMD", "PE",
    "IFMChannels", "OFMChannels",
    "IFMDim", "OFMDim",
    "KernelDim", "Stride", "Dilation", "Padding",
    "inputDataType", "outputDataType", "weightDataType", "accDataType",
    "numInputVectors", "numChannels", "Labels",
    "inWidth", "outWidth",
]

# =============================================================================


def _parse_dtype_bits(dtype_str):
    """Extrai número de bits de strings como UINT8, INT4, BIPOLAR."""
    if dtype_str is None:
        return None
    s = str(dtype_str).upper()
    if "BINARY" in s or "BIPOLAR" in s:
        return 1
    m = re.search(r"(UINT|INT)(\d+)", s)
    return int(m.group(2)) if m else None


def _flatten_attr(val):
    """Converte atributo ONNX para escalar numérico ou string."""
    if isinstance(val, (list, tuple)):
        arr = [v for v in val if v is not None]
        return int(np.prod(arr)) if arr else 0
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except Exception:
            return str(val)
    if isinstance(val, np.ndarray):
        return int(np.prod(val)) if val.size > 0 else 0
    return val


def load_onnx_attrs(onnx_path):
    """
    Retorna dict {node_name: {attr: value, ...}} com atributos de interesse.
    Tenta também derivar node_name sem sufixo de tipo (ex: MVAU_0 → MVAU_hls_0).
    """
    if not os.path.exists(onnx_path):
        return {}
    node_map = {}
    try:
        model = onnx.load(onnx_path)
        for node in model.graph.node:
            attrs = {"op_type": node.op_type}
            for attr in node.attribute:
                val = _flatten_attr(helper.get_attribute_value(attr))
                attrs[attr.name] = val
                # Para datatypes, adicionar versão em bits
                if "DataType" in attr.name or "datatype" in attr.name.lower():
                    attrs[attr.name + "_bits"] = _parse_dtype_bits(str(val))
            node_map[node.name] = attrs
    except Exception as e:
        print(f"  [!] Erro ao carregar ONNX {onnx_path}: {e}")
    return node_map


def get_onnx_attrs_for_layer(layer_name, onnx_map):
    """
    Tenta vários padrões para encontrar atributos ONNX do layer_name.
    Ex: MVAU_hls_0 → procura MVAU_0, MVAU_hls_0, MVAU, ...
    """
    # Tenta direto
    if layer_name in onnx_map:
        return onnx_map[layer_name]

    # Remove sufixo _rtl/_hls e tenta NOME_idx
    m = re.match(r"^(.+?)_(rtl|hls)_(\d+)$", layer_name)
    if m:
        base, _, idx = m.group(1), m.group(2), m.group(3)
        cands = [f"{base}_{idx}", base]
        for c in cands:
            if c in onnx_map:
                return onnx_map[c]

    return {}


def load_folding(folding_path):
    """Carrega folding JSON mantendo ORDEM (OrderedDict preserva topologia)."""
    if not os.path.exists(folding_path):
        return OrderedDict()
    try:
        with open(folding_path) as f:
            return json.load(f, object_pairs_hook=OrderedDict)
    except Exception:
        return OrderedDict()


def extract_scalar(d, key, default=0):
    val = d.get(key, default)
    if isinstance(val, (list, tuple)):
        return val[0] if val else default
    return val if val is not None else default


def build_layer_feature_dict(layer_name, folding_cfg, onnx_map, prefix=""):
    """
    Monta um dict de features para uma camada (upstream ou downstream).
    prefix: "up_" ou "dn_"
    """
    layer_fold = folding_cfg.get(layer_name, {})
    onnx_attrs = get_onnx_attrs_for_layer(layer_name, onnx_map)

    # Tipo base da camada (sem sufixo _rtl/_hls_N)
    m = re.match(r"^(.+?)_(rtl|hls)_(\d+)$", layer_name)
    if m:
        base_type = m.group(1)
        impl_type  = m.group(2)  # rtl ou hls
    else:
        base_type = layer_name
        impl_type = "unknown"

    feats = {
        f"{prefix}layer_type":  base_type,
        f"{prefix}impl_type":   impl_type,
        f"{prefix}PE":          extract_scalar(layer_fold, "PE", 1),
        f"{prefix}SIMD":        extract_scalar(layer_fold, "SIMD", 1),
        f"{prefix}op_type":     onnx_attrs.get("op_type", base_type),
        f"{prefix}MH":          extract_scalar(onnx_attrs, "MH", 0),
        f"{prefix}MW":          extract_scalar(onnx_attrs, "MW", 0),
        f"{prefix}IFMChannels": extract_scalar(onnx_attrs, "IFMChannels", 0),
        f"{prefix}OFMChannels": extract_scalar(onnx_attrs, "OFMChannels", 0),
        f"{prefix}numChannels": extract_scalar(onnx_attrs, "numChannels", 0),
        f"{prefix}Labels":      extract_scalar(onnx_attrs, "Labels", 0),
        f"{prefix}inWidth":     extract_scalar(onnx_attrs, "inWidth", 0),
        f"{prefix}outWidth":    extract_scalar(onnx_attrs, "outWidth", 0),
        # IFMDim/OFMDim podem ser lista [H, W] → produto
        f"{prefix}IFMDim":      extract_scalar(onnx_attrs, "IFMDim", 0),
        f"{prefix}OFMDim":      extract_scalar(onnx_attrs, "OFMDim", 0),
        f"{prefix}KernelDim":   extract_scalar(onnx_attrs, "KernelDim", 0),
        f"{prefix}Stride":      extract_scalar(onnx_attrs, "Stride", 1),
        # Bitwidths de datatypes
        f"{prefix}inputBits":   onnx_attrs.get("inputDataType_bits", 0) or 0,
        f"{prefix}outputBits":  onnx_attrs.get("outputDataType_bits", 0) or 0,
        f"{prefix}weightBits":  onnx_attrs.get("weightDataType_bits", 0) or 0,
        f"{prefix}accBits":     onnx_attrs.get("accDataType_bits", 0) or 0,
    }
    return feats


def collect_fifo_depths():
    """Itera sobre builds e coleta um registro por FIFO."""
    all_rows = []

    build_sessions = sorted(os.listdir(EXHAUSTIVE_BUILDS_DIR))

    for session_name in build_sessions:
        session_path = os.path.join(EXHAUSTIVE_BUILDS_DIR, session_name)
        if not os.path.isdir(session_path):
            continue

        if FILTER_MODEL_IDS:
            if not any(session_name.startswith(mid) for mid in FILTER_MODEL_IDS):
                continue

        # model_id e timestamp
        model_id  = session_name[:-20] if len(session_name) > 20 else session_name
        timestamp = session_name[-19:] if len(session_name) > 20 else ""

        print(f"\n[>] Sessão: {session_name}")

        # Lista de run_dirs
        run_dirs = sorted(
            [d for d in os.listdir(session_path)
             if os.path.isdir(os.path.join(session_path, d))
             and d.startswith("run") and not d.endswith("_model_files")],
            key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0
        )

        for run_name in run_dirs:
            run_path     = os.path.join(session_path, run_name)
            hw_cfg_path  = os.path.join(run_path, FOLDING_FILENAME)
            onnx_path    = os.path.join(run_path, ONNX_SUBDIR, ONNX_FILENAME)

            if not os.path.exists(hw_cfg_path):
                continue

            print(f"  -> {run_name}...")

            run_number  = int(re.search(r"\d+", run_name).group()) if re.search(r"\d+", run_name) else -1
            is_baseline = 1 if "baseline" in run_name else 0

            # Carrega folding completo (final_hw_config) com ordem preservada
            hw_cfg = load_folding(hw_cfg_path)
            # Carrega ONNX
            onnx_map = load_onnx_attrs(onnx_path)

            # Lista de camadas em ordem de pipeline (sem Defaults)
            pipeline = [k for k in hw_cfg.keys() if k != "Defaults"]

            for pos, layer_name in enumerate(pipeline):
                # Só nos interessa extrair features para as FIFOs
                if "StreamingFIFO" not in layer_name:
                    continue

                layer_cfg = hw_cfg[layer_name]
                depth = layer_cfg.get("depth", 2)
                impl_style = layer_cfg.get("impl_style", "rtl")

                # Camada upstream (posição anterior)
                upstream_name = pipeline[pos - 1] if pos > 0 else None
                # Camada downstream (posição seguinte)
                downstream_name = pipeline[pos + 1] if pos < len(pipeline) - 1 else None

                # Features do upstream
                up_feats = build_layer_feature_dict(upstream_name, hw_cfg, onnx_map, prefix="up_") \
                    if upstream_name else {f"up_{k}": 0 for k in ["PE","SIMD","MH","MW","IFMChannels","OFMChannels","numChannels","Labels","inWidth","outWidth","IFMDim","OFMDim","KernelDim","Stride","inputBits","outputBits","weightBits","accBits"]}

                # Features do downstream
                dn_feats = build_layer_feature_dict(downstream_name, hw_cfg, onnx_map, prefix="dn_") \
                    if downstream_name else {f"dn_{k}": 0 for k in ["PE","SIMD","MH","MW","IFMChannels","OFMChannels","numChannels","Labels","inWidth","outWidth","IFMDim","OFMDim","KernelDim","Stride","inputBits","outputBits","weightBits","accBits"]}

                # FIFO's own outFIFODepths do upstream (cross-check)
                up_out_fifo = extract_scalar(hw_cfg.get(upstream_name, {}), "outFIFODepths", 0) \
                    if upstream_name else 0

                row = {
                    # Identificadores
                    "model_id":     model_id,
                    "session":      session_name,
                    "timestamp":    timestamp,
                    "run_name":     run_name,
                    "run_number":   run_number,
                    "is_baseline":  is_baseline,
                    "fifo_name":    layer_name,
                    "fifo_pos":     pos,        # posição no pipeline
                    "fifo_impl_style": impl_style,
                    # Target
                    "depth":        depth,
                    "is_constrained": 1 if depth > 2 else 0,
                    # Cross-check upstream outFIFODepths
                    "up_out_fifo_depths": up_out_fifo,
                    **up_feats,
                    **dn_feats,
                }
                all_rows.append(row)

    return all_rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("HARA — FIFO Depth Dataset Builder")
    print(f"Diretório: {EXHAUSTIVE_BUILDS_DIR}")
    print(f"Filtro:    {FILTER_MODEL_IDS}")
    print("=" * 60)

    rows = collect_fifo_depths()

    if not rows:
        print("[!] Nenhum dado coletado.")
        return

    df = pd.DataFrame(rows)

    print(f"\n[✓] Total de FIFOs coletadas: {len(df)}")
    print(f"    is_constrained=1 (depth > 2): {df['is_constrained'].sum()} ({df['is_constrained'].mean()*100:.1f}%)")
    print(f"    depth range: {df['depth'].min()} → {df['depth'].max()}")
    print(f"    impl_style breakdown:\n{df['fifo_impl_style'].value_counts().to_string()}")

    out_path = os.path.join(OUTPUT_DIR, "exhaustive_fifo_depths.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[✓] Salvo: {out_path}")
    print(f"    {df.shape[0]} linhas × {df.shape[1]} colunas")
    print("\nPróximo passo: execute train_fifo_depth.py")


if __name__ == "__main__":
    main()
