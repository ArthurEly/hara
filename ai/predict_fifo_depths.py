"""
predict_fifo_depths.py

Prediz as profundidades (depths) de todos os StreamingFIFO_rtl_N que serão
inseridos pelo FINN no pipeline de um accelerator, ANTES da síntese HLS.

Entrada:
  - onnx_path  : caminho para step_generate_estimate_reports.onnx
  - folding    : dict de folding (run_N_optimized.json) com PE/SIMD por camada

Saída:
  dict { "StreamingFIFO_rtl_0": depth, "StreamingFIFO_rtl_1": depth, ... }
  pronto para ser injetado no folding JSON antes de rodar o HLS.

Algoritmo:
  1. Parseia ONNX → sequência de compute layers com atributos
  2. Para cada par adjacente: calcula outWidth_upstream e inWidth_downstream
     - Se diferentes → FINN insere um StreamingDataWidthConverter
  3. Reconstrói o pipeline completo com FIFOs entre tudo
  4. Para cada FIFO: monta feature vector (upstream/downstream)
  5. Stage 1 (classificador) → is_constrained (depth > 2)
  6. Stage 2 (regressor)     → depth exato (em espaço log)
"""

import os
import re
import json
import pickle
import numpy as np

import onnx
from onnx import helper

# =============================================================================
# CAMINHOS DOS MODELOS TREINADOS
# =============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_HERE, "retrieval", "results", "trained_models")

_CLS_PATH   = os.path.join(_MODELS_DIR, "StreamingFIFO_depth_classifier.pkl")
_REG_PATH   = os.path.join(_MODELS_DIR, "StreamingFIFO_depth_regressor.pkl")

# Lazy-loading dos modelos (carrega na primeira chamada)
_classifier = None
_regressor  = None
_feat_names = None   # lista de feature names na ordem do treino


def _load_models():
    global _classifier, _regressor, _feat_names
    if _classifier is not None:
        return

    if not os.path.exists(_CLS_PATH):
        raise FileNotFoundError(f"[FIFOPredictor] Classificador não encontrado: {_CLS_PATH}\n"
                                "  Execute train_fifo_depth.py primeiro.")
    if not os.path.exists(_REG_PATH):
        raise FileNotFoundError(f"[FIFOPredictor] Regressor não encontrado: {_REG_PATH}")

    with open(_CLS_PATH, "rb") as f:
        obj = pickle.load(f)
    _classifier = obj["model"]
    _feat_names = obj["feature_names"]

    with open(_REG_PATH, "rb") as f:
        obj = pickle.load(f)
    _regressor = obj["model"]

    print(f"[FIFOPredictor] Modelos carregados. Features: {len(_feat_names)}")


# =============================================================================
# PARSING ONNX
# =============================================================================

def _flatten(val):
    if isinstance(val, (list, tuple)):
        return int(np.prod(val)) if val else 0
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except Exception:
            return str(val)
    if isinstance(val, np.ndarray):
        return int(np.prod(val)) if val.size > 0 else 0
    return val


def _dtype_bits(s):
    s = str(s).upper()
    if "BINARY" in s or "BIPOLAR" in s:
        return 1
    m = re.search(r"(UINT|INT)(\d+)", s)
    return int(m.group(2)) if m else 0


def parse_onnx_nodes(onnx_path: str) -> list[dict]:
    """
    Retorna lista ordenada de nós compute do grafo ONNX, cada um com:
      name, op_type, attrs (MH, MW, PE, SIMD, inputBits, outputBits, ...)
    Exclui nós sem atributos relevantes (e.g. Flatten, Reshape).
    """
    model = onnx.load(onnx_path)
    nodes = []
    skip_types = {"Reshape", "Flatten", "Transpose", "Squeeze", "Unsqueeze",
                  "Gather", "Slice", "Cast", "Concat", "Split", "Identity"}

    for node in model.graph.node:
        if node.op_type in skip_types:
            continue
        attrs = {"op_type": node.op_type}
        for attr in node.attribute:
            v = _flatten(helper.get_attribute_value(attr))
            attrs[attr.name] = v
            if "DataType" in attr.name or "datatype" in attr.name.lower():
                attrs[attr.name + "_bits"] = _dtype_bits(str(v))

        # Só inclui nós que têm atributos HW relevantes
        hw_keys = {"MH", "MW", "IFMChannels", "OFMChannels", "numChannels",
                   "Labels", "inWidth", "outWidth", "SIMD", "PE"}
        if hw_keys & set(attrs.keys()):
            attrs["name"] = node.name or f"{node.op_type}_{len(nodes)}"
            nodes.append(attrs)

    return nodes


# =============================================================================
# RECONSTRUÇÃO DO PIPELINE (compute + SDC + FIFO)
# =============================================================================

def _get_out_width(layer_attrs: dict, folding: dict) -> int:
    """Calcula outWidth = PE × outputBitwidth (ou inWidth se disponível)."""
    name = layer_attrs.get("name", "")
    fold = folding.get(name, {})
    pe   = fold.get("PE", layer_attrs.get("PE", 1))

    # outWidth pode vir diretamente do ONNX
    if "outWidth" in layer_attrs and layer_attrs["outWidth"] > 0:
        return int(layer_attrs["outWidth"])

    # Senão estima: PE × outputBits
    out_bits = (layer_attrs.get("outputDataType_bits")
                or layer_attrs.get("outputBits")
                or layer_attrs.get("accDataType_bits")
                or 8)
    return max(1, int(pe) * int(out_bits))


def _get_in_width(layer_attrs: dict, folding: dict) -> int:
    """Calcula inWidth = SIMD × inputBitwidth (ou inWidth se disponível)."""
    name = layer_attrs.get("name", "")
    fold = folding.get(name, {})
    simd  = fold.get("SIMD", layer_attrs.get("SIMD", 1))

    if "inWidth" in layer_attrs and layer_attrs["inWidth"] > 0:
        return int(layer_attrs["inWidth"])

    in_bits = (layer_attrs.get("inputDataType_bits")
               or layer_attrs.get("inputBits")
               or 8)
    return max(1, int(simd) * int(in_bits))


def _base_type(name: str) -> tuple[str, str]:
    """Retorna (base_type, impl_type) de um nome como 'MVAU_hls_0'."""
    m = re.match(r"^(.+?)_(rtl|hls)_(\d+)$", name)
    if m:
        return m.group(1), m.group(2)
    return name, "unknown"


def build_pipeline(onnx_nodes: list[dict], folding: dict) -> list[dict]:
    """
    Constrói o pipeline completo incluindo FIFOs e SDCs sintéticos.

    Retorna lista de dicts com 'kind': 'compute' | 'sdc' | 'fifo'
    """
    pipeline = []
    fifo_idx = 0
    sdc_idx  = 0

    def _add_fifo():
        nonlocal fifo_idx
        pipeline.append({
            "kind": "fifo",
            "name": f"StreamingFIFO_rtl_{fifo_idx}",
        })
        fifo_idx += 1

    def _add_sdc(between_up, between_dn):
        nonlocal sdc_idx
        pipeline.append({
            "kind": "sdc",
            "name": f"StreamingDataWidthConverter_rtl_{sdc_idx}",
            "upstream": between_up,
            "downstream": between_dn,
        })
        sdc_idx += 1

    # Sempre começa com uma FIFO
    _add_fifo()

    for i, node in enumerate(onnx_nodes):
        pipeline.append({"kind": "compute", "name": node["name"], "attrs": node})
        _add_fifo()

        # Verifica se precisa de SDC entre este nó e o próximo
        if i < len(onnx_nodes) - 1:
            nxt = onnx_nodes[i + 1]
            out_w = _get_out_width(node, folding)
            in_w  = _get_in_width(nxt, folding)
            if out_w != in_w:
                _add_sdc(node["name"], nxt["name"])
                _add_fifo()

    return pipeline


# =============================================================================
# FEATURE VECTOR PARA UM FIFO
# =============================================================================

def _safe_scalar(d: dict, key, default=0):
    v = d.get(key, default)
    if isinstance(v, (list, tuple)):
        return v[0] if v else default
    return v if v is not None else default


# Categorias para one-hot (mesmas do treino)
_LAYER_TYPES = [
    "ConvolutionInputGenerator", "FMPadding", "LabelSelect", "MVAU",
    "StreamingDataWidthConverter", "StreamingFIFO", "Thresholding",
    "StreamingMaxPool",
]
_IMPL_TYPES = ["rtl", "hls", "vivado", "unknown"]
_OP_TYPES = [
    "MatrixVectorActivation", "ConvolutionInputGenerator_hls",
    "ConvolutionInputGenerator_rtl", "FMPadding_rtl", "LabelSelect_hls",
    "StreamingDataWidthConverter_rtl", "StreamingMaxPool_rtl",
    "Thresholding_rtl",
]
_IMPL_STYLES = ["rtl", "vivado", "auto"]


def _make_feature_row(fifo_pos: int,
                      upstream_info: dict | None,
                      downstream_info: dict | None,
                      folding: dict,
                      fifo_impl_style: str = "rtl") -> np.ndarray:
    """
    Cria um vetor de features para um FIFO alinhado com self._feat_names.
    Usa os mesmos campos gerados pelo get_exhaustive_fifo_depths.py.
    """
    _load_models()

    def _layer_feats(info: dict | None, prefix: str) -> dict:
        if info is None:
            return {f: 0 for f in _feat_names if f.startswith(prefix)}

        name = info.get("name", "")
        fold = folding.get(name, {})
        attrs = info.get("attrs", {})
        base, impl = _base_type(name)

        raw = {
            f"{prefix}layer_type":  base,
            f"{prefix}impl_type":   impl,
            f"{prefix}PE":          _safe_scalar(fold, "PE", 1),
            f"{prefix}SIMD":        _safe_scalar(fold, "SIMD", 1),
            f"{prefix}op_type":     attrs.get("op_type", base),
            f"{prefix}MH":          _safe_scalar(attrs, "MH", 0),
            f"{prefix}MW":          _safe_scalar(attrs, "MW", 0),
            f"{prefix}IFMChannels": _safe_scalar(attrs, "IFMChannels", 0),
            f"{prefix}OFMChannels": _safe_scalar(attrs, "OFMChannels", 0),
            f"{prefix}numChannels": _safe_scalar(attrs, "numChannels", 0),
            f"{prefix}Labels":      _safe_scalar(attrs, "Labels", 0),
            f"{prefix}inWidth":     _safe_scalar(attrs, "inWidth", 0),
            f"{prefix}outWidth":    _safe_scalar(attrs, "outWidth", 0),
            f"{prefix}IFMDim":      _safe_scalar(attrs, "IFMDim", 0),
            f"{prefix}OFMDim":      _safe_scalar(attrs, "OFMDim", 0),
            f"{prefix}KernelDim":   _safe_scalar(attrs, "KernelDim", 0),
            f"{prefix}Stride":      _safe_scalar(attrs, "Stride", 1),
            f"{prefix}inputBits":   attrs.get("inputDataType_bits", 0) or 0,
            f"{prefix}outputBits":  attrs.get("outputDataType_bits", 0) or 0,
            f"{prefix}weightBits":  attrs.get("weightDataType_bits", 0) or 0,
            f"{prefix}accBits":     attrs.get("accDataType_bits", 0) or 0,
        }
        return raw

    row_dict = {
        "fifo_pos":           fifo_pos,
        "is_baseline":        0,
        "run_number":         0,
        "up_out_fifo_depths": 0,
    }
    row_dict.update(_layer_feats(upstream_info, "up_"))
    row_dict.update(_layer_feats(downstream_info, "dn_"))

    # One-hot encode categorical fields to match training feature names
    cat_cols = {
        "up_layer_type": _LAYER_TYPES,
        "up_impl_type":  _IMPL_TYPES,
        "up_op_type":    _OP_TYPES,
        "dn_layer_type": _LAYER_TYPES,
        "dn_impl_type":  _IMPL_TYPES,
        "dn_op_type":    _OP_TYPES,
        "fifo_impl_style": _IMPL_STYLES,
    }

    row_dict["fifo_impl_style"] = fifo_impl_style

    for col, categories in cat_cols.items():
        val = row_dict.pop(col, None)
        for cat in categories:
            key = f"{col}_{cat}"
            row_dict[key] = 1 if val == cat else 0

    # Alinha com _feat_names
    vec = []
    for feat in _feat_names:
        vec.append(float(row_dict.get(feat, 0) or 0))

    return np.array(vec, dtype=np.float32)


# =============================================================================
# PREDIÇÃO PRINCIPAL
# =============================================================================

def predict_fifo_depths(onnx_path: str, folding: dict,
                        verbose: bool = False) -> dict:
    """
    Prediz a depth de cada StreamingFIFO que será inserido pelo FINN.

    Args:
        onnx_path : caminho para step_generate_estimate_reports.onnx
        folding   : dict de folding {layer_name: {PE: N, SIMD: M, ...}}
        verbose   : imprime detalhes de cada FIFO

    Returns:
        dict { "StreamingFIFO_rtl_0": 2, "StreamingFIFO_rtl_1": 868, ... }
    """
    _load_models()

    # 1. Parseia ONNX
    onnx_nodes = parse_onnx_nodes(onnx_path)
    if verbose:
        print(f"[FIFOPredictor] {len(onnx_nodes)} compute nodes no ONNX")

    # 2. Reconstrói pipeline (compute + SDC + FIFO)
    pipeline = build_pipeline(onnx_nodes, folding)

    # Indexa compute/sdc entries por posição para lookup de vizinhos
    layer_positions = {entry["name"]: i
                       for i, entry in enumerate(pipeline)
                       if entry["kind"] in ("compute", "sdc")}

    # 3. Para cada FIFO, identifica upstream e downstream
    results = {}
    fifo_entries = [(i, e) for i, e in enumerate(pipeline) if e["kind"] == "fifo"]

    for pos_idx, (pip_pos, fifo_entry) in enumerate(fifo_entries):
        fifo_name = fifo_entry["name"]

        # Upstream: entrada imediatamente antes do FIFO no pipeline
        upstream_info = None
        for j in range(pip_pos - 1, -1, -1):
            if pipeline[j]["kind"] in ("compute", "sdc"):
                upstream_info = pipeline[j]
                break

        # Downstream: saída imediatamente após o FIFO
        downstream_info = None
        for j in range(pip_pos + 1, len(pipeline)):
            if pipeline[j]["kind"] in ("compute", "sdc"):
                downstream_info = pipeline[j]
                break

        # 4. Feature vector
        X = _make_feature_row(
            fifo_pos=pos_idx,
            upstream_info=upstream_info,
            downstream_info=downstream_info,
            folding=folding,
        ).reshape(1, -1)

        # 5. Stage 1: é constrained?
        is_constrained = bool(_classifier.predict(X)[0])

        # 6. Stage 2: depth
        if is_constrained:
            depth_log = _regressor.predict(X)[0]
            depth = max(2, int(round(np.expm1(depth_log))))
        else:
            depth = 2

        results[fifo_name] = depth

        if verbose:
            up_name = upstream_info["name"] if upstream_info else "—"
            dn_name = downstream_info["name"] if downstream_info else "—"
            print(f"  {fifo_name}: [{up_name}] → [{dn_name}] "
                  f"constrained={is_constrained} depth={depth}")

    if verbose:
        print(f"[FIFOPredictor] {len(results)} FIFOs preditos. "
              f"Constrained: {sum(d > 2 for d in results.values())}")

    return results


def inject_fifo_depths(folding: dict, fifo_depths: dict) -> dict:
    """
    Injeta as depths previstas no folding config.
    Cria entradas StreamingFIFO_rtl_N se não existirem.
    Retorna folding atualizado (não modifica o original).
    """
    import copy
    new_folding = copy.deepcopy(folding)

    for fifo_name, depth in fifo_depths.items():
        if fifo_name not in new_folding:
            new_folding[fifo_name] = {}
        entry = new_folding[fifo_name]
        entry["depth"] = depth
        entry.setdefault("ram_style", "auto")
        entry.setdefault("impl_style", "rtl" if depth < 256 else "vivado")

    return new_folding


# =============================================================================
# CLI / TEST
# =============================================================================

def main():
    import argparse, sys

    parser = argparse.ArgumentParser(description="Prediz FIFO depths de um folding config")
    parser.add_argument("--onnx",    required=True, help="step_generate_estimate_reports.onnx")
    parser.add_argument("--folding", required=True, help="run_N_optimized.json ou final_hw_config.json")
    parser.add_argument("--out",     default=None,  help="Salvar resultado como JSON")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.folding) as f:
        folding = json.load(f)

    depths = predict_fifo_depths(args.onnx, folding, verbose=args.verbose)

    print("\n=== FIFO Depths Previstas ===")
    for k, v in depths.items():
        marker = "🔴" if v > 2 else "⚪"
        print(f"  {marker} {k}: {v}")
    print(f"\nTotal: {len(depths)} FIFOs | "
          f"Constrained: {sum(d>2 for d in depths.values())}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(depths, f, indent=2)
        print(f"\n[✓] Resultado salvo em: {args.out}")


# Esse bloco protege o script contra execuções acidentais no import!
if __name__ == "__main__":
    main()