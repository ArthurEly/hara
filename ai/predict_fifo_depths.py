"""
predict_fifo_depths.py — v3 (cycles-aware, DWC look-through, comp_* features)

Três fixes em relação à versão anterior:
  1. cycles_cfg: usa estimate_layer_cycles.json (ciclos reais) em vez de MH×MW/PE
  2. DWC look-through: atravessa StreamingDataWidthConverter e Thresholding para
     encontrar o nó computacional real (igual ao extract_fifo_relationships.py)
  3. comp_* features: calcula todos os 12 features computacionais que o modelo
     espera mas a versão anterior deixava zerados
"""

import numpy as np
import re
import warnings
from onnx import helper

# Nós "ponte" que não contribuem para o ciclo computacional
_BRIDGE_OPS = frozenset({
    "StreamingDataWidthConverter", "StreamingDataWidthConverter_rtl",
    "StreamingDataWidthConverter_hls",
    "Thresholding_rtl", "Thresholding_hls",
})


# ---------------------------------------------------------------------------
# Helpers de baixo nível
# ---------------------------------------------------------------------------

def _is_bridge(node) -> bool:
    if node is None:
        return False
    return any(b in node.op_type for b in ("StreamingDataWidthConverter", "Thresholding", "StreamingFIFO"))


def _traverse_to_compute(node, direction: str, prod_map: dict, cons_map: dict, max_hops: int = 6):
    """Atravessa nós-ponte (DWC/Thresholding) para encontrar o nó computacional real.
    direction='backward' = procura o produtor real; 'forward' = consumidor real."""
    cursor = node
    for _ in range(max_hops):
        if cursor is None or not _is_bridge(cursor):
            return cursor
        if direction == "backward":
            cursor = prod_map.get(cursor.input[0]) if cursor.input else None
        else:
            cursor = cons_map.get(cursor.output[0]) if cursor.output else None
    return cursor


def get_folding_attr(name: str, folding_cfg: dict, attr_name: str, default=1):
    if name in folding_cfg:
        return folding_cfg[name].get(attr_name, default)
    match = re.search(r'^(.*)_(\d+)$', name)
    if match:
        base, idx = match.groups()
        for backend in ("_rtl_", "_hls_", "_vivado_"):
            alt = f"{base}{backend}{idx}"
            if alt in folding_cfg:
                return folding_cfg[alt].get(attr_name, default)
    return default


def _extract_bits(dtype_str) -> int:
    if not dtype_str:
        return 8
    s = str(dtype_str).upper()
    if any(x in s for x in ("BINARY", "BIPOLAR", "INT1", "UINT1")):
        return 1
    nums = re.findall(r'\d+', s)
    return int(nums[0]) if nums else 8


def _onnx_attr(node, name: str, default=0):
    """Extrai atributo ONNX tratando tanto escalares quanto listas ([3,3] etc.)."""
    if node is None:
        return default
    for a in node.attribute:
        if a.name == name:
            try:
                val = helper.get_attribute_value(a)
                if isinstance(val, (int, float)):
                    return int(val)
                vals = list(val)
                return int(vals[0]) if vals else default
            except Exception:
                return a.i if a.type == 2 else default
    return default


def _cycles_from_json(node, cycles_cfg: dict | None) -> int | None:
    """Consulta estimate_layer_cycles.json. Retorna None se não encontrado."""
    if not cycles_cfg or node is None:
        return None
    name = node.name
    if name in cycles_cfg:
        return max(1, int(cycles_cfg[name]))
    # Normalização de sufixo (ex: MVAU_0 → MVAU_hls_0)
    parts = name.split("_")
    if len(parts) >= 2:
        for suffix in ("_rtl", "_hls"):
            alt = f"{parts[0]}{suffix}_{parts[-1]}"
            if alt in cycles_cfg:
                return max(1, int(cycles_cfg[alt]))
    return None


def _node_cycles(node, folding_cfg: dict, cycles_cfg: dict | None) -> int:
    """Ciclos reais do JSON quando disponível, senão MH×MW / (PE×SIMD)."""
    real = _cycles_from_json(node, cycles_cfg)
    if real is not None:
        return real
    if node is None:
        return 1
    mh = _onnx_attr(node, "MH", 1)
    mw = _onnx_attr(node, "MW", 1)
    pe = get_folding_attr(node.name, folding_cfg, "PE", 1)
    simd = get_folding_attr(node.name, folding_cfg, "SIMD", 1)
    return max(1, (mh * mw) // max(1, pe * simd))


# ---------------------------------------------------------------------------
# Preditor principal
# ---------------------------------------------------------------------------

def predict_fifo_depths_xgb(model_or_path, folding_cfg: dict,
                              depth_model_obj, cycles_cfg: dict | None = None) -> dict:
    from qonnx.core.modelwrapper import ModelWrapper
    import numpy as np
    import warnings

    if isinstance(model_or_path, str):
        model = ModelWrapper(model_or_path)
    else:
        model = model_or_path

    prod_map: dict = {}
    cons_map: dict = {}
    for node in model.graph.node:
        for t in node.output:
            prod_map[t] = node
        for t in node.input:
            cons_map[t] = node

    fifo_nodes = [n for n in model.graph.node if "FIFO" in n.op_type.upper()]

    if hasattr(depth_model_obj, "obj_dict"):
        m_data = depth_model_obj.obj_dict
    elif isinstance(depth_model_obj, dict):
        m_data = depth_model_obj
    else:
        m_data = {"model": depth_model_obj.model, "feature_names": depth_model_obj.feature_names}

    has_v2 = "stage1_classifier" in m_data
    depths: dict = {}

    for node in fifo_nodes:
        try:
            # --- Nós imediatos ---
            p_node = prod_map.get(node.input[0])
            c_node = cons_map.get(node.output[0])
            p_name = p_node.name if p_node else "Input_Node"
            p_op   = p_node.op_type if p_node else "Input"
            c_name = c_node.name if c_node else "Output_Node"
            c_op   = c_node.op_type if c_node else "Output"

            # --- Tensor ---
            try:
                shape        = model.get_tensor_shape(node.input[0])
                tensor_volume = int(np.prod(shape))
                bits          = _extract_bits(str(model.get_tensor_datatype(node.input[0])))
                tensor_spatial = shape[1] * shape[2] if len(shape) > 2 else 1
                tensor_H = float(shape[1] if len(shape) > 1 else 1)
                tensor_W = float(shape[2] if len(shape) > 2 else 1)
                tensor_C = float(shape[3] if len(shape) > 3 else bits)
            except Exception:
                tensor_volume, bits, tensor_spatial = 1, 8, 1
                tensor_H = tensor_W = tensor_C = 1.0
                shape = [1, 1, 1, 1]

            # --- Features imediatas ---
            p_pe   = get_folding_attr(p_name, folding_cfg, "PE",   1)
            p_simd = get_folding_attr(p_name, folding_cfg, "SIMD", 1)
            c_pe   = get_folding_attr(c_name, folding_cfg, "PE",   1)
            c_simd = get_folding_attr(c_name, folding_cfg, "SIMD", 1)

            p_cycles = _node_cycles(p_node, folding_cfg, cycles_cfg)
            c_cycles = _node_cycles(c_node, folding_cfg, cycles_cfg)

            p_throughput = float(p_pe)   / max(1, p_cycles)
            c_throughput = float(c_simd) / max(1, c_cycles)
            cycle_ratio  = float(p_cycles) / max(1, c_cycles)
            speed_ratio  = p_throughput / max(1e-12, c_throughput)
            p_transfers  = float(tensor_volume) / max(1, p_pe)
            c_transfers  = float(tensor_volume) / max(1, c_simd)

            theo_accum = float(tensor_volume) * (1.0 - cycle_ratio) if cycle_ratio < 1.0 else 0.0
            theo_depth = theo_accum / max(1, p_pe)

            # --- Nós computacionais reais (atravessando ponte) ---
            cp_node = _traverse_to_compute(p_node, "backward", prod_map, cons_map)
            cc_node = _traverse_to_compute(c_node, "forward",  prod_map, cons_map)
            comp_p_name = cp_node.name if cp_node else p_name
            comp_p_op   = cp_node.op_type if cp_node else p_op
            comp_c_name = cc_node.name if cc_node else c_name
            comp_c_op   = cc_node.op_type if cc_node else c_op

            # --- Extração de PE/SIMD e Ciclos Computacionais ---
            comp_p_pe   = get_folding_attr(comp_p_name, folding_cfg, "PE",   1)
            comp_p_simd = get_folding_attr(comp_p_name, folding_cfg, "SIMD", 1)
            comp_c_pe   = get_folding_attr(comp_c_name, folding_cfg, "PE",   1)
            comp_c_simd = get_folding_attr(comp_c_name, folding_cfg, "SIMD", 1)

            comp_p_cycles = _node_cycles(cp_node, folding_cfg, cycles_cfg)
            comp_c_cycles = _node_cycles(cc_node, folding_cfg, cycles_cfg)

            comp_p_thr   = float(comp_p_pe)   / max(1, comp_p_cycles)
            comp_c_thr   = float(comp_c_simd) / max(1, comp_c_cycles)
            
            # ---> DEFINIÇÃO CORRETA ANTES DO USO <---
            comp_cr      = float(comp_p_cycles) / max(1, comp_c_cycles)
            comp_pm      = 1 if comp_p_pe != comp_c_simd else 0

            # --- Acúmulo Computacional Original Limpo ---
            if comp_cr < 1.0:
                comp_theo_accum = float(tensor_volume) * (1.0 - comp_cr)
            else:
                comp_theo_accum = 0.0

            comp_theo_depth = comp_theo_accum / max(1, comp_p_pe)

            # --- Atributos ONNX espaciais (do nó computacional) ---
            p_fa = cp_node  
            c_fa = cc_node

            p_IFMDim     = _onnx_attr(p_fa, "IFMDim",      0)
            p_OFMDim     = _onnx_attr(p_fa, "OFMDim",      0)
            p_KernelDim  = _onnx_attr(p_fa, "ConvKernelDim", 0) or _onnx_attr(p_fa, "KernelDim", 0)
            p_Stride     = _onnx_attr(p_fa, "Stride",      1)
            p_IFMCh      = _onnx_attr(p_fa, "IFMChannels", 0)
            p_OFMCh      = _onnx_attr(p_fa, "OFMChannels", 0)
            c_IFMDim     = _onnx_attr(c_fa, "IFMDim",      0)
            c_OFMDim     = _onnx_attr(c_fa, "OFMDim",      0)
            c_KernelDim  = _onnx_attr(c_fa, "ConvKernelDim", 0) or _onnx_attr(c_fa, "KernelDim", 0)
            c_IFMCh      = _onnx_attr(c_fa, "IFMChannels", 0)
            
            win_vol      = p_KernelDim * p_KernelDim * p_IFMCh if p_KernelDim > 0 else 0

            # CIG startup features
            is_cig_prod    = "ConvolutionInputGenerator" in comp_p_op
            cig_warmup_rows = (p_KernelDim - 1) if (is_cig_prod and p_KernelDim > 0) else 0
            cig_startup_vol = p_IFMDim * cig_warmup_rows * p_IFMCh

            drain_time = float(tensor_volume) / max(1, comp_c_simd)
            fill_time  = float(tensor_volume) / max(1, comp_p_pe)

            # --- raw_data: todos os campos que o modelo espera ---
            raw_data = {
                "dataType_bits":  float(bits), "tensor_volume":  float(tensor_volume),
                "tensor_spatial": float(tensor_spatial), "tensor_C": tensor_C,
                "produtor_PE":      float(p_pe), "produtor_cycles":  float(p_cycles),
                "p_throughput":     float(p_throughput), "p_transfers": float(p_transfers),
                "consumidor_SIMD":  float(c_simd), "consumidor_cycles": float(c_cycles),
                "c_throughput":     float(c_throughput), "c_transfers": float(c_transfers),
                "cycle_ratio":      float(cycle_ratio), "speed_ratio": float(speed_ratio),
                "parallelism_mismatch": float(abs(p_pe - c_simd)),
                "theoretical_accumulation": float(theo_accum),
                "theoretical_fifo_depth":   float(theo_depth),
                "comp_produtor_PE":      float(comp_p_pe), "comp_produtor_SIMD": float(comp_p_simd),
                "comp_produtor_cycles":  float(comp_p_cycles), "comp_p_throughput": float(comp_p_thr),
                "comp_consumidor_PE":    float(comp_c_pe), "comp_consumidor_SIMD": float(comp_c_simd),
                "comp_consumidor_cycles": float(comp_c_cycles), "comp_c_throughput": float(comp_c_thr),
                "comp_cycle_ratio":      float(comp_cr), "comp_theo_accumulation": float(comp_theo_accum),
                "comp_theo_depth":        float(comp_theo_depth), "comp_parallelism_mismatch": float(comp_pm),
                "p_IFMDim": float(p_IFMDim), "p_OFMDim": float(p_OFMDim),
                "p_KernelDim": float(p_KernelDim), "p_Stride": float(p_Stride),
                "p_IFMChannels": float(p_IFMCh), "p_OFMChannels": float(p_OFMCh),
                "c_IFMDim": float(c_IFMDim), "c_OFMDim": float(c_OFMDim),
                "c_KernelDim": float(c_KernelDim), "c_IFMChannels": float(c_IFMCh),
                "window_volume": float(win_vol), "p_IFMDim_sq": float(p_IFMDim ** 2),
                "p_OFMDim_sq": float(p_OFMDim ** 2), "window_area": float(p_KernelDim ** 2),
                "cig_warmup_rows": float(cig_warmup_rows), "cig_startup_vol": float(cig_startup_vol),
                "drain_time": drain_time, "fill_time": fill_time,
                "channel_per_spatial": tensor_C / max(1.0, float(tensor_spatial)),
                "produtor_op": p_op, "consumidor_op": c_op,
                "op_pair": f"{p_op}→{c_op}", "comp_produtor_op": comp_p_op,
                "comp_consumidor_op": comp_c_op, "ram_style": "auto", "impl_style": "rtl",
            }

            # Log-transforms (sem chain_length — removido por ser data leakage)
            for col in ("cycle_ratio", "comp_cycle_ratio", "tensor_volume",
                        "theoretical_fifo_depth", "comp_theo_depth",
                        "consumidor_cycles", "comp_consumidor_cycles",
                        "cig_startup_vol", "drain_time"):
                raw_data[f"log_{col}"] = np.log1p(max(0.0, raw_data.get(col, 0)))

            raw_data["log_speed_ratio"]   = np.log1p(max(0.0, speed_ratio))
            raw_data["spatial_x_cycle"]   = np.log1p(float(tensor_spatial) * cycle_ratio)
            raw_data["is_spatial_large"]  = 1 if tensor_spatial > 256 else 0
            if "window_volume" in raw_data and "consumidor_cycles" in raw_data:
                raw_data["burst_x_cycles"] = np.log1p(float(win_vol) * float(c_cycles))

            # --- Alinhamento de features (one-hot imune a sufixos) ---
            def match_op(feat_name: str, prefix: str, current_op: str) -> bool:
                if not feat_name.startswith(prefix): return False
                return feat_name[len(prefix):].startswith(current_op)

            final_features = []
            for f in m_data["feature_names"]:
                if f in raw_data and not isinstance(raw_data[f], str):
                    final_features.append(float(raw_data[f]))
                elif match_op(f, "produtor_op_", p_op):
                    final_features.append(1.0)
                elif match_op(f, "consumidor_op_", c_op):
                    final_features.append(1.0)
                elif match_op(f, "comp_produtor_op_", comp_p_op):
                    final_features.append(1.0)
                elif match_op(f, "comp_consumidor_op_", comp_c_op):
                    final_features.append(1.0)
                elif f.startswith("op_pair_"):
                    pair_str = f[len("op_pair_"):]
                    if "→" in pair_str:
                        p_train, c_train = pair_str.split("→", 1)
                        if p_train.startswith(p_op) and c_train.startswith(c_op):
                            final_features.append(1.0)
                        else: final_features.append(0.0)
                    else: final_features.append(0.0)
                elif f.startswith("ram_style_") and f.endswith("auto"):
                    final_features.append(1.0)
                elif f.startswith("impl_style_") and f.endswith("rtl"):
                    final_features.append(1.0)
                else:
                    final_features.append(0.0)

            X_arr = np.array(final_features, dtype=np.float32).reshape(1, -1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if has_v2:
                    proba = m_data["stage1_classifier"].predict_proba(X_arr)[:, 1][0]
                    threshold = m_data.get("stage1_threshold", 0.55)
                    if proba >= threshold:
                        pred_real = 2.0
                    else:
                        pred_real = np.expm1(m_data["stage2_regressor"].predict(X_arr)[0])
                else:
                    pred_real = np.expm1(m_data["model"].predict(X_arr)[0])

            depths[node.name] = int(max(2, np.round(pred_real)))

        except Exception as e:
            import traceback
            print(f"\n[!] ERRO FATAL na FIFO {node.name}: {e}")
            traceback.print_exc()
            depths[node.name] = 2

    return depths