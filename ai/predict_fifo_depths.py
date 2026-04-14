import numpy as np
import re
import warnings

def get_folding_attr(name, folding_cfg, attr_name, default=1):
    if name in folding_cfg:
        return folding_cfg[name].get(attr_name, default)
    
    match = re.search(r'^(.*)_(\d+)$', name)
    if match:
        base, idx = match.groups()
        for backend in ["_rtl_", "_hls_", "_vivado_"]:
            alt_name = f"{base}{backend}{idx}"
            if alt_name in folding_cfg:
                return folding_cfg[alt_name].get(attr_name, default)
                
    return default

def _extract_bits(dtype_str):
    if not dtype_str: return 8 
    s = str(dtype_str).upper()
    if any(x in s for x in ["BINARY", "BIPOLAR", "INT1", "UINT1"]): return 1
    nums = re.findall(r'\d+', s)
    if nums: return int(nums[0])
    return 8

def predict_fifo_depths_xgb(model_or_path, folding_cfg, depth_model_obj):
    # Support for String or ModelWrapper
    from qonnx.core.modelwrapper import ModelWrapper
    if isinstance(model_or_path, str):
        model = ModelWrapper(model_or_path)
    else:
        model = model_or_path
        
    depths = {}
    prod_map = {}
    cons_map = {}
    
    for node in model.graph.node:
        for out_t in node.output: prod_map[out_t] = node
        for in_t in node.input: cons_map[in_t] = node

    fifo_nodes = [n for n in model.graph.node if "FIFO" in n.op_type.upper()]

    for node in fifo_nodes:
        try:
            p_node = prod_map.get(node.input[0])
            c_node = cons_map.get(node.output[0])
            
            p_name = p_node.name if p_node else "Input_Node"
            p_op = p_node.op_type if p_node else "Input"
            c_name = c_node.name if c_node else "Output_Node"
            c_op = c_node.op_type if c_node else "Output"
            
            # --- Extract tensor info ---
            try:
                shape = model.get_tensor_shape(node.input[0])
                tensor_volume = int(np.prod(shape))
                dtype = model.get_tensor_datatype(node.input[0])
                bits = _extract_bits(str(dtype))
            except:
                tensor_volume = 1
                bits = 8

            p_pe = get_folding_attr(p_name, folding_cfg, "PE", 1)
            c_simd = get_folding_attr(c_name, folding_cfg, "SIMD", 1)
            
            # Estimate cycles based on heuristics (fallback from true cycles config)
            # If producer is MVAU, cycles ~ (MH*MW)/(PE*SIMD)
            p_mh = get_folding_attr(p_name, folding_cfg, "MH", p_pe)
            p_mw = get_folding_attr(p_name, folding_cfg, "MW", p_pe)
            p_simd = get_folding_attr(p_name, folding_cfg, "SIMD", 1)
            p_cycles = max(1, (p_mh * p_mw) // (p_pe * p_simd))

            c_mh = get_folding_attr(c_name, folding_cfg, "MH", c_simd)
            c_mw = get_folding_attr(c_name, folding_cfg, "MW", c_simd)
            c_pe = get_folding_attr(c_name, folding_cfg, "PE", 1)
            c_cycles = max(1, (c_mh * c_mw) // (c_pe * c_simd))

            p_throughput = p_pe / max(1, p_cycles)
            c_throughput = c_simd / max(1, c_cycles)
            p_transfers = tensor_volume / max(1, p_pe)
            c_transfers = tensor_volume / max(1, c_simd)
            
            parallelism_mismatch = 1 if p_pe != c_simd else 0
            
            cycle_ratio = p_cycles / max(1, c_cycles)
            theoretical_accumulation = tensor_volume * (1.0 - cycle_ratio) if cycle_ratio < 1.0 else 0.0
            theoretical_fifo_depth = theoretical_accumulation / max(1, p_pe)
            
            # Assemble raw features dictionary for XGBoost Model
            raw_data = {
                "dataType_bits": bits,
                "tensor_volume": np.log1p(tensor_volume),
                "produtor_PE": p_pe,
                "produtor_cycles": np.log1p(p_cycles),
                "p_throughput": p_throughput,
                "p_transfers": np.log1p(p_transfers),
                "consumidor_SIMD": c_simd,
                "consumidor_cycles": np.log1p(c_cycles),
                "c_throughput": c_throughput,
                "c_transfers": np.log1p(c_transfers),
                "parallelism_mismatch": parallelism_mismatch,
                "cycle_ratio": cycle_ratio,
                "theoretical_accumulation": np.log1p(theoretical_accumulation),
                "theoretical_fifo_depth": np.log1p(theoretical_fifo_depth)
            }
            
            # Categorical OHE (produtor_op, consumidor_op, ram_style, impl_style)
            # Default ram_style and impl_style since they might not be known yet
            ram_style = "auto"
            impl_style = "rtl"
            
            for cat, val in [
                ("produtor_op", p_op), 
                ("consumidor_op", c_op), 
                ("ram_style", ram_style), 
                ("impl_style", impl_style)
            ]:
                if val:
                    clean = val.lower()
                    raw_data[f"{cat}_{clean}"] = 1
            
            # Form feature array
            X_input = []
            for f in depth_model_obj.feature_names:
                X_input.append(float(raw_data.get(f, 0.0)))
                
            X_arr = np.array(X_input, dtype=np.float32).reshape(1, -1)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                log_pred = depth_model_obj.model.predict(X_arr)[0]
                
            pred_real = int(np.round(np.expm1(log_pred)))
            depths[node.name] = max(2, min(pred_real, 65536))
            
        except Exception as e:
            # print(f"Erro FIFO XGB: {e}")
            depths[node.name] = 2
                
    return depths
