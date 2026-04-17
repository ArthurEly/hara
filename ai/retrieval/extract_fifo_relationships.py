"""
extract_fifo_relationships.py
Versão 2.0 - Agglomerated Cascades
Analisa a vizinhança de cada StreamingFIFO no grafo ONNX do FINN.
Aglutina FIFOs em cascata (ex: _0_0, _0_1) numa única FIFO lógica,
extraindo métricas do Verdadeiro Produtor e do Verdadeiro Consumidor.
"""

import os
import json
import glob
import re
import numpy as np
import pandas as pd
import onnx
from onnx import helper

# Bibliotecas Core do FINN/QONNX
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
BASE_BUILDS_DIR = "/home/arthurely/Desktop/finn_chi2p/hara/exhaustive_hw_builds"

# Foco nas precisões ultra baixas
ALLOWED_DATASETS = ["MNIST_1W1A", "SAT6_T2W2", "CIFAR10_1W1A", "CIFAR10_2W2A"]

def extract_bitwidth(x_str):
    if not x_str: return 8
    x_str = str(x_str).upper()
    if any(k in x_str for k in ["BINARY", "BIPOLAR", "B'BINARY"]): return 1
    match = re.search(r'(UINT|INT)(\d+)', x_str)
    return int(match.group(2)) if match else 8

def get_node_by_output_tensor(wrapper, tensor_name):
    for node in wrapper.graph.node:
        if tensor_name in node.output:
            return node
    return None

def get_node_by_input_tensor(wrapper, tensor_name):
    consumers = []
    for node in wrapper.graph.node:
        if tensor_name in node.input:
            consumers.append(node)
    return consumers[0] if consumers else None

def get_node_folding_cfg(base_name, folding_cfg):
    if base_name in folding_cfg: return folding_cfg[base_name]
    parts = base_name.split("_")
    if len(parts) >= 2:
        for suffix in ["_rtl", "_hls", "_vivado"]:
            test_name = f"{parts[0]}{suffix}_{parts[1]}"
            if test_name in folding_cfg:
                return folding_cfg[test_name]
    return {}

def extract_cycles_from_json(node_name, cycles_cfg):
    if node_name in cycles_cfg: return cycles_cfg[node_name]
    parts = node_name.split("_")
    if len(parts) >= 2:
        for suffix in ["_rtl", "_hls"]:
            test_name = f"{parts[0]}{suffix}_{parts[1]}"
            if test_name in cycles_cfg:
                return cycles_cfg[test_name]
    return 1

def main():
    print(f"[{'='*75}]")
    print("[HARA] Extração de Relacionamentos (Aglutinação de FIFOs Cascata)")
    print(f"[{'='*75}]")

    search_pattern = os.path.join(BASE_BUILDS_DIR, "*", "run*", "intermediate_models", "step_create_stitched_ip.onnx")
    onnx_files = glob.glob(search_pattern)
    
    if not onnx_files:
        print(f"[!] Nenhum ONNX encontrado em {BASE_BUILDS_DIR}")
        return

    all_data = []
    
    for onnx_path in onnx_files:
        run_dir = os.path.dirname(os.path.dirname(onnx_path))
        session_name = os.path.basename(os.path.dirname(run_dir))
        run_name = os.path.basename(run_dir)
        
        if not any(ds in session_name for ds in ALLOWED_DATASETS):
            continue
            
        print(f"  -> Processando: {session_name} / {run_name}")
        
        config_path = os.path.join(run_dir, "final_hw_config.json")
        cycles_path = os.path.join(run_dir, "report", "estimate_layer_cycles.json")
        
        if not os.path.exists(config_path) or not os.path.exists(cycles_path):
            continue
            
        with open(config_path, "r") as f: folding_cfg = json.load(f)
        with open(cycles_path, "r") as f: cycles_cfg = json.load(f)

        try:
            model = ModelWrapper(onnx_path)
        except Exception as e:
            print(f"    [!] Erro ao abrir ONNX: {e}")
            continue

        processed_fifos = set()

        for node in model.graph.node:
            if not node.op_type.startswith("StreamingFIFO"):
                continue
            
            if node.name in processed_fifos:
                continue

            # 1. Verifica se é a "Cabeça" da cascata
            producer = get_node_by_output_tensor(model, node.input[0])
            if producer and producer.op_type.startswith("StreamingFIFO"):
                # Se o produtor for uma FIFO, não é a cabeça. Ignorar e deixar a cabeça encontrá-lo.
                continue

            # 2. Navegar e aglutinar a cascata
            chain = [node]
            processed_fifos.add(node.name)
            current_node = node

            while True:
                consumer = get_node_by_input_tensor(model, current_node.output[0])
                if consumer and consumer.op_type.startswith("StreamingFIFO"):
                    chain.append(consumer)
                    processed_fifos.add(consumer.name)
                    current_node = consumer
                else:
                    break

            # 3. Nome base da FIFO (remove sufixos _0, _1 finais para limpar)
            base_fifo_name = re.sub(r'_\d+$', '', chain[0].name)
            
            # 4. Somar Depths Totais
            total_depth = 0
            for f_node in chain:
                attrs = {attr.name: helper.get_attribute_value(attr) for attr in f_node.attribute}
                total_depth += attrs.get("depth", 2)

            # Extrair o tipo de RAM da primeira FIFO (normalmente o maior bloco)
            first_attrs = {attr.name: helper.get_attribute_value(attr) for attr in chain[0].attribute}
            ram_style = first_attrs.get("ram_style", b"auto")
            if isinstance(ram_style, bytes): ram_style = ram_style.decode('utf-8')
            impl_style = first_attrs.get("impl_style", b"rtl")
            if isinstance(impl_style, bytes): impl_style = impl_style.decode('utf-8')

            # 5. Volume do Tensor + Dimensões Espaciais
            in_tensor = chain[0].input[0]
            out_tensor = chain[-1].output[0]
            
            try:
                shape = model.get_tensor_shape(in_tensor)
                tensor_volume = int(np.prod(shape))
                dtype = model.get_tensor_datatype(in_tensor)
                bits = extract_bitwidth(str(dtype))
            except:
                shape = [1]
                tensor_volume = 1
                bits = 8
            
            # Extrair dimensões espaciais H, W, C do tensor (layout: [N, H, W, C])
            if len(shape) == 4:
                tensor_H = shape[1]
                tensor_W = shape[2]
                tensor_C = shape[3]
            elif len(shape) == 2:
                tensor_H = 1
                tensor_W = 1
                tensor_C = shape[1]
            else:
                tensor_H = 1
                tensor_W = 1
                tensor_C = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            
            tensor_spatial = tensor_H * tensor_W  # área espacial do feature map

            # 6. O PRODUTOR IMEDIATO
            true_producer = get_node_by_output_tensor(model, in_tensor)
            p_name = true_producer.name if true_producer else "Input_Node"
            p_op = true_producer.op_type if true_producer else "Input"
            
            p_folding = get_node_folding_cfg(p_name, folding_cfg)
            p_pe = p_folding.get("PE", 1)
            p_simd = p_folding.get("SIMD", 1)
            p_cycles = extract_cycles_from_json(p_name, cycles_cfg)

            # 7. O CONSUMIDOR IMEDIATO
            true_consumer = get_node_by_input_tensor(model, out_tensor)
            c_name = true_consumer.name if true_consumer else "Output_Node"
            c_op = true_consumer.op_type if true_consumer else "Output"
            
            c_folding = get_node_folding_cfg(c_name, folding_cfg)
            c_simd = c_folding.get("SIMD", 1)
            c_pe = c_folding.get("PE", 1)
            c_cycles = extract_cycles_from_json(c_name, cycles_cfg)

            # 8. OLHAR ATRAVÉS de nós-ponte para o VERDADEIRO nó computacional
            def is_bridge(op_type):
                if not op_type: return False
                return any(b in op_type for b in ("StreamingDataWidthConverter", "Thresholding", "StreamingFIFO"))
            
            # Produtor computacional real (olha para trás)
            comp_p_name, comp_p_op, comp_p_cycles, comp_p_pe, comp_p_simd = p_name, p_op, p_cycles, p_pe, p_simd
            if true_producer and is_bridge(p_op):
                cursor = true_producer
                for _ in range(5):  
                    prev = get_node_by_output_tensor(model, cursor.input[0])
                    if prev is None:
                        comp_p_name, comp_p_op = "Input_Node", "Input"
                        comp_p_cycles, comp_p_pe, comp_p_simd = 1, 1, 1
                        break
                    if is_bridge(prev.op_type):
                        cursor = prev
                        continue
                    comp_p_name = prev.name
                    comp_p_op = prev.op_type
                    comp_p_folding = get_node_folding_cfg(prev.name, folding_cfg)
                    comp_p_pe = comp_p_folding.get("PE", 1)
                    comp_p_simd = comp_p_folding.get("SIMD", 1)
                    comp_p_cycles = extract_cycles_from_json(prev.name, cycles_cfg)
                    break
            
            # Consumidor computacional real (olha para frente)
            comp_c_name, comp_c_op, comp_c_cycles, comp_c_pe, comp_c_simd = c_name, c_op, c_cycles, c_pe, c_simd
            if true_consumer and is_bridge(c_op):
                cursor = true_consumer
                for _ in range(5):
                    nxt = get_node_by_input_tensor(model, cursor.output[0])
                    if nxt is None:
                        comp_c_name, comp_c_op = "Output_Node", "Output"
                        comp_c_cycles, comp_c_pe, comp_c_simd = 1, 1, 1
                        break
                    if is_bridge(nxt.op_type):
                        cursor = nxt
                        continue
                    comp_c_name = nxt.name
                    comp_c_op = nxt.op_type
                    comp_c_folding = get_node_folding_cfg(nxt.name, folding_cfg)
                    comp_c_pe = comp_c_folding.get("PE", 1)
                    comp_c_simd = comp_c_folding.get("SIMD", 1)
                    comp_c_cycles = extract_cycles_from_json(nxt.name, cycles_cfg)
                    break

            # 9. Extrair atributos ONNX essenciais
            def get_onnx_attr(node_obj, attr_name, default=0):
                if node_obj is None:
                    return default
                for attr in node_obj.attribute:
                    if attr.name == attr_name:
                        val = helper.get_attribute_value(attr)
                        if isinstance(val, (int, float)):
                            return int(val)
                        try:
                            vals = list(val)
                            return int(vals[0]) if vals else default
                        except (TypeError, IndexError):
                            return default
                return default
            
            p_node_for_attrs = true_producer
            if true_producer and is_bridge(p_op):
                for n in model.graph.node:
                    if n.name == comp_p_name:
                        p_node_for_attrs = n
                        break
            
            p_KernelDim = get_onnx_attr(p_node_for_attrs, "ConvKernelDim", 0)
            if p_KernelDim == 0:
                p_KernelDim = get_onnx_attr(p_node_for_attrs, "KernelDim", 0)
            p_IFMChannels = get_onnx_attr(p_node_for_attrs, "IFMChannels", 0)
            p_IFMDim  = get_onnx_attr(p_node_for_attrs, "IFMDim", 0)
            p_OFMDim  = get_onnx_attr(p_node_for_attrs, "OFMDim", 0)
            p_OFMChannels = get_onnx_attr(p_node_for_attrs, "OFMChannels", 0)
            p_Stride  = get_onnx_attr(p_node_for_attrs, "Stride", 1)

            c_node_for_attrs = true_consumer
            if true_consumer and is_bridge(c_op):
                for n in model.graph.node:
                    if n.name == comp_c_name:
                        c_node_for_attrs = n
                        break
            c_IFMDim     = get_onnx_attr(c_node_for_attrs, "IFMDim", 0)
            c_OFMDim     = get_onnx_attr(c_node_for_attrs, "OFMDim", 0)
            c_IFMChannels = get_onnx_attr(c_node_for_attrs, "IFMChannels", 0)
            c_KernelDim  = get_onnx_attr(c_node_for_attrs, "ConvKernelDim", 0)
            if c_KernelDim == 0:
                c_KernelDim = get_onnx_attr(c_node_for_attrs, "KernelDim", 0)

            window_volume = p_KernelDim * p_KernelDim * p_IFMChannels if p_KernelDim > 0 else 0

            # CIG startup features: warmup buffer before first output
            is_cig_prod = "ConvolutionInputGenerator" in comp_p_op
            cig_warmup_rows = (p_KernelDim - 1) if (is_cig_prod and p_KernelDim > 0) else 0
            cig_startup_vol = p_IFMDim * cig_warmup_rows * p_IFMChannels  # elements buffered during CIG startup

            # Acumulação teórica (backpressure when consumer is slower)
            comp_p_throughput = comp_p_pe / max(1, comp_p_cycles)
            comp_c_throughput = comp_c_simd / max(1, comp_c_cycles)
            comp_cycle_ratio = comp_p_cycles / max(1, comp_c_cycles)
            comp_parallelism_mismatch = 1 if comp_p_pe != comp_c_simd else 0

            if comp_cycle_ratio < 1.0:
                comp_theo_accum = float(tensor_volume * (1.0 - comp_cycle_ratio))
            else:
                comp_theo_accum = 0.0
            comp_theo_depth = comp_theo_accum / max(1, comp_p_pe)

            p_throughput = p_pe / max(1, p_cycles)
            c_throughput = c_simd / max(1, c_cycles)
            drain_time   = float(tensor_volume) / max(1, comp_c_simd)
            fill_time    = float(tensor_volume) / max(1, comp_p_pe)
            p_transfers  = float(tensor_volume) / max(1, p_pe)
            c_transfers  = float(tensor_volume) / max(1, c_simd)
            channel_per_spatial = float(tensor_C) / max(1, tensor_spatial)

            row = {
                "session": session_name,
                "run_name": run_name,
                "logical_fifo_name": base_fifo_name,
                # NOTE: chain_length removed — it's determined BY depth, not a predictor of it
                "dataType_bits": bits,
                "tensor_volume": tensor_volume,
                "tensor_H": tensor_H, "tensor_W": tensor_W,
                "tensor_C": tensor_C, "tensor_spatial": tensor_spatial,
                "ram_style": ram_style, "impl_style": impl_style,

                "produtor_op": p_op,
                "produtor_PE": p_pe, "produtor_SIMD": p_simd,
                "produtor_cycles": p_cycles,
                "p_throughput": p_throughput, "p_transfers": p_transfers,

                "consumidor_op": c_op,
                "consumidor_SIMD": c_simd, "consumidor_PE": c_pe,
                "consumidor_cycles": c_cycles,
                "c_throughput": c_throughput, "c_transfers": c_transfers,

                "cycle_ratio": p_cycles / max(1, c_cycles),
                "parallelism_mismatch": 1 if p_pe != c_simd else 0,

                "comp_produtor_op": comp_p_op,
                "comp_produtor_PE": comp_p_pe, "comp_produtor_SIMD": comp_p_simd,
                "comp_produtor_cycles": comp_p_cycles, "comp_p_throughput": comp_p_throughput,

                "comp_consumidor_op": comp_c_op,
                "comp_consumidor_PE": comp_c_pe, "comp_consumidor_SIMD": comp_c_simd,
                "comp_consumidor_cycles": comp_c_cycles, "comp_c_throughput": comp_c_throughput,

                "comp_cycle_ratio": comp_cycle_ratio,
                "comp_theo_accumulation": comp_theo_accum,
                "comp_theo_depth": comp_theo_depth,
                "comp_parallelism_mismatch": comp_parallelism_mismatch,

                # Spatial attrs from compute nodes
                "p_IFMDim": p_IFMDim, "p_OFMDim": p_OFMDim,
                "p_KernelDim": p_KernelDim, "p_Stride": p_Stride,
                "p_IFMChannels": p_IFMChannels, "p_OFMChannels": p_OFMChannels,
                "c_IFMDim": c_IFMDim, "c_OFMDim": c_OFMDim,
                "c_KernelDim": c_KernelDim, "c_IFMChannels": c_IFMChannels,
                "window_volume": window_volume,
                "p_IFMDim_sq": p_IFMDim ** 2, "p_OFMDim_sq": p_OFMDim ** 2,
                "window_area": p_KernelDim ** 2,

                # CIG startup (key for predicting large FIFOs before/after CIG)
                "cig_warmup_rows": cig_warmup_rows,
                "cig_startup_vol": cig_startup_vol,

                # Derived throughput features
                "drain_time": drain_time,
                "fill_time": fill_time,
                "channel_per_spatial": channel_per_spatial,

                "real_depth": total_depth,
            }
            all_data.append(row)

    if not all_data:
        print("[!] Nenhum dado relacional extraído.")
        return
        
    df = pd.DataFrame(all_data)

    df["theoretical_accumulation"] = df.apply(
        lambda r: r["tensor_volume"] * (1.0 - r["cycle_ratio"]) if r["cycle_ratio"] < 1.0 else 0.0,
        axis=1
    )
    df["theoretical_fifo_depth"] = df.apply(
        lambda r: r["theoretical_accumulation"] / max(1, r["produtor_PE"]),
        axis=1
    )
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "fifo_depth")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fifo_backpressure_dataset.csv")
    
    df.to_csv(out_path, index=False)
    print(f"\n[✓] Sucesso! Dataset relacional (Aglutinado) salvo em:")
    print(f"    {out_path}")
    print(f"    (Total de FIFOs lógicas mapeadas: {len(df)})")

if __name__ == "__main__":
    main()