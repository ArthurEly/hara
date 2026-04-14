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
ALLOWED_DATASETS = ["MNIST_1W1A", "SAT6_T2W2"]

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

            # 5. Volume do Tensor (A partir da entrada da cabeça da cadeia)
            in_tensor = chain[0].input[0]
            out_tensor = chain[-1].output[0]
            
            try:
                shape = model.get_tensor_shape(in_tensor)
                tensor_volume = int(np.prod(shape))
                dtype = model.get_tensor_datatype(in_tensor)
                bits = extract_bitwidth(str(dtype))
            except:
                tensor_volume = 1
                bits = 8

            # 6. O VERDADEIRO PRODUTOR
            true_producer = get_node_by_output_tensor(model, in_tensor)
            p_name = true_producer.name if true_producer else "Input_Node"
            p_op = true_producer.op_type if true_producer else "Input"
            
            p_folding = get_node_folding_cfg(p_name, folding_cfg)
            p_pe = p_folding.get("PE", 1)
            p_cycles = extract_cycles_from_json(p_name, cycles_cfg)

            # 7. O VERDADEIRO CONSUMIDOR
            true_consumer = get_node_by_input_tensor(model, out_tensor)
            c_name = true_consumer.name if true_consumer else "Output_Node"
            c_op = true_consumer.op_type if true_consumer else "Output"
            
            c_folding = get_node_folding_cfg(c_name, folding_cfg)
            c_simd = c_folding.get("SIMD", 1)
            c_cycles = extract_cycles_from_json(c_name, cycles_cfg)

            # --- Atributos Adicionais ---
            p_throughput = p_pe / max(1, p_cycles)
            c_throughput = c_simd / max(1, c_cycles)
            p_transfers = tensor_volume / max(1, p_pe)
            c_transfers = tensor_volume / max(1, c_simd)

            # Monta a linha
            row = {
                "session": session_name,
                "run_name": run_name,
                "logical_fifo_name": base_fifo_name,
                "chain_length": len(chain),
                "dataType_bits": bits,
                "tensor_volume": tensor_volume,
                "ram_style": ram_style,
                "impl_style": impl_style,
                
                "produtor_node": p_name,
                "produtor_op": p_op,
                "produtor_PE": p_pe,
                "produtor_cycles": p_cycles,
                "p_throughput": p_throughput,
                "p_transfers": p_transfers,
                
                "consumidor_node": c_name,
                "consumidor_op": c_op,
                "consumidor_SIMD": c_simd,
                "consumidor_cycles": c_cycles,
                "c_throughput": c_throughput,
                "c_transfers": c_transfers,
                
                "parallelism_mismatch": 1 if p_pe != c_simd else 0,
                "real_depth": total_depth # O Nosso Ground Truth Aglutinado!
            }
            all_data.append(row)

    if not all_data:
        print("[!] Nenhum dado relacional extraído.")
        return
        
    df = pd.DataFrame(all_data)
    
    # Acúmulo Teórico (Backpressure)
    df["cycle_ratio"] = df["produtor_cycles"] / df.apply(lambda r: max(1, r["consumidor_cycles"]), axis=1)
    df["theoretical_accumulation"] = df.apply(
        lambda row: row["tensor_volume"] * (1.0 - row["cycle_ratio"]) if row["cycle_ratio"] < 1.0 else 0.0, 
        axis=1
    )
    df["theoretical_fifo_depth"] = df.apply(
        lambda row: row["theoretical_accumulation"] / max(1, row["produtor_PE"]),
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