"""
extract_split_fifo_area.py
Extrai a área exata e os atributos de CADA FATIA individual das StreamingFIFOs
(ex: StreamingFIFO_rtl_0_0, _0_1) cruzando o ONNX com o Report do Vivado.
"""

import os
import re
import json
import glob
import numpy as np
import pandas as pd
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
BASE_BUILDS_DIR = "/home/arthurely/Desktop/finn_chi2p/hara/exhaustive_hw_builds"

def extract_bitwidth(x_str):
    if not x_str: return 8
    x_str = str(x_str).upper()
    if any(k in x_str for k in ["BINARY", "BIPOLAR", "B'BINARY"]): return 1
    match = re.search(r'(UINT|INT)(\d+)', x_str)
    return int(match.group(2)) if match else 8

def parse_vivado_hierarchical_utilization(rpt_path):
    """
    Lê o relatório hierárquico do Vivado e extrai a área de cada módulo.
    Retorna um dicionário: { "nome_da_instancia": { "Logic LUTs": X, ... } }
    """
    area_dict = {}
    with open(rpt_path, 'r') as f:
        lines = f.readlines()

    # O Vivado Partition Util RPT tem este padrão aproximado:
    # | Instance | Module | Total LUTs | Logic LUTs | LUTRAMs | SRLs | FFs | RAMB36 | RAMB18 | DSPs |
    # Adaptado para capturar linhas com instâncias (ex: StreamingFIFO_rtl_0_0)
    for line in lines:
        if not line.startswith("|") or "Instance" in line or "Total LUTs" in line:
            continue
            
        parts = [p.strip() for p in line.split("|") if p.strip() != ""]
        if len(parts) >= 10:
            instance_name = parts[0]
            # O Vivado as vezes coloca "inst" ou recuos. Vamos limpar:
            clean_inst_name = instance_name.split()[-1] 
            
            try:
                area_dict[clean_inst_name] = {
                    "Total LUTs": float(parts[2]),
                    "Logic LUTs": float(parts[3]),
                    "LUTRAMs": float(parts[4]),
                    "SRLs": float(parts[5]),
                    "Total FFs": float(parts[6]),
                    "RAMB36": float(parts[7]),
                    "RAMB18": float(parts[8]),
                    "DSP Blocks": float(parts[9] if len(parts)>9 else 0)
                }
            except ValueError:
                continue
                
    return area_dict

def main():
    print(f"[{'='*75}]")
    print("[HARA] Extração de Área por Fatia de FIFO (Split FIFOs)")
    print(f"[{'='*75}]")

    # Precisamos do ONNX final (que tem as FIFOs fatiadas) e do Report do Vivado
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
        
        rpt_path = os.path.join(run_dir, "stitched_ip", "finn_design_partition_util.rpt")
        if not os.path.exists(rpt_path):
            continue
            
        print(f"  -> Processando: {session_name} / {run_name}")
        
        # 1. Obter a "Verdade Física" (Área gasta no Vivado)
        vivado_area = parse_vivado_hierarchical_utilization(rpt_path)
        
        # 2. Obter a "Verdade Lógica" (ONNX Atributos)
        try:
            model = ModelWrapper(onnx_path)
            model = model.transform(GiveUniqueNodeNames())
        except Exception as e:
            print(f"    [!] Erro ao abrir ONNX: {e}")
            continue

        for node in model.graph.node:
            if not node.op_type.startswith("StreamingFIFO"):
                continue
                
            node_name = node.name
            
            # --- Buscar Área no Vivado ---
            # O Vivado pode chamar 'StreamingFIFO_rtl_0_0' ou apenas 'inst' dentro dessa hierarquia.
            # Se não acharmos direto, procuramos substrings.
            area_info = vivado_area.get(node_name)
            if not area_info:
                # Fallback: procurar quem contém o nome
                for k, v in vivado_area.items():
                    if node_name in k:
                        area_info = v
                        break
            
            if not area_info:
                # Se o Vivado otimizou e removeu essa FIFO por completo (área 0)
                area_info = {
                    "Total LUTs": 0.0, "Logic LUTs": 0.0, "LUTRAMs": 0.0,
                    "SRLs": 0.0, "Total FFs": 0.0, "RAMB36": 0.0, "RAMB18": 0.0, "DSP Blocks": 0.0
                }

            # --- Atributos da FIFO ---
            attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
            depth = attrs.get("depth", 2)
            
            ram_style = attrs.get("ram_style", b"auto")
            if isinstance(ram_style, bytes): ram_style = ram_style.decode('utf-8')
            
            impl_style = attrs.get("impl_style", b"rtl")
            if isinstance(impl_style, bytes): impl_style = impl_style.decode('utf-8')

            # --- Largura de Entrada (inWidth) ---
            # Calculada via folded_shape * dataType_bits
            dataType = attrs.get("dataType", b"INT8")
            if isinstance(dataType, bytes): dataType = dataType.decode('utf-8')
            bits = extract_bitwidth(dataType)
            
            folded_shape = attrs.get("folded_shape", [1])
            if isinstance(folded_shape, (list, tuple, np.ndarray)) and len(folded_shape) > 0:
                simd = folded_shape[-1] 
            else:
                simd = 1
                
            inWidth = bits * simd

            # --- Identificar a qual Cascata Pertence ---
            # Ex: StreamingFIFO_rtl_5_2 -> base_fifo = StreamingFIFO_rtl_5, split_idx = 2
            match = re.match(r"(StreamingFIFO_\w+_\d+)_(\d+)$", node_name)
            if match:
                base_fifo = match.group(1)
                split_idx = int(match.group(2))
            else:
                base_fifo = node_name
                split_idx = 0

            # Monta a linha de dados
            row = {
                "session": session_name,
                "run_name": run_name,
                "node_name": node_name,
                "base_fifo": base_fifo,
                "split_idx": split_idx,
                "dataType_bits": bits,
                "simd": simd,
                "inWidth": inWidth,
                "depth": depth,
                "bit_capacity": inWidth * depth,
                "ram_style": ram_style,
                "impl_style": impl_style,
            }
            # Adiciona as colunas de área
            row.update(area_info)
            all_data.append(row)

    if not all_data:
        print("[!] Nenhum dado extraído.")
        return
        
    df = pd.DataFrame(all_data)
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "fifo_area")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "split_fifo_area_dataset.csv")
    
    df.to_csv(out_path, index=False)
    print(f"\n[✓] Sucesso! Dataset de Área Fatiada salvo em:")
    print(f"    {out_path}")
    print(f"    (Total de blocos físicos mapeados: {len(df)})")

if __name__ == "__main__":
    main()