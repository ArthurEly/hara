import numpy as np
import re
from qonnx.core.modelwrapper import ModelWrapper

def get_folding_attr(name, folding_cfg, attr_name, default=1):
    """Busca um atributo no folding_cfg com suporte a nomes com/sem sufixo de backend."""
    # 1. Tenta o nome exato
    if name in folding_cfg:
        return folding_cfg[name].get(attr_name, default)
    
    # 2. Tenta buscar injetando sufixos comuns (_rtl_ ou _hls_) antes do índice final
    # Ex: StreamingFIFO_0 -> StreamingFIFO_rtl_0
    match = re.search(r'^(.*)_(\d+)$', name)
    if match:
        base, idx = match.groups()
        for backend in ["_rtl_", "_hls_", "_vivado_"]:
            alt_name = f"{base}{backend}{idx}"
            if alt_name in folding_cfg:
                return folding_cfg[alt_name].get(attr_name, default)
                
    return default

def predict_fifo_depths(model_or_path, folding_cfg):
    # Suporte para path ou ModelWrapper direto
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
            p_name = p_node.name if p_node else "INPUT_NODE"
            c_name = c_node.name if c_node else "OUTPUT_NODE"
            
            # --- EXTRAÇÃO DE THROUGHPUT REAL ---
            # Produtor: PE para MVAU, SIMD para outros (DWCs, Padding, etc)
            p_pe = get_folding_attr(p_name, folding_cfg, "PE", 1)
            p_simd = get_folding_attr(p_name, folding_cfg, "SIMD", 1)
            p_thr = p_pe if p_node and "MVAU" in p_node.op_type else p_simd
            
            # Consumidor: SIMD para MVAU, PE para Thresholding/Labels
            c_simd = get_folding_attr(c_name, folding_cfg, "SIMD", 1)
            c_pe = get_folding_attr(c_name, folding_cfg, "PE", 1)
            c_thr = c_simd if c_node and "MVAU" in c_node.op_type else c_pe
            
            # --- HEURÍSTICA DE DIMENSIONAMENTO ---
            if p_thr > c_thr:
                # Produtor mais rápido: Acúmulo de dados
                ratio = p_thr / c_thr
                depth = int(2 ** np.ceil(np.log2(ratio * 32)))
            elif p_name == "INPUT_NODE" or "FMPadding" in p_name or "ConvolutionInputGenerator" in p_name:
                # Camadas de entrada e janelamento (Padding/SWG) exigem buffers de linha/imagem
                # Heurística de segurança para SAT6/MNIST baseada nos seus JSONs
                depth = 512 
            else:
                # Pipeline balanceado ou consumidor mais rápido
                depth = 2
                
            depths[node.name] = max(2, min(depth, 8192))
            
        except Exception:
            depths[node.name] = 2
                
    return depths