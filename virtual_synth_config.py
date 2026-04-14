"""
virtual_synth_config.py

Gera o arquivo final_hw_config.json através da expansão virtual do grafo ONNX
e predição de profundidade de FIFOs usando Machine Learning (HARA V3).
Garante 100% de alinhamento com as features MINIMALISTAS do treino.
"""

import os
import re
import json
import pickle
import numpy as np

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import ApplyConfig

from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO

# Caminho dos Modelos XGBoost
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai", "retrieval", "results", "trained_models")

HW_ATTRS = [
    "PE", "SIMD", "parallel_window", "ram_style", "depth", 
    "impl_style", "resType", "mem_mode", "runtime_writeable_weights", 
    "inFIFODepths", "outFIFODepths"
]

def safe_get_attr(node, attr, default=1):
    """Extrai atributos numéricos de hardware multiplicando listas (ex: [32, 32] -> 1024)."""
    if node is None: return default
    try:
        inst = getCustomOp(node)
        v = inst.get_nodeattr(attr)
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = [x for x in v if x is not None]
            return int(np.prod(arr)) if len(arr) > 0 else default
        return float(v) if v is not None else default
    except:
        for a in node.attribute:
            if a.name == attr:
                from onnx import helper
                v = helper.get_attribute_value(a)
                if isinstance(v, (list, tuple, np.ndarray)):
                    arr = [x for x in v if x is not None]
                    return int(np.prod(arr)) if len(arr) > 0 else default
                if isinstance(v, bytes):
                    try: return float(v.decode('utf-8'))
                    except: pass
                return float(v)
        return default

def extract_minimal_features(node, prefix):
    """Monta o dicionário de features minimalista para bater com o CSV."""
    if node is None:
        return {f"{prefix}{k}": 1 for k in ["PE","SIMD","IFMDim","OFMDim","IFMCh","OFMCh","MH","MW"]} | {f"{prefix}type": "none"}

    name = node.name
    m = re.match(r"^(.+?)_(rtl|hls|vivado)_(\d+)$", name)
    base_type = m.group(1) if m else name
    
    # Normalização dos nomes para bater com o pandas get_dummies do treino
    if "StreamingDataWidthConverter" in base_type: base_type = "StreamingDataWidthConverter"
    elif "StreamingFIFO" in base_type: base_type = "StreamingFIFO"
    elif "MVAU" in base_type: base_type = "MVAU"
    elif "ConvolutionInputGenerator" in base_type: base_type = "ConvolutionInputGenerator"
    elif "FMPadding" in base_type: base_type = "FMPadding"
    elif "StreamingMaxPool" in base_type: base_type = "StreamingMaxPool"
    elif "Thresholding" in base_type: base_type = "Thresholding"

    return {
        f"{prefix}type": base_type,
        f"{prefix}PE": safe_get_attr(node, "PE", 1),
        f"{prefix}SIMD": safe_get_attr(node, "SIMD", 1),
        f"{prefix}IFMDim": safe_get_attr(node, "IFMDim", 1),
        f"{prefix}OFMDim": safe_get_attr(node, "OFMDim", 1),
        f"{prefix}IFMCh": safe_get_attr(node, "IFMChannels", 1),
        f"{prefix}OFMCh": safe_get_attr(node, "OFMChannels", 1),
        f"{prefix}MH": safe_get_attr(node, "MH", 1),
        f"{prefix}MW": safe_get_attr(node, "MW", 1)
    }

def generate_virtual_hw_config(onnx_path, folding_json_path, output_json_path):
    print("=" * 60)
    print("🚀 HARA V3 - SÍNTESE VIRTUAL (100% ALIGNMENT MINIMALISTA)")
    print("=" * 60)

    # 1. Carrega Modelos XGBoost
    try:
        with open(os.path.join(MODELS_DIR, "StreamingFIFO_depth_classifier.pkl"), "rb") as f:
            clf_data = pickle.load(f)
            xgb_clf = clf_data["model"]
            feat_names = clf_data["feature_names"]
        with open(os.path.join(MODELS_DIR, "StreamingFIFO_depth_regressor.pkl"), "rb") as f:
            xgb_reg = pickle.load(f)["model"]
        print("[✓] Modelos XGBoost carregados com sucesso.")
    except Exception as e:
        print(f"❌ Erro ao carregar XGBoost: {e}")
        return

    # 2. Pipeline FINN
    model = ModelWrapper(onnx_path)
    print(f"[1/4] Aplicando folding: {os.path.basename(folding_json_path)}")
    model = model.transform(ApplyConfig(folding_json_path))
    print("[2/4] Expandindo grafo FINN original (DWCs e FIFOs)...")
    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(GiveUniqueNodeNames())
    
    # Ajuste de nomenclatura (o FINN adiciona apenas ao nome, não ao op_type)
    for node in model.graph.node:
        if node.op_type in ["StreamingFIFO", "StreamingDataWidthConverter"]:
            if "_rtl" not in node.name:
                node.name = node.name.replace("StreamingFIFO", "StreamingFIFO_rtl")
                node.name = node.name.replace("StreamingDataWidthConverter", "StreamingDataWidthConverter_rtl")

    # 3. Predição Nativa Sincronizada
    print("\n[3/4] Extraindo topologia e prevendo Depths...")
    
    # === A CORREÇÃO ESTÁ AQUI ===
    # op_type é 'StreamingFIFO', o nome é que leva o '_rtl'
    fifo_nodes = model.get_nodes_by_op_type("StreamingFIFO")
    
    print(f"  ➜ ENCONTRADAS {len(fifo_nodes)} FIFOs PARA PREDIÇÃO! Iniciando XGBoost...\n")

    for pos_idx, fifo_node in enumerate(fifo_nodes):
        inst = getCustomOp(fifo_node)
        up_node = model.find_producer(fifo_node.input[0])
        dn_node = model.find_consumer(fifo_node.output[0])

        up_feats = extract_minimal_features(up_node, "up_")
        dn_feats = extract_minimal_features(dn_node, "dn_")
        
        # Inicia vetor com zeros
        row_dict = {f: 0.0 for f in feat_names}

        # Numéricos Diretos
        for k in ["PE", "SIMD", "IFMDim", "OFMDim", "IFMCh", "OFMCh", "MH", "MW"]:
            if f"up_{k}" in row_dict: row_dict[f"up_{k}"] = float(up_feats[f"up_{k}"])
            if f"dn_{k}" in row_dict: row_dict[f"dn_{k}"] = float(dn_feats[f"dn_{k}"])

        # Throughput
        up_c = (up_feats["up_MH"] * up_feats["up_MW"]) / (up_feats["up_PE"] * up_feats["up_SIMD"])
        dn_c = (dn_feats["dn_MH"] * dn_feats["dn_MW"]) / (dn_feats["dn_PE"] * dn_feats["dn_SIMD"])
        
        if "up_cycles" in row_dict: row_dict["up_cycles"] = float(up_c)
        if "dn_cycles" in row_dict: row_dict["dn_cycles"] = float(dn_c)
        if "th_ratio" in row_dict:  row_dict["th_ratio"] = float(up_c / (dn_c + 1e-6))

        # One-Hot Encoding Mapeado
        up_type_col = f"up_type_{up_feats['up_type']}"
        dn_type_col = f"dn_type_{dn_feats['dn_type']}"
        
        if up_type_col in row_dict: row_dict[up_type_col] = 1.0
        if dn_type_col in row_dict: row_dict[dn_type_col] = 1.0

        # Monta o array exato
        vec = np.array([[row_dict[f] for f in feat_names]], dtype=np.float32)
        
        # PREDIÇÃO
        is_constrained = xgb_clf.predict(vec)[0]
        if is_constrained == 0:
            depth = 2
        else:
            depth = int(round(np.expm1(xgb_reg.predict(vec)[0])))
            depth = max(2, depth)
            
        print(f"  🔍 {fifo_node.name} | UP: {up_feats['up_type']:<28} | DN: {dn_feats['dn_type']:<28} | XGBoost = {depth}")
            
        # Aplica no Grafo
        try:
            inst.set_nodeattr("depth", depth)
            if depth > 256:
                inst.set_nodeattr("impl_style", "vivado")
                inst.set_nodeattr("ram_style", "block")
            else:
                inst.set_nodeattr("impl_style", "rtl")
                inst.set_nodeattr("ram_style", "distributed")
        except: pass

    # 4. Extração JSON
    print("\n[4/4] Gerando final_hw_config.json...")
    final_config = {"Defaults": {}}
    for node in model.graph.node:
        try:
            inst = getCustomOp(node)
            node_cfg = {}
            for attr in HW_ATTRS:
                try:
                    val = inst.get_nodeattr(attr)
                    if isinstance(val, (np.ndarray, list)): val = [int(x) for x in val]
                    node_cfg[attr] = val
                except: pass
            if node_cfg:
                final_config[node.name] = node_cfg
        except: pass

    with open(output_json_path, "w") as f:
        json.dump(final_config, f, indent=2)

    print(f"\n✅ Concluído! Arquivo gerado: {output_json_path}")

    # Comparação Ground Truth
    gt_path = os.path.join(os.path.dirname(os.path.dirname(onnx_path)), "final_hw_config.json")
    if os.path.exists(gt_path):
        print("\n" + "-" * 50)
        print("🕵️  DISCREPÂNCIA (HARA vs FINN RTL)")
        print("-" * 50)
        with open(gt_path, "r") as f: gt_config = json.load(f)
        common_nodes = (set(final_config.keys()) & set(gt_config.keys())) - {"Defaults"}
        mismatches = 0
        for node_name in sorted(list(common_nodes)):
            g_attrs, r_attrs = final_config[node_name], gt_config[node_name]
            if "depth" in g_attrs and "depth" in r_attrs:
                if int(g_attrs["depth"]) != int(r_attrs["depth"]):
                    print(f"  [{node_name}] Depth: Gerado={g_attrs['depth']} | Real={r_attrs['depth']}")
                    mismatches += 1
        print(f"  Divergências Restantes de Depth: {mismatches}")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4: sys.exit(1)
    generate_virtual_hw_config(sys.argv[1], sys.argv[2], sys.argv[3])