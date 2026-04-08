"""
multi_module_learner.py - Versão HARA v3 (Produção)

Destaques desta versão:
1. Expansão Virtual: Insere FIFOs e DWCs simulando o pipeline real do FINN.
2. Tensor Intelligence: Recupera bitwidths e inWidth diretamente do ModelWrapper.
3. Name Sync: Sincroniza nomes entre o preditor de depth e o somador de área.
4. Debug Mode: Logs específicos para validar a binarização (1-bit) no hardware.
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd

import onnx
from onnx import helper

# Bibliotecas Core do FINN/QONNX
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO

# Integração com o Preditor de Profundidade
try:
    from predict_fifo_depths import predict_fifo_depths
    _FIFO_PREDICTOR_AVAILABLE = True
except ImportError:
    _FIFO_PREDICTOR_AVAILABLE = False
    print("[MultiModuleLearner] AVISO: predict_fifo_depths não encontrado.")

# =============================================================================
# MAPEAMENTOS E CONFIGURAÇÕES
# =============================================================================

MODULE_MODEL_MAP = {
    "MVAU": "MVAU",
    "MatrixVectorActivation": "MVAU",
    "ConvolutionInputGenerator": "ConvolutionInputGenerator",
    "FMPadding": "FMPadding",
    "LabelSelect": "LabelSelect",
    "Thresholding": "Thresholding",
    "StreamingDataWidthConverter": "StreamingDataWidthConverter",
    "StreamingFIFO": "StreamingFIFO",
}

TARGET_COLS = ["Total LUT", "Total FFs", "BRAM (36k eq.)", "DSP Blocks"]

# =============================================================================
# MODELO ESPECIALISTA (Wrapper XGBoost)
# =============================================================================

class _PerModuleModel:
    def __init__(self, pkl_path: str):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        self.model = obj["model"]
        self.feature_names = obj.get("feature_names", [])
        self.target_cols = obj.get("target_cols", TARGET_COLS)

    def predict(self, X: np.ndarray) -> dict[str, float]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self.model.predict(X.reshape(1, -1))[0]
        return {col: max(0.0, float(v)) for col, v in zip(self.target_cols, raw)}

# =============================================================================
# CLASSE PRINCIPAL: MULTI-MODULE LEARNER
# =============================================================================

class MultiModuleLearner:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self._models: dict[str, _PerModuleModel] = {}
        self._onnx_cache: dict[str, list] = {}
        self.current_wrapper = None # Mantém o grafo expandido para consulta de tensores
        self._load_all_models()

    def _extract_bitwidth(self, x_str):
        """Traduz tipos FINN para inteiros de bitwidth."""
        if not x_str: return 0
        x_str = str(x_str).upper()
        if any(k in x_str for k in ["BINARY", "BIPOLAR", "B'BINARY"]):
            return 1
        match = re.search(r'(UINT|INT)(\d+)', x_str)
        return int(match.group(2)) if match else 0

    def _load_all_models(self):
        """Carrega os pkls da pasta retrieval/results/trained_models/."""
        if not os.path.isdir(self.models_dir): return
        for fname in os.listdir(self.models_dir):
            if not fname.endswith(".pkl") or "depth" in fname: continue
            m = re.match(r"(?:exhaustive_)?(.+?)_model\.pkl$", fname)
            if not m: continue
            key = m.group(1)
            try:
                self._models[key] = _PerModuleModel(os.path.join(self.models_dir, fname))
                print(f"[MultiModuleLearner] ✓ Modelo {key} carregado.")
            except Exception as e:
                print(f"[MultiModuleLearner] ! Erro ao carregar {key}: {e}")

    def is_loaded(self) -> bool:
        return len(self._models) > 0

    # ------------------------------------------------------------------
    # GESTÃO DE TOPOLOGIA (Pre-HLS)
    # ------------------------------------------------------------------

    def _expand_topology(self, onnx_path: str):
        """Expande o grafo inserindo FIFOs e DWCs virtuais."""
        print(f"\n[HARA] Analisando pipeline: {os.path.basename(onnx_path)}")
        model = ModelWrapper(onnx_path)
        
        # Transformações lógicas do FINN
        model = model.transform(InsertDWC())
        model = model.transform(InsertFIFO(create_shallow_fifos=True))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        
        self.current_wrapper = model # Salva para consulta de bitwidths via tensores
        
        nodes_list = []
        for node in model.graph.node:
            attrs = {"op_type": node.op_type, "name": node.name}
            for attr in node.attribute:
                val = helper.get_attribute_value(attr)
                if isinstance(val, bytes):
                    try: val = val.decode("utf-8")
                    except: val = str(val)
                elif isinstance(val, (list, tuple, np.ndarray)):
                    val = int(np.prod(val)) if len(val) > 0 else 0
                attrs[attr.name] = val
            
            # Metadata interna para busca de largura de banda
            attrs["_in_tensor"] = node.input[0]
            nodes_list.append(attrs)
            
        return nodes_list

    # ------------------------------------------------------------------
    # PREDITOR DE ÁREA POR NÓ
    # ------------------------------------------------------------------

    # Adicionar como método da classe MultiModuleLearner
    def _classify_fifo_memory(self, node_attrs: dict, folding_cfg: dict, depth: int) -> str:
        """
        Determina deterministicamente se a FIFO usará BRAM ou LUT/SRL.
        Deve ser chamado ANTES do modelo XGBoost para injetar 'is_bram' como feature.
        """
        layer_name = node_attrs.get("name", "")
        impl_style  = str(node_attrs.get("impl_style", "rtl")).lower()
        ram_style   = str(folding_cfg.get(layer_name, {}).get("ram_style",
                        node_attrs.get("ram_style", "auto"))).lower()

        # Calcula largura real (antes do log-scaling)
        input_tensor = node_attrs.get("_in_tensor")
        bits = 0
        if input_tensor and self.current_wrapper:
            dtype = self.current_wrapper.get_tensor_datatype(input_tensor)
            bits = self._extract_bitwidth(str(dtype))
        if bits == 0:
            bits = 8
        inst_folding = folding_cfg.get(layer_name, {})
        parallel  = inst_folding.get("PE", inst_folding.get("SIMD", 1))
        in_width  = bits * parallel
        bit_cap   = in_width * depth

        BRAM_THRESHOLD = 512  # bits — empírico para RAMB18E1

        # FIFOs Xilinx XPM (impl_style="vivado"): Vivado decide via MEMORY_PRIMITIVE=0
        if impl_style == "vivado":
            return "BRAM" if bit_cap > BRAM_THRESHOLD else "LUT"

        # FIFOs FINN RTL (impl_style="rtl"): controlado pelo ram_style do nó
        if ram_style == "block":
            return "BRAM"
        if ram_style == "distributed":
            return "LUT"
        # ram_style="auto": FINN aplica o mesmo threshold
        return "BRAM" if bit_cap > BRAM_THRESHOLD else "LUT"

    def _predict_layer(self, module_key: str, node_attrs: dict, 
                       folding_cfg: dict, depth: int) -> dict | None:
        layer_name = node_attrs.get("name", "")
        op_type = node_attrs.get("op_type", "").lower()
        
        # --- O ROTEADOR DE ESPECIALISTAS (Mixture of Experts) ---
        if "StreamingFIFO" in module_key:
            # Decide o universo usando a função determinística que criamos
            mem_type = self._classify_fifo_memory(node_attrs, folding_cfg, depth)
            
            # Muda o module_key dinamicamente para o modelo correto!
            module_key = f"StreamingFIFO_{mem_type}"
            
        model_obj = self._models.get(module_key)
        if not model_obj: return None

        raw_data = dict(node_attrs)
        inst_folding = folding_cfg.get(layer_name, {})
        raw_data.update(inst_folding)
        
        raw_data["depth"] = depth
        raw_data["isRTL"] = 1 if "rtl" in layer_name.lower() else 0
        raw_data["isHLS"] = 1 if "hls" in layer_name.lower() else 0

        # --- RECUPERAÇÃO DE BITS E LARGURA ---
        input_tensor = node_attrs.get("_in_tensor")
        bits = 0
        if input_tensor and self.current_wrapper:
            dtype = self.current_wrapper.get_tensor_datatype(input_tensor)
            bits = self._extract_bitwidth(str(dtype))
        if bits == 0: bits = 8
        
        parallel = inst_folding.get("PE", inst_folding.get("SIMD", 1))
        in_width = bits * parallel
        raw_data["inWidth"] = in_width
        
        # --- CLASSIFICAÇÃO BRAM vs LUT (determinística) ---
        mem_type = self._classify_fifo_memory(node_attrs, folding_cfg, depth)
        raw_data["is_bram"] = 1 if mem_type == "BRAM" else 0

        if "StreamingFIFO" in module_key:
            print(f"[DEBUG FIFO] {layer_name} | mem={mem_type} | depth={depth} | inW={in_width} | cap={in_width*depth}")
            # Precisamos recriar exatamente as features logarítmicas do treino
            bit_cap = in_width * depth
            raw_data["bit_capacity"] = np.log1p(bit_cap)
            raw_data["inWidth"] = np.log1p(in_width)
            raw_data["depth"] = np.log1p(depth)
            
            if in_width >= 1:
                print(f"[DEBUG FIFO] {layer_name} | log(Width)={raw_data['inWidth']:.2f} | log(Cap)={raw_data['bit_capacity']:.2f}")

        # One-Hot Encoding
        for cat in ["ram_style", "resType", "mem_mode", "binaryXnorMode", "impl_style"]:
            if cat in raw_data:
                val = str(raw_data[cat])
                clean = ''.join(e for e in val if e.isalnum()).capitalize()
                raw_data[f"is{cat[0].upper() + cat[1:]}{clean}"] = 1

        # Alinhamento e Predição
        X_input = [float(raw_data.get(f, 0.0)) for f in model_obj.feature_names]
        X_arr = np.array(X_input, dtype=np.float32).reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # A predição bruta está na escala Logarítmica
            real_pred = model_obj.model.predict(X_arr)[0]

        # --- REVERSÃO DO LOG (Scale Back to Hardware) ---
        # np.expm1(x) calcula e^x - 1, revertendo o log(1+x) do treino
        #real_pred = np.expm1(log_pred)

        return {col: max(0.0, float(v)) for col, v in zip(model_obj.target_cols, real_pred)}

    # ------------------------------------------------------------------
    # PIPELINE DE PREDIÇÃO END-TO-END
    # ------------------------------------------------------------------

    def predict(self, onnx_path: str, foldings: list[dict]) -> list[dict]:
        if not self.is_loaded(): return [{}] * len(foldings)
        
        # Expande o grafo (InsertFIFO/DWC)
        nodes = self._onnx_cache.get(onnx_path) or self._expand_topology(onnx_path)
        self._onnx_cache[onnx_path] = nodes
        
        results = []
        for folding in foldings:
            # Predição 2-stage de profundidade
            raw_depths = predict_fifo_depths(onnx_path, folding) if _FIFO_PREDICTOR_AVAILABLE else {}
            # Normaliza nomes: StreamingFIFO_rtl_0 -> StreamingFIFO_0
            fifo_depths = {k.replace("_rtl", ""): v for k, v in raw_depths.items()}

            total = {col: 0.0 for col in TARGET_COLS}
            details = {}

            for node in nodes:
                op = node.get("op_type", "")
                name = node.get("name", "")
                
                module_key = None
                for k, mk in MODULE_MODEL_MAP.items():
                    if k in op:
                        module_key = mk
                        break

                if not module_key: continue

                # Define depth (ML para FIFOs, default 2 para o resto)
                d = fifo_depths.get(name, 2) if "StreamingFIFO" in op else 2
                
                pred = self._predict_layer(module_key, node, folding, d)
                if pred:
                    details[name] = pred
                    for col in TARGET_COLS:
                        total[col] += pred.get(col, 0.0)

            results.append({
                "Total LUTs":  int(round(total["Total LUT"])),
                "FFs":         int(round(total["Total FFs"])),
                "BRAM (36k)":  round(total["BRAM (36k eq.)"], 1),
                "DSP Blocks":  int(round(total["DSP Blocks"])),
                "_details":    details
            })
        return results

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys, json, pprint
    if len(sys.argv) < 2:
        print("Uso: python multi_module_learner.py <run_dir>")
        sys.exit(1)

    run_dir = sys.argv[1]
    onnx_file = os.path.join(run_dir, "intermediate_models", "step_generate_estimate_reports.onnx")
    hw_config = os.path.join(run_dir, "final_hw_config.json")
    
    with open(hw_config, "r") as f: cfg = json.load(f)
    m_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieval", "results", "trained_models")
    
    learner = MultiModuleLearner(m_dir)
    p = learner.predict(onnx_file, [cfg])
    
    print("\n=== ESTIMATIVA FINAL HARA (Área Total Estimada) ===")
    pprint.pprint(p[0], sort_dicts=False)