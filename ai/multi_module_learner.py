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
    from predict_fifo_depths import predict_fifo_depths_xgb
    _FIFO_PREDICTOR_AVAILABLE = True
except ImportError:
    _FIFO_PREDICTOR_AVAILABLE = False
    print("[MultiModuleLearner] AVISO: predict_fifo_depths não encontrado.")

# =============================================================================
# MAPEAMENTOS E CONFIGURAÇÕES
# =============================================================================

MODULE_MODEL_MAP = {
    "MVAU": "MVAU",  # Chave base
    "ConvolutionInputGenerator": "ConvolutionInputGenerator",
    "FMPadding": "FMPadding",
    "LabelSelect": "LabelSelect",
    "Thresholding": "Thresholding",
    "StreamingDataWidthConverter": "StreamingDataWidthConverter",
    "StreamingFIFO": "StreamingFIFO",
}

TARGET_COLS = ["Total LUT", "Total FFs", "BRAM (36k eq.)", "DSP Blocks"]
SUPPORTED_TOPOLOGIES = [
    "MNIST_1W1A", 
    "MNIST_2W2A", 
    "SAT6_T2W2", 
    "SAT6_T2W4", 
    "SAT6_T2W8"
]
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
        self.fifo_classifier = None
        # Altere o nome do arquivo aqui também:
        clf_path = os.path.join(models_dir, "StreamingFIFO_Classifier.pkl")
        if os.path.exists(clf_path):
            with open(clf_path, "rb") as f:
                self.fifo_classifier = pickle.load(f)
            print("[MultiModuleLearner] ✓ Decision Tree Classifier de BRAM/LUT carregado.")

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
            if not fname.endswith(".pkl"): continue
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
        # Extrair a largura real (in_width)
        input_tensor = node_attrs.get("_in_tensor")
        bits = 8
        if input_tensor and self.current_wrapper:
            dtype = self.current_wrapper.get_tensor_datatype(input_tensor)
            bits = self._extract_bitwidth(str(dtype))
        
        layer_name = node_attrs.get("name", "")
        parallel = folding_cfg.get(layer_name, {}).get("PE", folding_cfg.get(layer_name, {}).get("SIMD", 1))
        in_width = bits * parallel
        
        # Usa a Árvore de Decisão se ela existir
        if self.fifo_classifier is not None:
            # O modelo espera um array 2D com [inWidth, depth] (valores reais, sem log!)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = self.fifo_classifier.predict([[in_width, depth]])[0]
            return "BRAM" if pred == 1 else "LUT"
        else:
            # Fallback para a regra de segurança se o modelo sumir
            return "BRAM" if (in_width * depth) >= 2048 else "LUT"

    def _predict_layer(self, module_key: str, node_attrs: dict, 
                       folding_cfg: dict, depth: int) -> dict | None:
        layer_name = node_attrs.get("name", "")
        op_type = node_attrs.get("op_type", "").lower()
        raw_data = dict(node_attrs)
        inst_folding = folding_cfg.get(layer_name, {})
        raw_data.update(inst_folding)
        
        # --- ROTEADOR MVAU (1-Bit vs Multi-Bit) ---
        if "MVAU" in module_key:            
            print(f"\n--- [DEBUG MVAU INPUT: {layer_name}] ---")
            print(f"  Especialista Alvo: {module_key}")
            print(f"  weightDataType: {raw_data.get('weightDataType')}")
            print(f"  weightDataType (bits): {raw_data.get('weightDataType (bits)')}")
            print(f"  mac_complexity: {raw_data.get('mac_complexity')}")
            print(f"  PE: {raw_data.get('PE')} | SIMD: {raw_data.get('SIMD')}")
            print("------------------------------------------\n")
        
        
        # --- O ROTEADOR DE ESPECIALISTAS (Mixture of Experts) ---
        if "StreamingFIFO" in module_key:
            module_key = "SplitFIFO_area"
            
        model_obj = self._models.get(module_key)
        if not model_obj: return None
        
        # --- LÓGICA ESPECIAL PARA FATIAS E AUTO-STYLE (SPLIT_FIFO) ---
        if module_key == "SplitFIFO_area":
            from predict_fifo_utils import finn_partition_fifo, prepare_fifo_features
            
            # RECUPERA O TENSOR (BITS e IN_WIDTH)
            input_tensor = node_attrs.get("_in_tensor")
            bits = 0
            if input_tensor and self.current_wrapper:
                dtype = self.current_wrapper.get_tensor_datatype(input_tensor)
                bits = self._extract_bitwidth(str(dtype))
            if bits == 0: bits = 8
            
            parallel = inst_folding.get("PE", inst_folding.get("SIMD", 1))
            in_width = bits * parallel
            
            r_style = node_attrs.get("ram_style", "auto")
            if isinstance(r_style, bytes): r_style = r_style.decode("utf-8")
            
            i_style = node_attrs.get("impl_style", "rtl")
            if isinstance(i_style, bytes): i_style = i_style.decode("utf-8")
            r_style = r_style.lower()
            
            slices = []
            
            # Config override do JSON
            matched_cfg = {}
            for k, v in folding_cfg.items():
                clean_k = k.replace("_rtl", "").replace("_hls", "")
                if clean_k == layer_name:
                    matched_cfg = v
                    break
                    
            user_r_style = matched_cfg.get("ram_style")
            if user_r_style is not None:
                r_style = user_r_style
                
            user_i_style = matched_cfg.get("impl_style")
            if user_i_style is not None:
                i_style = user_i_style
                
            # Se ainda for "auto", aplicamos a heurística do Vivado:
            if "auto" in r_style or r_style == "":
                decision_style = "block" if depth > 512 else "distributed"
            else:
                decision_style = r_style
                
            # Regra de Ouro do hardware: Vivado/FINN não gastam BRAM e forçam LUT (distributed)
            # para qualquer FIFO com depth menor ou igual a 256.
            if depth <= 256:
                decision_style = "distributed"
                
            # FINN executa a pass *SplitLargeFIFOs* sobre o grafo ONNX logo no início,
            # então ele particionará em pedaços binários independentemente de ser rtl ou vivado!
            slices = finn_partition_fifo(depth, decision_style)
            
            accumulated = {"Total LUT": 0.0, "Total FFs": 0.0, "BRAM (36k eq.)": 0.0, "DSP Blocks": 0.0}
            
            for s in slices:
                specialist_name = "SplitFIFO_block" if s["ram_style"] == "block" else "SplitFIFO_distributed"
                spec_model = self._models.get(specialist_name)
                
                if not spec_model:
                    continue

                feat = prepare_fifo_features(s["depth"], in_width, s["ram_style"], s["impl_style"], bits, parallel)
                X_input = [float(feat.get(f, 0.0)) for f in spec_model.feature_names]
                X_arr = np.array(X_input, dtype=np.float32).reshape(1, -1)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    preds = spec_model.model.predict(X_arr)[0]
                
                preds = np.maximum(0, preds)
                logic_luts, lutrams, srls, total_ffs, ramb36_ml, ramb18_ml, dsp = preds
                
                # Heurística Determinística (Substitui XGBoost em casos de Extrapolação)
                ramb36_final = 0.0
                ramb18_final = 0.0
                
                # Vivado xpm_fifo_axis (AXI-Stream) sempre padas a in_width para múltiplos de 8 bits
                axi_width = int(np.ceil(in_width / 8.0) * 8)
                capacity = s["depth"] * axi_width
                
                if s["ram_style"] == "block":
                    # Regra de Alocação de Macro do Vivado (xpm_fifo_axis)
                    if capacity <= 18432:
                        ramb18_final = 1.0
                    else:
                        ramb36_final = float(np.ceil(capacity / 36864.0))
                elif s["ram_style"] == "distributed" and s.get("impl_style") == "vivado":
                    # Árvores de Decisão (XGBoost) não interpolam fora do domínio de treino. 
                    # Uma depth gigante de 8192 causará um erro de flatline.
                    # Mas em hardware real (LUTRAM logic), a área é 100% linear à capacidade AXI!
                    logic_luts = float((capacity / 20.4) + 40.0)
                    lutrams = 0.0
                    srls = 0.0
                    total_ffs = float((capacity / 34.0) + 60.0)
                
                accumulated["Total LUT"] += float(logic_luts + lutrams + srls)
                accumulated["Total FFs"] += float(total_ffs)
                accumulated["BRAM (36k eq.)"] += float(ramb36_final + (ramb18_final * 0.5))
                accumulated["DSP Blocks"] += float(np.round(dsp * 2) / 2)
            
            return accumulated

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

        # Nenhuma lógica adicional necessária para StreamingFIFO (já processada)

        # One-Hot Encoding
        for cat in ["ram_style", "resType", "mem_mode", "binaryXnorMode", "impl_style"]:
            if cat in raw_data:
                val = str(raw_data[cat])
                clean = ''.join(e for e in val if e.isalnum()).capitalize()
                raw_data[f"is{cat[0].upper() + cat[1:]}{clean}"] = 1

        # NOVO: Trava para DataWidthConverter
        if "StreamingDataWidthConverter" in module_key:
            return {
                "Total LUT": 20.0,
                "Total FFs": 40.0,
                "BRAM (36k eq.)": 0.0,
                "DSP Blocks": 0.0
            }

        # TENSOR INTELLIGENCE: Complexidade Multiplicativa
        if "MVAU" in module_key:
            in_bits = float(raw_data.get("inputDataType (bits)", 1))
            w_bits = float(raw_data.get("weightDataType (bits)", 1))
            pe = float(raw_data.get("PE", 1))
            simd = float(raw_data.get("SIMD", 1))
            
            raw_data["mac_complexity"] = in_bits * w_bits * pe * simd

        # Alinhamento e Predição
        X_input = [float(raw_data.get(f, 0.0)) for f in model_obj.feature_names]
        X_arr = np.array(X_input, dtype=np.float32).reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # A predição bruta agora está na escala LOG
            log_pred = model_obj.model.predict(X_arr)[0]

        # REVERSÃO DO LOG: np.expm1 reverte o log1p do treino
        real_pred = np.expm1(log_pred)
        return {col: max(0.0, float(v)) for col, v in zip(model_obj.target_cols, real_pred)}

    def _extract_bits(self, dtype_str):
        """Extrai numericamente o bitwidth de strings complexas do FINN."""
        if not dtype_str: 
            return 8 # Fallback seguro
        
        s = str(dtype_str).upper()
        
        # 1. Checagem de tipos binarizados (Prioridade Máxima)
        if any(x in s for x in ["BINARY", "BIPOLAR", "INT1", "UINT1"]): 
            return 1
            
        # 2. Busca por números na string (ex: INT17 -> 17, UINT2 -> 2)
        nums = re.findall(r'\d+', s)
        if nums:
            return int(nums[0])
            
        # 3. Fallback: Se não houver número mas for um tipo conhecido, assume-se 8 bits
        return 8

    def predict(self, onnx_path: str, foldings: list[dict]) -> list[dict]:
        if not self.is_loaded(): return [{}] * len(foldings)
        
        topology_key = "UNKNOWN"
        for topo in SUPPORTED_TOPOLOGIES:
            if topo in onnx_path:
                topology_key = topo
                break
        
        onnx_model = onnx.load(onnx_path)
        graph_nodes = {n.name: n for n in onnx_model.graph.node}
        nodes_to_process = self._onnx_cache.get(onnx_path) or self._expand_topology(onnx_path)
        self._onnx_cache[onnx_path] = nodes_to_process
        
        results = []
        for folding in foldings:
            raw_depths = {}
            if _FIFO_PREDICTOR_AVAILABLE and "StreamingFIFO_depth" in self._models:
                try: 
                    depth_model = self._models["StreamingFIFO_depth"]
                    raw_depths = predict_fifo_depths_xgb(onnx_path, folding, depth_model)
                except Exception as e:
                    print(f"[MultiModuleLearner] Erro ao prever depths: {e}")
            
            fifo_depths = {k.replace("_rtl", ""): v for k, v in raw_depths.items()}
            total = {col: 0.0 for col in TARGET_COLS}
            details = {}

            for node_dict in nodes_to_process:
                op = node_dict.get("op_type", "")
                name = node_dict.get("name", "")
                
                # 1. Identificação do Módulo Base
                base_key = None
                for k in ["MVAU", "StreamingFIFO", "ConvolutionInputGenerator", "FMPadding", "Thresholding", "LabelSelect", "StreamingDataWidthConverter"]:
                    if k in op:
                        base_key = k
                        break
                if not base_key: continue

                # 2. Extração Dinâmica de Atributos do Grafo
                w_type = ""
                in_type = ""
                if name in graph_nodes:
                    for attr in graph_nodes[name].attribute:
                        if attr.name == "weightDataType": w_type = attr.s.decode('utf-8')
                        if attr.name == "inputDataType": in_type = attr.s.decode('utf-8')

                # Fallback para o dicionário se o ONNX falhar
                w_type = w_type or node_dict.get("weightDataType", "INT8")
                in_type = in_type or node_dict.get("inputDataType", "INT8")
                
                # Conversão para bits reais
                w_bits = self._extract_bits(w_type)
                in_bits = self._extract_bits(in_type)
                
                # 3. Engenharia de Recursos (Feature Engineering) Dinâmica
                # Extrai PE e SIMD do folding config para este nó específico
                node_folding = folding.get(name, {})
                pe = float(node_folding.get("PE", 1))
                simd = float(node_folding.get("SIMD", 1))
                
                # Atualiza node_dict com as features que o XGBoost espera
                node_dict["weightDataType (bits)"] = w_bits
                node_dict["inputDataType (bits)"] = in_bits
                node_dict["PE"] = pe
                node_dict["SIMD"] = simd
                node_dict["inWidth"] = in_bits * simd
                node_dict["mac_complexity"] = float(in_bits * w_bits * pe * simd)
                
                # 4. Roteamento Inteligente
                matched_cfg = {}
                for k, v in folding.items():
                    clean_k = k.replace("_rtl", "").replace("_hls", "")
                    if clean_k == name:
                        matched_cfg = v
                        break
                
                user_depth = matched_cfg.get("depth")
                if user_depth is not None:
                    d = user_depth
                else:
                    d = fifo_depths.get(name, 2)
                    
                module_key = base_key
                
                if base_key == "MVAU":
                    module_key = "MVAU_1Bit" if w_bits == 1 else "MVAU_MultiBit"
                elif base_key == "StreamingFIFO":
                    module_key = "SplitFIFO_area"
    
                if "MVAU" in base_key:
                    # O module_key agora é mapeado diretamente para o especialista daquela rede
                    module_key = f"MVAU_{topology_key}"
                    
                    # Injeta as features para o XGBoost (bits e complexidade)
                    node_dict["weightDataType (bits)"] = self._extract_bits(w_type)
                    node_dict["inputDataType (bits)"] = self._extract_bits(in_type)
                    pe = float(folding.get(name, {}).get("PE", 1))
                    simd = float(folding.get(name, {}).get("SIMD", 1))
                    node_dict["mac_complexity"] = float(node_dict["inputDataType (bits)"] * node_dict["weightDataType (bits)"] * pe * simd)
                    
                # 5. Predição
                pred = self._predict_layer(module_key, node_dict, folding, d)
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