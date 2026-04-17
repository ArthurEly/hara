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
import sys 

import onnx
from onnx import helper

# Bibliotecas Core do FINN/QONNX
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO

# Integração com o Preditor de Profundidade
_FIFO_PREDICTOR_AVAILABLE = False
try:
    from predict_fifo_depths import predict_fifo_depths_xgb
    _FIFO_PREDICTOR_AVAILABLE = True
except ImportError:
    try:
        # Fallback: resolve relative to this file's directory
        _ai_dir = os.path.dirname(os.path.abspath(__file__))
        if _ai_dir not in sys.path:
            sys.path.insert(0, _ai_dir)
        from predict_fifo_depths import predict_fifo_depths_xgb
        _FIFO_PREDICTOR_AVAILABLE = True
    except ImportError:
        print("[MultiModuleLearner] AVISO: predict_fifo_depths não encontrado.")
print(f"[MultiModuleLearner] FIFO Predictor: {'ATIVO' if _FIFO_PREDICTOR_AVAILABLE else 'INATIVO'}")

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

TARGET_COLS = ["Total LUT", "Logic LUTs", "LUTRAMs", "SRLs", "Total FFs", "BRAM (36k eq.)", "DSP Blocks"]
SUPPORTED_TOPOLOGIES = [
    "MNIST_1W1A",
    "MNIST_2W2A",
    "SAT6_T2W2",
    "SAT6_T2W4",
    "SAT6_T2W8",
    "CIFAR10_1W1A",
    "CIFAR10_2W2A",
]
# =============================================================================
# MODELO ESPECIALISTA (Wrapper XGBoost)
# =============================================================================

class _PerModuleModel:
    def __init__(self, pkl_path: str):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        self.obj_dict = obj
        self.model = obj.get("model") or obj.get("stage2_regressor")
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
                        # IMPORTANTE: Preservar o folded_shape como lista para extrair a largura do barramento!
                        if attr.name == "folded_shape":
                            val = list(val)
                        else:
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
        #if "MVAU" in module_key:            
            #print(f"\n--- [DEBUG MVAU INPUT: {layer_name}] ---")
            #print(f"  Especialista Alvo: {module_key}")
            #print(f"  weightDataType: {raw_data.get('weightDataType')}")
            #print(f"  weightDataType (bits): {raw_data.get('weightDataType (bits)')}")
            #print(f"  mac_complexity: {raw_data.get('mac_complexity')}")
            #print(f"  PE: {raw_data.get('PE')} | SIMD: {raw_data.get('SIMD')}")
            #print("------------------------------------------\n")
        
        
        # --- O ROTEADOR DE ESPECIALISTAS (Mixture of Experts) ---
        if "StreamingFIFO" in module_key:
            module_key = "SplitFIFO_area"
            
        model_obj = self._models.get(module_key)
        if not model_obj: return None
        
        # --- LÓGICA ESPECIAL PARA FATIAS E AUTO-STYLE (SPLIT_FIFO) ---
        if module_key == "SplitFIFO_area":
            try:
                from ai.predict_fifo_utils import finn_partition_fifo, prepare_fifo_features
            except ImportError:
                from predict_fifo_utils import finn_partition_fifo, prepare_fifo_features
            
            # RECUPERA O TENSOR (BITS e IN_WIDTH)
            input_tensor = node_attrs.get("_in_tensor")
            bits = 0
            if input_tensor and self.current_wrapper:
                dtype = self.current_wrapper.get_tensor_datatype(input_tensor)
                bits = self._extract_bitwidth(str(dtype))
            if bits == 0: bits = 8

            # inWidth correto: folded_shape[-1] × bits (AXI-S bus width per transfer).
            folded_shape = node_attrs.get("folded_shape", [])
            
            if isinstance(folded_shape, list) and len(folded_shape) > 0:
                # A última dimensão do folded_shape dita exatamente quantos elementos passam no barramento por ciclo.
                parallel = int(folded_shape[-1])
                in_width = bits * parallel
            else:
                # Fallback de segurança (muito raro falhar após InsertFIFO)
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
            
            accumulated = {
                "Total LUT": 0.0, 
                "Logic LUTs": 0.0,
                "LUTRAMs": 0.0,
                "SRLs": 0.0,
                "Total FFs": 0.0, 
                "BRAM (36k eq.)": 0.0, 
                "DSP Blocks": 0.0
            }
            
            for s in slices:
                # --- DETERMINISTIC RTL FIFO MODEL ---
                # Substitui o ML por física de hardware para FIFOs RTL (que não usam BRAM)
                if s["impl_style"] == "rtl":
                    if s["depth"] <= 2:
                        # FIFOs muito rasas: sem shift registers, apenas FFs.
                        # Overhead de controle mínimo de ~22 FFs verificado empiricamente.
                        ff_base = in_width * s["depth"] + 6.0
                        total_ffs = max(22.0, ff_base)
                        logic_luts = 15.0
                        srls = 0.0
                        lutrams = 0.0
                    else:
                        # FIFOs baseadas em SRL (Shift Register LUT).
                        # O Vivado mapeia cadeias de até 32 bits num SRLC32E.
                        srl_chains = int(np.ceil(s["depth"] / 32.0))
                        srls = srl_chains * in_width
                        # A lógica de controle escala com o log2 da profundidade (ponteiros)
                        logic_luts = 30.0 + np.ceil(np.log2(s["depth"])) * 4.0
                        lutrams = 0.0
                        # FFs: Output register (in_width) + controle state machine/flags (~20)
                        total_ffs = float(in_width) + 20.0
                    
                    accumulated["Total LUT"]   += float(logic_luts + srls)
                    accumulated["Logic LUTs"]  += float(logic_luts)
                    accumulated["SRLs"]        += float(srls)
                    accumulated["Total FFs"]   += float(total_ffs)
                    continue

                # --- VIVADO/BRAM FIFO MODEL (Keep existing logic) ---
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
                accumulated["Logic LUTs"] += float(logic_luts)
                accumulated["LUTRAMs"] += float(lutrams)
                accumulated["SRLs"] += float(srls)
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

    def _resolve_topology_key(self, onnx_path: str) -> str:
        # Tenta buscar no caminho completo do arquivo
        for topo in SUPPORTED_TOPOLOGIES:
            if topo in onnx_path:
                return topo
                
        # Fallbacks estáticos baseados em palavras-chave do dataset
        path_lower = onnx_path.lower()
        if "sat6" in path_lower:
            return "SAT6_T2W2"
        if "mnist" in path_lower:
            return "MNIST_1W1A"
        if "cifar10" in path_lower:
            return "CIFAR10_1W1A"
            
        print(f"[!] AVISO: Topologia desconhecida para {onnx_path}. Assumindo UNKNOWN.")
        return "UNKNOWN"

    def _select_depth_specialist_key(self, topology_key: str) -> str:
        if "SAT6" in topology_key and "StreamingFIFO_depth_SAT6_T2" in self._models:
            return "StreamingFIFO_depth_SAT6_T2"
        if "MNIST" in topology_key and "StreamingFIFO_depth_MNIST_TFC" in self._models:
            return "StreamingFIFO_depth_MNIST_TFC"
        if "CIFAR10" in topology_key and "StreamingFIFO_depth_CIFAR10_CNV" in self._models:
            return "StreamingFIFO_depth_CIFAR10_CNV"
        if "StreamingFIFO_depth_UNIFIED" in self._models:
            return "StreamingFIFO_depth_UNIFIED"
        return "StreamingFIFO_depth"

    def _get_onnx_fallback_depths(self, expanded_model) -> dict:
        depths = {}
        for gn in expanded_model.graph.node:
            if "FIFO" in gn.op_type.upper():
                d_val = 2
                for a in gn.attribute:
                    if a.name == "depth":
                        d_val = int(helper.get_attribute_value(a))
                depths[gn.name] = d_val
        return depths

    def predict_fifo_depths_batch(self, onnx_path: str, foldings: list[dict],
                                   cycles_per_folding: list[dict] | None = None) -> list[dict]:
        """Predicts FIFO depths for all foldings using the topology-specific specialist.

        Args:
            cycles_per_folding: lista de dicts com conteúdo de estimate_layer_cycles.json,
                                 um por folding. Quando fornecido, usa ciclos reais em vez
                                 da fórmula MH×MW — melhora muito a qualidade da predição.
        """
        topology_key = self._resolve_topology_key(onnx_path)
        depth_key = self._select_depth_specialist_key(topology_key)

        if onnx_path not in self._onnx_cache:
            self._onnx_cache[onnx_path] = self._expand_topology(onnx_path)
            fifo_count = sum(1 for n in self.current_wrapper.graph.node if "FIFO" in n.op_type.upper())
            total_count = len(list(self.current_wrapper.graph.node))
            print(f"[HARA] Topologia expandida: {total_count} nós, {fifo_count} FIFOs")

        expanded_model = self.current_wrapper
        onnx_fifo_depths = self._get_onnx_fallback_depths(expanded_model)

        using_cycles = cycles_per_folding is not None
        print(f"[HARA-Depth] Especialista: {depth_key} | Topologia: {topology_key} | "
              f"FIFOs: {len(onnx_fifo_depths)} | cycles_json: {'SIM' if using_cycles else 'NAO (formula)'}")

        all_depths = []
        for i, folding in enumerate(foldings):
            cycles_cfg = cycles_per_folding[i] if (cycles_per_folding and i < len(cycles_per_folding)) else None
            if _FIFO_PREDICTOR_AVAILABLE and depth_key in self._models:
                try:
                    raw = predict_fifo_depths_xgb(
                        expanded_model, folding, self._models[depth_key], cycles_cfg=cycles_cfg
                    )
                    raw = raw if raw else dict(onnx_fifo_depths)
                except Exception:
                    raw = dict(onnx_fifo_depths)
            else:
                raw = dict(onnx_fifo_depths)
            all_depths.append({k.replace("_rtl", ""): v for k, v in raw.items()})

        if all_depths and all_depths[0]:
            vals = list(all_depths[0].values())
            non_trivial = sum(1 for d in vals if d > 2)
            print(f"[HARA-Depth] Folding-0 amostra: {len(vals)} FIFOs, {non_trivial} com depth>2, "
                  f"max={max(vals)}, median={sorted(vals)[len(vals)//2]}")
        return all_depths

    def predict(self, onnx_path: str, foldings: list[dict],
                precomputed_depths: list[dict] | None = None,
                cycles_cfg_list: list[dict] | None = None) -> list[dict]:
        if not self.is_loaded(): return [{}] * len(foldings)

        topology_key = self._resolve_topology_key(onnx_path)

        # 1. Expand topology ONCE to get the FIFOs and DWCs stitcheadas
        if onnx_path not in self._onnx_cache:
            self._onnx_cache[onnx_path] = self._expand_topology(onnx_path)
            fifo_count = sum(1 for n in self.current_wrapper.graph.node if "FIFO" in n.op_type.upper())
            total_count = len(list(self.current_wrapper.graph.node))
            print(f"[HARA] Topologia expandida: {total_count} nós, {fifo_count} FIFOs")

        nodes_to_process = self._onnx_cache[onnx_path]
        expanded_model = self.current_wrapper
        graph_nodes = {n.name: n for n in expanded_model.graph.node}

        onnx_fifo_depths = self._get_onnx_fallback_depths(expanded_model)

        # Choose depth specialist (used only when precomputed_depths is not provided)
        depth_key = self._select_depth_specialist_key(topology_key)
        if precomputed_depths is None:
            print(f"[MultiModuleLearner] -> Usando especialista de depth: {depth_key} (Topo: {topology_key})")

        results = []
        for i, folding in enumerate(foldings):
            # Use pre-computed depths if available, otherwise predict inline
            if precomputed_depths is not None and i < len(precomputed_depths):
                fifo_depths = precomputed_depths[i]
            else:
                raw_depths = {}
                if _FIFO_PREDICTOR_AVAILABLE and depth_key in self._models:
                    try:
                        cycles_cfg = cycles_cfg_list[i] if (cycles_cfg_list and i < len(cycles_cfg_list)) else None
                        raw_depths = predict_fifo_depths_xgb(expanded_model, folding, self._models[depth_key], cycles_cfg=cycles_cfg)
                        if not raw_depths:
                            if i == 0: print(f"[MultiModuleLearner] AVISO: XGBoost retornou vazio, usando depths do ONNX")
                            raw_depths = dict(onnx_fifo_depths)
                        else:
                            if i == 0: print(f"[MultiModuleLearner] ✓ XGBoost predisse depths para {len(raw_depths)} FIFOs")
                    except Exception as e:
                        if i == 0: print(f"[MultiModuleLearner] Erro depths XGBoost: {e}, usando fallback ONNX")
                        raw_depths = dict(onnx_fifo_depths)
                else:
                    raw_depths = dict(onnx_fifo_depths)
                    if i == 0 and onnx_fifo_depths:
                        avail = "ATIVO" if _FIFO_PREDICTOR_AVAILABLE else "INATIVO"
                        has_model = depth_key in self._models
                        print(f"[MultiModuleLearner] FIFO XGBoost={avail}, modelo_carregado={has_model} → usando depths do ONNX ({len(onnx_fifo_depths)} FIFOs)")
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

            # Ensure Logic LUTs + LUTRAMs + SRLs == Total LUTs.
            # Modules that only predict Total LUT (no breakdown) leave the
            # sub-fields at 0; attribute the gap to Logic LUTs.
            total_lut = total.get("Total LUT", 0)
            breakdown  = total.get("Logic LUTs", 0) + total.get("LUTRAMs", 0) + total.get("SRLs", 0)
            if breakdown < total_lut:
                total["Logic LUTs"] = total.get("Logic LUTs", 0) + (total_lut - breakdown)

            results.append({
                "Total LUTs":  int(round(total_lut)),
                "Logic LUTs":  int(round(total.get("Logic LUTs", 0))),
                "LUTRAMs":     int(round(total.get("LUTRAMs", 0))),
                "SRLs":        int(round(total.get("SRLs", 0))),
                "FFs":         int(round(total.get("Total FFs", total.get("FFs", 0)))),
                "BRAM (36k)":  round(total.get("BRAM (36k eq.)", total.get("BRAM (36k)", 0)), 1),
                "DSP Blocks":  int(round(total.get("DSP Blocks", 0))),
                "fifo_depths": fifo_depths,
                "_details":    details
            })
        return results

# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_util_rpt(rpt_path: str) -> dict:
    """Parse finn_design_partition_util.rpt.
    Returns {instance_name: {LUT, LogLUT, LUTRAM, SRL, FF, BRAM, DSP}}

    Mirrors the indent-tracking approach in get_exhaustive_area_results.py:
    finds finn_design_i, then takes direct children at indent+2.
    FIFO chains aggregated: StreamingFIFO_rtl_N_K → StreamingFIFO_N.
    """
    results = {}
    try:
        with open(rpt_path, "r", encoding="utf-8") as f:
            content = f.readlines()
    except Exception as e:
        print(f"[parse_util_rpt] Erro: {e}")
        return results

    in_utilization_table = False
    header_indices: dict[str, int] = {}
    col_headers: list[str] = []
    found_finn_design_i = False
    finn_design_i_indent = -1

    for line_raw in content:
        line_s = line_raw.strip()

        if not line_s.startswith("|") and "1. Utilization by Hierarchy" in line_s:
            in_utilization_table = True
            continue
        if not in_utilization_table:
            continue
        if not line_s.startswith("|"):
            if found_finn_design_i:
                break
            continue

        # Detect header row
        if not col_headers and "Instance" in line_s and "Module" in line_s:
            temp = [h.strip() for h in line_s.split("|") if h.strip()]
            if temp and temp[0] == "Instance" and temp[1] == "Module":
                col_headers = temp
                for hdr in ["Total LUTs", "Logic LUTs", "LUTRAMs", "SRLs",
                             "FFs", "RAMB36", "RAMB18", "DSP Blocks"]:
                    if hdr in col_headers:
                        header_indices[hdr] = col_headers.index(hdr)
            continue

        if not header_indices:
            continue

        try:
            first_cell_raw = line_raw.split("|", 2)[1]
        except IndexError:
            continue

        indent        = len(first_cell_raw) - len(first_cell_raw.lstrip(" "))
        instance_name = first_cell_raw.strip()
        data_parts    = [p.strip() for p in line_s.split("|")[1:-1]]

        if len(data_parts) != len(col_headers):
            continue

        if not found_finn_design_i:
            if instance_name == "finn_design_i":
                found_finn_design_i = True
                finn_design_i_indent = indent
            continue

        # Direct children of finn_design_i
        if indent == finn_design_i_indent + 2:
            if instance_name.startswith("("):
                continue
            try:
                def _i(h): return int(data_parts[header_indices[h]])   if h in header_indices else 0
                def _f(h): return float(data_parts[header_indices[h]]) if h in header_indices else 0.0
                lut    = _i("Total LUTs"); llut  = _i("Logic LUTs")
                lutram = _i("LUTRAMs");    srl   = _i("SRLs")
                ff     = _i("FFs");        dsp   = _i("DSP Blocks")
                bram   = _f("RAMB36") + _f("RAMB18") * 0.5
            except (ValueError, KeyError):
                continue

            # Aggregate FIFO chains: StreamingFIFO_rtl_N_K → StreamingFIFO_N
            m = re.match(r"StreamingFIFO_(?:rtl|hls)_(\d+)(?:_\d+)?$", instance_name)
            key = f"StreamingFIFO_{m.group(1)}" if m else instance_name

            if key not in results:
                results[key] = dict(LUT=0, LogLUT=0, LUTRAM=0, SRL=0, FF=0, BRAM=0.0, DSP=0)
            results[key]["LUT"]    += lut;  results[key]["LogLUT"] += llut
            results[key]["LUTRAM"] += lutram; results[key]["SRL"]  += srl
            results[key]["FF"]     += ff;   results[key]["BRAM"]   += bram
            results[key]["DSP"]    += dsp

        elif indent <= finn_design_i_indent:
            found_finn_design_i = False

    return results

if __name__ == "__main__":
    import sys, json, pprint
    if len(sys.argv) < 2:
        print("Uso: python multi_module_learner.py <run_dir> [--true-depths]")
        sys.exit(1)

    run_dir = sys.argv[1]
    
    # Flag para forçar o uso de profundidades reais no debug
    use_true_depths = "--true-depths" in sys.argv
    
    onnx_file = os.path.join(run_dir, "intermediate_models", "step_generate_estimate_reports.onnx")
    hw_config = os.path.join(run_dir, "final_hw_config.json")
    stitched_onnx = os.path.join(run_dir, "intermediate_models", "step_create_stitched_ip.onnx")

    with open(hw_config, "r") as f: cfg = json.load(f)
    m_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieval", "results", "trained_models")

    cycles_path = os.path.join(run_dir, "report", "estimate_layer_cycles.json")
    cycles_cfg = {}
    if os.path.exists(cycles_path):
        with open(cycles_path) as f: cycles_cfg = json.load(f)
        print(f"[HARA] Ciclos carregados: {len(cycles_cfg)} nós de {cycles_path}")
    else:
        print(f"[HARA] AVISO: {cycles_path} não encontrado — usando fórmula MH×MW")

    # Extrai profundidades reais do modelo costurado, se solicitado e disponível
    true_depths = None
    if use_true_depths and os.path.exists(stitched_onnx):
        try:
            model_stitch = ModelWrapper(stitched_onnx)
            true_depths = {}
            for node in model_stitch.graph.node:
                if "FIFO" in node.op_type:
                    name_clean = re.sub(r'_\d+$', '', node.name)
                    depth = 2
                    for attr in node.attribute:
                        if attr.name == "depth":
                            val = helper.get_attribute_value(attr)
                            if isinstance(val, (list, tuple, np.ndarray)):
                                depth = int(val[0]) if len(val) > 0 else 2
                            else:
                                depth = int(val)
                            break
                    true_depths[name_clean] = true_depths.get(name_clean, 0) + depth
            print(f"[HARA-Debug] ✓ TRUE DEPTHS carregadas do ONNX final ({len(true_depths)} FIFOs)")
        except Exception as e:
            print(f"[!] Erro ao extrair True Depths: {e}")

    learner = MultiModuleLearner(m_dir)
    
    # Predição com injeção de True Depths
    p = learner.predict(
        onnx_path=onnx_file, 
        foldings=[cfg], 
        cycles_cfg_list=[cycles_cfg],
        precomputed_depths=[true_depths] if true_depths else None
    )

    # --- GT FIFO depths (from final_hw_config.json) ---
    # OBS: O GT do config às vezes não bate com o ONNX final porque o FINN as estica/comprime!
    gt_depths = {k: v.get("depth") for k, v in cfg.items()
                 if isinstance(v, dict) and "depth" in v}
    # Normalize names (strip _rtl/_hls suffixes for matching)
    gt_norm = {}
    for k, v in gt_depths.items():
        clean = re.sub(r'_(rtl|hls|vivado)$', '', k)
        clean = re.sub(r'_(rtl|hls|vivado)_(\d+)$', r'_\2', clean)
        gt_norm[clean] = v
        
    # Se injetamos true_depths, vamos usar elas como o Ground Truth real da tabela
    if true_depths:
        gt_norm = true_depths

    pred_depths = p[0].get("fifo_depths", {})

    print("\n=== FIFO DEPTHS: GT vs PREDITO ===")
    print(f"{'FIFO':<40} {'GT':>8} {'Predito':>10} {'Erro%':>8}")
    print("-" * 70)

    all_keys = sorted(set(list(pred_depths.keys()) + list(gt_norm.keys())))
    for key in all_keys:
        pred_val = pred_depths.get(key)
        # Try matching GT by clean name
        gt_val = gt_norm.get(key) or gt_norm.get(re.sub(r'_(rtl|hls)$', '', key))
        gt_str   = str(gt_val)   if gt_val   is not None else "—"
        pred_str = str(pred_val) if pred_val is not None else "—"
        if gt_val and pred_val:
            err = (pred_val - gt_val) / max(1, gt_val) * 100
            err_str = f"{err:+.1f}%"
        else:
            err_str = ""
        print(f"{key:<40} {gt_str:>8} {pred_str:>10} {err_str:>8}")

    details = p[0].get("_details", {})

    # Load GT utilization from Vivado RPT if available
    rpt_path = os.path.join(run_dir, "stitched_ip", "finn_design_partition_util.rpt")
    gt_util  = parse_util_rpt(rpt_path) if os.path.exists(rpt_path) else {}
    has_gt   = bool(gt_util)

    if details:
        if has_gt:
            hdr = (f"{'Módulo':<42} "
                   f"{'LUT(GT)':>8} {'LUT(P)':>7} {'Err%':>6}  "
                   f"{'FF(GT)':>7} {'FF(P)':>7} {'Err%':>6}  "
                   f"{'BRAM(GT)':>8} {'BRAM(P)':>7} {'Err%':>6}  "
                   f"{'DSP(GT)':>7} {'DSP(P)':>6}")
            print(f"\n=== RECURSOS POR MÓDULO (GT vs PREDITO) ===")
        else:
            hdr = f"{'Módulo':<45} {'LUT':>7} {'LogLUT':>7} {'LUTRAM':>7} {'SRL':>5} {'FF':>7} {'BRAM':>6} {'DSP':>5}"
            print(f"\n=== RECURSOS POR MÓDULO ===")
        print(hdr)
        print("-" * len(hdr))

        # Collect all names from both prediction and GT
        all_names = sorted(set(list(details.keys()) + list(gt_util.keys())))
        for name in all_names:
            pred = details.get(name, {})
            gt   = gt_util.get(name)

            # Also try matching GT by stripping _rtl/_hls suffix from prediction name
            if gt is None:
                clean = re.sub(r'_(rtl|hls)_(\d+)$', r'_\2', name)
                gt = gt_util.get(clean)

            p_lut  = int(round(pred.get("Total LUT",    pred.get("Total LUTs", 0))))
            p_llut = int(round(pred.get("Logic LUTs",   0)))
            p_lram = int(round(pred.get("LUTRAMs",      0)))
            p_srl  = int(round(pred.get("SRLs",         0)))
            p_ff   = int(round(pred.get("Total FFs",    pred.get("FFs", 0))))
            p_bram = round(pred.get("BRAM (36k eq.)",   pred.get("BRAM (36k)", 0)), 1)
            p_dsp  = int(round(pred.get("DSP Blocks",   0)))

            if has_gt and gt:
                g_lut  = gt["LUT"]; g_ff = gt["FF"]
                g_bram = gt["BRAM"]; g_dsp = gt["DSP"]
                def erp(p, g): return f"{(p-g)/max(1,g)*100:+.0f}%" if g or p else "  —"
                print(f"{name:<42} "
                      f"{g_lut:>8} {p_lut:>7} {erp(p_lut,g_lut):>6}  "
                      f"{g_ff:>7} {p_ff:>7} {erp(p_ff,g_ff):>6}  "
                      f"{g_bram:>8.1f} {p_bram:>7.1f} {erp(p_bram,g_bram):>6}  "
                      f"{g_dsp:>7} {p_dsp:>6}")
            else:
                print(f"{name:<45} {p_lut:>7} {p_llut:>7} {p_lram:>7} {p_srl:>5} {p_ff:>7} {p_bram:>6} {p_dsp:>5}")

    print("\n=== ESTIMATIVA FINAL HARA (Área Total Estimada) ===")
    summary = {k: v for k, v in p[0].items() if k not in ("fifo_depths", "_details")}
    pprint.pprint(summary, sort_dicts=False)