# ai/hardware_learner.py
"""
HardwareLearner: Módulo de aprendizado de máquina para predição e fine-tuning
de recursos de hardware no loop de exploração HARA.

Responsabilidades:
  - featurize : extrai vetor de features de um folding config + ONNX
  - predict   : prediz [Total LUTs, FFs, BRAM (36k), DSP Blocks] para N configs
  - calculate_discrepancy : compara predições vs. resultado real do Vivado
  - fine_tune : re-treina o XGBoost dando maior peso ao build atual
"""

import os
import pickle
import warnings
import numpy as np
import onnx
from onnx import helper

# Targets de recurso — mesma ordem usada no treinamento offline
RESOURCE_TARGETS = ["Total LUTs", "FFs", "BRAM (36k)", "DSP Blocks"]

# Atributos por op_type que queremos extrair como features
_OP_ATTRS = {
    "MVAU":                       ["MH", "MW", "WBits"],
    "MatrixVectorActivation":     ["MH", "MW", "WBits"],
    "ConvolutionInputGenerator":  ["IFMChannels"],
    "Downsampler":                ["IFMChannels"],
    "FMPadding":                  ["NumChannels"],
    "FMPadding_Pixel":            ["NumChannels"],
    "Thresholding":               ["MH"],
    "LabelSelect":                ["Labels"],
    "Globalaccpool":              ["PE"],           # apenas PE controla
    "AddStreams":                 [],
    "ChannelwiseOp":              [],
    "StreamingEltwise":           [],
    "DuplicateStreams":           [],
    "VectorVectorActivation":     ["NumChannels"],
}


# ---------------------------------------------------------------------------
# Utilitários de feature extraction
# ---------------------------------------------------------------------------

def _get_node_attrs(node: onnx.NodeProto) -> dict:
    """Retorna dicionário {attr_name: valor} de um nó ONNX."""
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def _match_op_key(op_type: str) -> str | None:
    """Encontra a chave de _OP_ATTRS correspondente ao op_type do nó."""
    for key in _OP_ATTRS:
        if key in op_type:
            return key
    return None


def extract_onnx_layer_features(onnx_path: str) -> dict:
    """
    Percorre o grafo ONNX e extrai dimensões fixas de cada camada de hw
    (MH, MW, WBits, IFMChannels, etc.), que são invariantes ao folding.

    Retorna:
        dict{ layer_name -> { attr_name: valor, ... } }
    """
    model = onnx.load(onnx_path)
    layer_feats = {}
    for node in model.graph.node:
        key = _match_op_key(node.op_type)
        if key is None:
            continue
        attrs = _get_node_attrs(node)
        feat = {"op_type": node.op_type}
        for attr_name in _OP_ATTRS[key]:
            val = attrs.get(attr_name, 0)
            # Alguns atributos são listas (ex: numInputVectors); pega produto
            if isinstance(val, (list, tuple)):
                val = int(np.prod(val)) if len(val) > 0 else 0
            feat[attr_name] = int(val)
        layer_feats[node.name] = feat
    return layer_feats


def featurize(onnx_path: str, folding_config: dict) -> np.ndarray:
    """
    Converte (onnx_path, folding_config) em um vetor de features 1D.

    Estratégia:
      Para cada camada HW presente no grafo ONNX (exceto 'Defaults'):
        - features fixas: MH, MW, WBits, IFMChannels, NumChannels, Labels
        - features variáveis: PE, SIMD (do folding_config)
      O vetor é construído na ordem determinística dos nós no grafo.

    Retorna:
        np.ndarray de shape (N_features,), dtype float32
    """
    layer_feats = extract_onnx_layer_features(onnx_path)
    feature_vec = []

    # Percorre na ordem dos nós do grafo (determinística)
    model = onnx.load(onnx_path)
    for node in model.graph.node:
        if node.name not in layer_feats:
            continue
        feat = layer_feats[node.name]
        fold = folding_config.get(node.name, {})

        # Features fixas (estrutura da rede)
        feature_vec.append(float(feat.get("MH", 0)))
        feature_vec.append(float(feat.get("MW", 0)))
        feature_vec.append(float(feat.get("WBits", 0)))
        feature_vec.append(float(feat.get("IFMChannels", 0)))
        feature_vec.append(float(feat.get("NumChannels", 0)))
        feature_vec.append(float(feat.get("Labels", 0)))

        # Features variáveis (paralelismo — dependem do folding_config)
        feature_vec.append(float(fold.get("PE", 1)))
        feature_vec.append(float(fold.get("SIMD", 1)))

    return np.array(feature_vec, dtype=np.float32)


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class HardwareLearner:
    """
    Encapsula um modelo XGBoost multi-output para predição de recursos de
    hardware e suporta fine-tuning online com sample weighting.

    Uso típico:
        learner = HardwareLearner()
        learner.load("ai/trained_model.pkl")

        predictions = learner.predict(onnx_path, [folding1, folding2, ...])
        disc = learner.calculate_discrepancy(predictions[0], actual_resources, resource_limits)
        learner.fine_tune(onnx_path, folding1, actual_resources, current_build_weight=10)
    """

    def __init__(self):
        self.model = None                # MultiOutputRegressor(XGBRegressor)
        self._X_buffer: list[np.ndarray] = []   # features de cada build real
        self._y_buffer: list[np.ndarray] = []   # recursos reais de cada build
        self._weights_buffer: list[float] = []  # peso de cada ponto histórico

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load(self, model_path: str) -> None:
        """
        Carrega um modelo XGBoost pré-treinado de um arquivo pickle.

        O arquivo deve conter um objeto sklearn MultiOutputRegressor ou
        equivalente com método .predict(X).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[HardwareLearner] Modelo não encontrado: {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"[HardwareLearner] Modelo carregado de: {model_path}")

    def is_loaded(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    # Featurization (exposta para uso externo se necessário)
    # ------------------------------------------------------------------

    def featurize(self, onnx_path: str, folding_config: dict) -> np.ndarray:
        """Wrapper público para a função featurize() do módulo."""
        return featurize(onnx_path, folding_config)

    # ------------------------------------------------------------------
    # Predição
    # ------------------------------------------------------------------

    def predict(self, onnx_path: str, folding_configs: list[dict]) -> list[dict]:
        """
        Prediz recursos de hardware para uma lista de configurações de folding.

        Args:
            onnx_path       : caminho para o ONNX intermediário do FINN
                              (step_generate_estimate_reports.onnx)
            folding_configs : lista de dicts de folding (um por configuração)

        Retorna:
            Lista de dicts { "Total LUTs": v, "FFs": v, "BRAM (36k)": v, "DSP Blocks": v }
            na mesma ordem da entrada.
        """
        if not self.is_loaded():
            raise RuntimeError("[HardwareLearner] Modelo não carregado. Chame .load() primeiro.")

        if not folding_configs:
            return []

        X = np.stack([featurize(onnx_path, fc) for fc in folding_configs])
        raw_preds = self.model.predict(X)  # shape (N, 4)

        results = []
        for row in raw_preds:
            pred = {}
            for i, target in enumerate(RESOURCE_TARGETS):
                pred[target] = max(0.0, float(row[i]))
            results.append(pred)

        print(f"[HardwareLearner] Predição para {len(folding_configs)} configs concluída.")
        return results

    def predict_single(self, onnx_path: str, folding_config: dict) -> dict:
        """Atalho para predizer uma única configuração."""
        return self.predict(onnx_path, [folding_config])[0]

    # ------------------------------------------------------------------
    # Discrepância
    # ------------------------------------------------------------------

    def calculate_discrepancy(
        self,
        predicted: dict,
        actual: dict,
        resource_limits: dict,
    ) -> dict:
        """
        Calcula a discrepância entre predição e resultado real do Vivado.

        A discrepância de cada recurso é: actual - predicted (valores absolutos).
        A utilização relativa é: actual / limit (em %, considerando 100% = budget).
        O dominant resource é o recurso com maior utilização relativa.

        Args:
            predicted        : dict de recursos preditos pelo learner
            actual           : dict de recursos reais extraídos do Vivado
            resource_limits  : dict com os limites máximos da FPGA

        Retorna dict com:
            "absolute"       : { recurso: (actual - predicted) }
            "relative_error" : { recurso: (actual - predicted) / max(predicted, 1) }
            "utilization_pct": { recurso: actual / limit * 100 }
            "dominant_resource": nome do recurso dominante
            "dominant_utilization_pct": valor percentual do dominante
        """
        absolute = {}
        relative_error = {}
        utilization_pct = {}

        for res in RESOURCE_TARGETS:
            pred_val  = predicted.get(res, 0.0)
            real_val  = actual.get(res, 0.0)
            limit_val = resource_limits.get(res, 1)

            absolute[res] = real_val - pred_val
            relative_error[res] = (real_val - pred_val) / max(abs(pred_val), 1.0)
            utilization_pct[res] = (real_val / max(limit_val, 1)) * 100.0

        # Dominant resource = recurso com maior % de utilização da FPGA
        dominant = max(utilization_pct, key=utilization_pct.get)

        result = {
            "absolute":                 absolute,
            "relative_error":          relative_error,
            "utilization_pct":         utilization_pct,
            "dominant_resource":       dominant,
            "dominant_utilization_pct": utilization_pct[dominant],
        }

        print(f"[HardwareLearner] Discrepância calculada:")
        for res in RESOURCE_TARGETS:
            print(
                f"   {res:15s}: pred={predicted.get(res, 0):.0f}  "
                f"real={actual.get(res, 0):.0f}  "
                f"diff={absolute[res]:+.0f}  "
                f"util={utilization_pct[res]:.1f}%"
            )
        print(f"   → Dominant resource: {dominant} ({utilization_pct[dominant]:.1f}%)")

        return result

    # ------------------------------------------------------------------
    # Fine-tuning online
    # ------------------------------------------------------------------

    def fine_tune(
        self,
        onnx_path: str,
        folding_config: dict,
        actual_resources: dict,
        current_build_weight: float = 10.0,
    ) -> None:
        """
        Adiciona o novo ponto real ao buffer e re-treina o XGBoost com
        sample weighting, dando maior peso ao build atual.

        Args:
            onnx_path             : caminho para o ONNX intermediário
            folding_config        : folding do build que acaba de ser medido
            actual_resources      : dict de recursos reais { "Total LUTs": v, ... }
            current_build_weight  : peso a dar ao ponto atual (histórico = 1.0)

        Comportamento:
            - O ponto atual recebe peso `current_build_weight`.
            - Pontos históricos no buffer recebem peso 1.0.
            - Re-treina via model.fit(X, y, sample_weight=w).
        """
        if not self.is_loaded():
            raise RuntimeError("[HardwareLearner] Modelo não carregado. Chame .load() primeiro.")

        # Vetoriza o novo ponto
        x_new = featurize(onnx_path, folding_config)
        y_new = np.array([
            actual_resources.get(res, 0.0) for res in RESOURCE_TARGETS
        ], dtype=np.float32)

        # Acumula no buffer com peso histórico = 1.0
        self._X_buffer.append(x_new)
        self._y_buffer.append(y_new)
        self._weights_buffer.append(1.0)

        # Eleva o peso do último ponto (build atual)
        self._weights_buffer[-1] = current_build_weight

        # Monta matrizes de treino
        X_train = np.stack(self._X_buffer)       # (N, features)
        y_train = np.stack(self._y_buffer)        # (N, 4)
        weights  = np.array(self._weights_buffer) # (N,)

        n = len(self._X_buffer)
        print(f"[HardwareLearner] Fine-tuning com {n} ponto(s) "
              f"(peso atual={current_build_weight:.1f})...")

        # Re-treina cada estimador individual do MultiOutputRegressor
        # usando o parâmetro sample_weight via fit_params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # sklearn >= 1.4: MultiOutputRegressor aceita diretamente
                self.model.fit(X_train, y_train,
                               sample_weight=weights)
            except TypeError:
                # Fallback: re-treina estimador por estimador
                import copy
                from sklearn.multioutput import MultiOutputRegressor
                for idx, estimator in enumerate(self.model.estimators_):
                    estimator.fit(X_train, y_train[:, idx],
                                  sample_weight=weights)

        print(f"[HardwareLearner] Fine-tune concluído.")

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def select_best_config(
        self,
        predictions: list[dict],
        folding_configs: list[dict],
        resource_limits: dict,
        budget_pct: float = 1.0,
    ) -> tuple[int, dict, dict]:
        """
        Seleciona a configuração de folding com maior índice na lista
        (maior paralelismo / FPS estimado) cujos recursos preditos ainda cabem
        dentro do budget.

        Args:
            predictions     : saída de self.predict() — lista de dicts de recursos
            folding_configs : lista de configs de folding alinhados com predictions
            resource_limits : dict com limites máximos da FPGA
            budget_pct      : fração do limite a usar como teto (default=1.0 = 100%)

        Retorna:
            (índice, folding_config selecionado, predição selecionada)
            Retorna (-1, {}, {}) se nenhuma config coube.
        """
        best_idx = -1
        best_folding = {}
        best_pred = {}

        for i, (pred, fold) in enumerate(zip(predictions, folding_configs)):
            fits = all(
                pred.get(res, 0) <= resource_limits.get(res, float("inf")) * budget_pct
                for res in RESOURCE_TARGETS
            )
            if fits:
                best_idx = i
                best_folding = fold
                best_pred = pred

        if best_idx == -1:
            print("[HardwareLearner] ⚠ Nenhuma configuração predita cabe no budget.")
        else:
            print(f"[HardwareLearner] Config selecionada: índice #{best_idx}")

        return best_idx, best_folding, best_pred
