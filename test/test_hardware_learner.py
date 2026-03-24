# test/test_hardware_learner.py
"""
Testes unitários do HardwareLearner.
Crie um modelo XGBoost dummy antes de rodar:

    python -c "
    import pickle
    import numpy as np
    from sklearn.multioutput import MultiOutputRegressor
    from xgboost import XGBRegressor

    N, F = 20, 40   # amostras, features (8 por camada * 5 camadas)
    X = np.random.rand(N, F).astype(np.float32)
    y = np.random.rand(N, 4).astype(np.float32) * 10000

    model = MultiOutputRegressor(XGBRegressor(n_estimators=10, random_state=42))
    model.fit(X, y)
    with open('test/dummy_learner.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('Dummy model criado em test/dummy_learner.pkl')
    "

Depois rode:
    python -m pytest test/test_hardware_learner.py -v
"""

import os
import sys
import pickle
import pytest
import numpy as np

# Adiciona a raiz do projeto ao path para imports relativos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai.hardware_learner import HardwareLearner, featurize, RESOURCE_TARGETS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DUMMY_MODEL_PATH = os.path.join(os.path.dirname(__file__), "dummy_learner.pkl")
FAKE_FOLDING = {
    "Defaults": {"PE": 1, "SIMD": 1},
    "MVAU_hls_0": {"PE": 2, "SIMD": 4},
    "ConvolutionInputGenerator_rtl_0": {"SIMD": 2},
    "MVAU_hls_1": {"PE": 1, "SIMD": 1},
}
FAKE_RESOURCE_LIMITS = {
    "Total LUTs":  53200,
    "FFs":        106400,
    "BRAM (36k)":    140,
    "DSP Blocks":    220,
}


@pytest.fixture(scope="module")
def learner_with_dummy_model():
    """Carrega o modelo dummy e retorna um HardwareLearner pronto."""
    if not os.path.exists(DUMMY_MODEL_PATH):
        pytest.skip(
            f"Modelo dummy não encontrado em {DUMMY_MODEL_PATH}. "
            "Execute o script de criação documentado no topo deste arquivo."
        )
    hl = HardwareLearner()
    hl.load(DUMMY_MODEL_PATH)
    return hl


# ---------------------------------------------------------------------------
# Teste 1: Carregamento do modelo
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_missing_file_raises(self):
        hl = HardwareLearner()
        with pytest.raises(FileNotFoundError):
            hl.load("nao_existe.pkl")

    def test_load_sets_model(self, learner_with_dummy_model):
        assert learner_with_dummy_model.is_loaded()


# ---------------------------------------------------------------------------
# Teste 2: featurize (sem onnx real — apenas smoke test estrutural)
# ---------------------------------------------------------------------------

class TestFeaturize:
    def test_featurize_without_onnx_returns_empty(self):
        """Sem um ONNX válido, featurize deve lançar erro (não silenciar)."""
        with pytest.raises(Exception):
            featurize("nao_existe.onnx", FAKE_FOLDING)

    def test_folding_config_integration(self):
        """Garante que o folding config é um dict serializável."""
        assert isinstance(FAKE_FOLDING, dict)
        assert "Defaults" in FAKE_FOLDING


# ---------------------------------------------------------------------------
# Teste 3: predict — usa um vetor de features sintético injetado diretamente
# ---------------------------------------------------------------------------

class TestPredict:
    def _inject_features(self, learner, n_features=40):
        """
        Monkey-patcha featurize temporariamente para retornar um vetor fixo,
        permitindo testar predict sem um ONNX real.
        """
        import ai.hardware_learner as hl_module
        original = hl_module.featurize

        dummy_vec = np.ones(n_features, dtype=np.float32)
        hl_module.featurize = lambda onnx_path, config: dummy_vec

        return original, hl_module

    def test_predict_returns_list_of_dicts(self, learner_with_dummy_model):
        original_fn, hl_module = self._inject_features(learner_with_dummy_model)
        try:
            results = learner_with_dummy_model.predict("fake.onnx", [FAKE_FOLDING, FAKE_FOLDING])
        finally:
            hl_module.featurize = original_fn

        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, dict)
            for key in RESOURCE_TARGETS:
                assert key in r
                assert r[key] >= 0.0

    def test_predict_empty_returns_empty(self, learner_with_dummy_model):
        results = learner_with_dummy_model.predict("fake.onnx", [])
        assert results == []

    def test_predict_without_load_raises(self):
        hl = HardwareLearner()
        with pytest.raises(RuntimeError, match="não carregado"):
            hl.predict("fake.onnx", [FAKE_FOLDING])


# ---------------------------------------------------------------------------
# Teste 4: calculate_discrepancy
# ---------------------------------------------------------------------------

class TestDiscrepancy:
    def _sample_pred(self):
        return {"Total LUTs": 40000, "FFs": 50000, "BRAM (36k)": 80, "DSP Blocks": 10}

    def _sample_actual(self):
        return {"Total LUTs": 55000, "FFs": 48000, "BRAM (36k)": 90, "DSP Blocks": 12}

    def test_returns_expected_keys(self):
        hl = HardwareLearner()
        disc = hl.calculate_discrepancy(
            self._sample_pred(), self._sample_actual(), FAKE_RESOURCE_LIMITS
        )
        for key in ("absolute", "relative_error", "utilization_pct",
                    "dominant_resource", "dominant_utilization_pct"):
            assert key in disc

    def test_dominant_resource_is_valid(self):
        hl = HardwareLearner()
        disc = hl.calculate_discrepancy(
            self._sample_pred(), self._sample_actual(), FAKE_RESOURCE_LIMITS
        )
        assert disc["dominant_resource"] in RESOURCE_TARGETS

    def test_dominant_is_highest_utilization(self):
        hl = HardwareLearner()
        disc = hl.calculate_discrepancy(
            self._sample_pred(), self._sample_actual(), FAKE_RESOURCE_LIMITS
        )
        dom = disc["dominant_resource"]
        dom_pct = disc["utilization_pct"][dom]
        for res, pct in disc["utilization_pct"].items():
            assert dom_pct >= pct - 1e-6

    def test_absolute_discrepancy_sign(self):
        hl = HardwareLearner()
        disc = hl.calculate_discrepancy(
            self._sample_pred(), self._sample_actual(), FAKE_RESOURCE_LIMITS
        )
        # LUTs: actual(55000) > pred(40000) -> diff > 0
        assert disc["absolute"]["Total LUTs"] > 0
        # FFs: actual(48000) < pred(50000) -> diff < 0
        assert disc["absolute"]["FFs"] < 0

    def test_luts_dominant_when_most_utilized(self):
        """Com LUTs a 103%, deve ser o dominante."""
        pred   = {"Total LUTs": 10000, "FFs": 10000, "BRAM (36k)": 0, "DSP Blocks": 0}
        actual = {"Total LUTs": 55000, "FFs":  5000, "BRAM (36k)": 0, "DSP Blocks": 0}
        hl = HardwareLearner()
        disc = hl.calculate_discrepancy(pred, actual, FAKE_RESOURCE_LIMITS)
        assert disc["dominant_resource"] == "Total LUTs"


# ---------------------------------------------------------------------------
# Teste 5: fine_tune
# ---------------------------------------------------------------------------

class TestFineTune:
    def _inject_features(self, n_features=40):
        import ai.hardware_learner as hl_module
        original = hl_module.featurize
        dummy_vec = np.ones(n_features, dtype=np.float32)
        hl_module.featurize = lambda onnx_path, config: dummy_vec
        return original, hl_module

    def test_fine_tune_adds_to_buffer(self, learner_with_dummy_model):
        hl = learner_with_dummy_model
        initial_len = len(hl._X_buffer)
        original_fn, hl_module = self._inject_features()
        actual = {"Total LUTs": 40000, "FFs": 50000, "BRAM (36k)": 80, "DSP Blocks": 10}
        try:
            hl.fine_tune("fake.onnx", FAKE_FOLDING, actual, current_build_weight=10.0)
        finally:
            hl_module.featurize = original_fn
        assert len(hl._X_buffer) == initial_len + 1

    def test_fine_tune_sets_current_weight(self, learner_with_dummy_model):
        hl = learner_with_dummy_model
        original_fn, hl_module = self._inject_features()
        actual = {"Total LUTs": 40000, "FFs": 50000, "BRAM (36k)": 80, "DSP Blocks": 10}
        try:
            hl.fine_tune("fake.onnx", FAKE_FOLDING, actual, current_build_weight=15.0)
        finally:
            hl_module.featurize = original_fn
        # O último peso deve ser 15.0
        assert hl._weights_buffer[-1] == pytest.approx(15.0)

    def test_fine_tune_without_load_raises(self):
        hl = HardwareLearner()
        actual = {"Total LUTs": 1000, "FFs": 1000, "BRAM (36k)": 5, "DSP Blocks": 0}
        with pytest.raises(RuntimeError, match="não carregado"):
            hl.fine_tune("fake.onnx", FAKE_FOLDING, actual)


# ---------------------------------------------------------------------------
# Teste 6: select_best_config
# ---------------------------------------------------------------------------

class TestSelectBestConfig:
    def _make_predictions(self):
        return [
            {"Total LUTs": 20000, "FFs": 30000, "BRAM (36k)": 50, "DSP Blocks": 5},
            {"Total LUTs": 50000, "FFs": 60000, "BRAM (36k)": 120, "DSP Blocks": 10},
            {"Total LUTs": 80000, "FFs": 90000, "BRAM (36k)": 200, "DSP Blocks": 20},  # excede LUTs
        ]

    def test_selects_highest_fitting_index(self):
        hl = HardwareLearner()
        preds = self._make_predictions()
        folds = [{}, {}, {}]
        idx, _, _ = hl.select_best_config(preds, folds, FAKE_RESOURCE_LIMITS)
        # Índice 2 excede LUTs (80000 > 53200), então deve retornar 1
        assert idx == 1

    def test_returns_minus1_if_none_fits(self):
        hl = HardwareLearner()
        preds = [{"Total LUTs": 99999, "FFs": 99999, "BRAM (36k)": 999, "DSP Blocks": 999}]
        idx, fold, pred = hl.select_best_config(preds, [{}], FAKE_RESOURCE_LIMITS)
        assert idx == -1
        assert fold == {}
        assert pred == {}
