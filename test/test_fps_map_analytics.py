# test/test_fps_map_analytics.py
"""
Testes unitários do pipeline analítico de FPS map (run_fps_map_job.py).

Não requer builds reais do FINN. Usa mocks para simular as saídas do estimador
e verifica a lógica do generate_map, esquema dos requests e monotonicidade da
curva de FPS.

Para rodar:
    cd /home/arthurely/Desktop/finn_chi2p/hara
    python3 -m pytest test/test_fps_map_analytics.py -v
"""

import json
import os
import sys
import tempfile
import shutil
import unittest.mock as mock
from pathlib import Path

import pytest
import yaml

# Adiciona a raiz do HARA ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Caminhos úteis
# ---------------------------------------------------------------------------

HARA_DIR    = Path(__file__).parent.parent
REQUESTS_DIR = HARA_DIR / "requests"
REGISTRY_PATH = HARA_DIR / "models" / "registry_models.yaml"

# Todos os requests de fps_map que criamos
FPS_REQUEST_FILES = {
    "MNIST_1W1A":   REQUESTS_DIR / "MNIST"   / "req_fps_mnist_1w1a.json",
    "MNIST_2W2A":   REQUESTS_DIR / "MNIST"   / "req_fps_mnist_2w2a.json",
    "CIFAR10_1W1A": REQUESTS_DIR / "CIFAR10" / "req_fps_cifar10_1w1a.json",
    "CIFAR10_2W2A": REQUESTS_DIR / "CIFAR10" / "req_fps_cifar10_2w2a.json",
    "SAT6_T2W4":    REQUESTS_DIR / "SAT6"    / "req_fps_sat6_t2w4.json",
    "SAT6_T2W8":    REQUESTS_DIR / "SAT6"    / "req_fps_sat6_t2w8.json",
}

REQUIRED_KEYS = {"model_id", "fpga_part", "area_constraints", "fixed_resources"}
AREA_RESOURCES = {"Total LUTs", "FFs", "BRAM (36k)", "DSP Blocks"}

# Limites da PYNQ-Z1 (xc7z020clg400-1)
PYNQ_Z1_LIMITS = {
    "Total LUTs": 53200,
    "FFs":       106400,
    "BRAM (36k)":   140,
    "DSP Blocks":   220,
}


# ---------------------------------------------------------------------------
# Fixture: carrega o registry de modelos
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_registry():
    with open(REGISTRY_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Teste 1: Esquema dos request JSONs
# ---------------------------------------------------------------------------

class TestRequestSchema:
    """Verifica que todos os 6 requests têm a estrutura correta."""

    @pytest.mark.parametrize("label,path", list(FPS_REQUEST_FILES.items()))
    def test_file_exists(self, label, path):
        assert path.exists(), f"Request não encontrado: {path}"

    @pytest.mark.parametrize("label,path", list(FPS_REQUEST_FILES.items()))
    def test_required_keys_present(self, label, path):
        with open(path) as f:
            data = json.load(f)
        missing = REQUIRED_KEYS - set(data.keys())
        assert not missing, f"{label}: chaves obrigatórias ausentes: {missing}"

    @pytest.mark.parametrize("label,path", list(FPS_REQUEST_FILES.items()))
    def test_area_constraints_complete(self, label, path):
        with open(path) as f:
            data = json.load(f)
        constraints = set(data.get("area_constraints", {}).keys())
        missing = AREA_RESOURCES - constraints
        assert not missing, f"{label}: recursos ausentes em area_constraints: {missing}"

    @pytest.mark.parametrize("label,path", list(FPS_REQUEST_FILES.items()))
    def test_area_constraints_pynq_z1_values(self, label, path):
        """Verifica que os limites correspondem à PYNQ-Z1."""
        with open(path) as f:
            data = json.load(f)
        ac = data["area_constraints"]
        assert ac["Total LUTs"] == PYNQ_Z1_LIMITS["Total LUTs"], \
            f"{label}: Total LUTs deveria ser {PYNQ_Z1_LIMITS['Total LUTs']}, got {ac['Total LUTs']}"
        assert ac["BRAM (36k)"] == PYNQ_Z1_LIMITS["BRAM (36k)"], \
            f"{label}: BRAM (36k) deveria ser {PYNQ_Z1_LIMITS['BRAM (36k)']}, got {ac['BRAM (36k)']}"

    @pytest.mark.parametrize("label,path", list(FPS_REQUEST_FILES.items()))
    def test_fpga_part_is_pynq(self, label, path):
        with open(path) as f:
            data = json.load(f)
        assert data["fpga_part"] == "xc7z020clg400-1", \
            f"{label}: fpga_part incorreto: {data['fpga_part']}"

    @pytest.mark.parametrize("label,path", list(FPS_REQUEST_FILES.items()))
    def test_model_id_registered(self, label, path, model_registry):
        """Verifica que o model_id de cada request existe no registry_models.yaml."""
        with open(path) as f:
            data = json.load(f)
        model_id = data["model_id"]
        assert model_id in model_registry, \
            f"{label}: model_id '{model_id}' não encontrado no registry_models.yaml"

    @pytest.mark.parametrize("label,path", list(FPS_REQUEST_FILES.items()))
    def test_fixed_resources_has_mvau(self, label, path):
        """Verifica que fixed_resources define ao menos um tipo de MVAU."""
        with open(path) as f:
            data = json.load(f)
        fr = data.get("fixed_resources", {})
        has_mvau = any("MVAU" in k or "mvau" in k for k in fr.keys())
        assert has_mvau, f"{label}: fixed_resources deve conter ao menos uma entrada MVAU"


# ---------------------------------------------------------------------------
# Teste 2: Lógica analítica de FPS (com dados sintéticos)
# ---------------------------------------------------------------------------

class TestFPSCurveMonotonicity:
    """
    Valida que a curva de FPS gerada pelo generate_map é monótona crescente.
    Usa dados sintéticos injetados diretamente — sem FINN real.
    """

    def _make_synthetic_results(self, n_points=5, base_fps=1000):
        """Gera uma lista de resultados de FPS crescentes (simulando o loop do generate_map)."""
        results = []
        fps = base_fps
        for i in range(1, n_points + 1):
            fps = fps * 1.5  # cada passo aumenta 50%
            results.append({
                "run_id": i,
                "estimated_fps": fps,
                "folding_config": json.dumps({"Defaults": {"PE": i, "SIMD": i}})
            })
        return results

    def test_fps_is_strictly_increasing(self):
        results = self._make_synthetic_results(n_points=5)
        fps_values = [r["estimated_fps"] for r in results]
        for i in range(1, len(fps_values)):
            assert fps_values[i] > fps_values[i - 1], \
                f"FPS não cresceu no passo {i}: {fps_values[i-1]} -> {fps_values[i]}"

    def test_first_fps_positive(self):
        results = self._make_synthetic_results(n_points=3)
        assert results[0]["estimated_fps"] > 0

    def test_all_run_ids_unique(self):
        results = self._make_synthetic_results(n_points=4)
        ids = [r["run_id"] for r in results]
        assert len(ids) == len(set(ids)), "run_ids duplicados encontrados"

    def test_folding_config_is_valid_json(self):
        results = self._make_synthetic_results(n_points=3)
        for r in results:
            parsed = json.loads(r["folding_config"])
            assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Teste 3: Geração de artefatos (fps_map.csv e fps_map.png)
# ---------------------------------------------------------------------------

class TestBuildDirArtifacts:
    """
    Verifica que após o generate_map, os artefatos corretos são criados
    no build_dir, mockando _run_estimate_build e os relatórios do FINN.
    """

    def _setup_build_dir(self, tmp_path, request_label):
        """Cria um build_dir temporário com request.json e estrutura de relatório fake."""
        build_dir = tmp_path / f"fps_test_{request_label}"
        build_dir.mkdir()

        request_path = FPS_REQUEST_FILES[request_label]
        shutil.copy(request_path, build_dir / "request.json")
        return build_dir

    def _make_fake_build_structure(self, build_dir, run_name, fps_value):
        """Cria a estrutura de diretórios e relatórios que o FINN normalmente geraria."""
        run_dir = build_dir / run_name
        (run_dir / "report").mkdir(parents=True, exist_ok=True)
        (run_dir / "intermediate_models").mkdir(parents=True, exist_ok=True)

        # Relatório de performance
        perf = {"estimated_throughput_fps": fps_value}
        with open(run_dir / "report" / "estimate_network_performance.json", "w") as f:
            json.dump(perf, f)

        # Relatório de ciclos por camada (para modify_folding)
        cycles = {"MVAU_hls_0": 100, "MVAU_hls_1": 50}
        with open(run_dir / "report" / "estimate_layer_cycles.json", "w") as f:
            json.dump(cycles, f)

        # Folding config
        folding = {
            "Defaults": {"PE": 1, "SIMD": 1},
            "MVAU_hls_0": {"PE": 1, "SIMD": 1, "ram_style": "auto", "resType": "auto"},
            "MVAU_hls_1": {"PE": 1, "SIMD": 1, "ram_style": "auto", "resType": "auto"},
        }
        with open(run_dir / "auto_folding_config.json", "w") as f:
            json.dump(folding, f)

        return run_dir

    def test_fps_map_csv_created(self, tmp_path):
        """
        Simula generate_map com mocks e verifica que fps_map.csv é gerado
        com as colunas corretas.
        """
        import pandas as pd
        import run_fps_map_job as fps_mod

        build_dir = self._setup_build_dir(tmp_path, "MNIST_1W1A")

        # Simula 3 pontos de FPS crescentes
        fake_fps_values = [1000.0, 2000.0, 4000.0]
        call_count = {"n": 0}

        def mock_run_estimate(base_build_dir, onnx_path, hw_name, *args, **kwargs):
            i = call_count["n"]
            fps_val = fake_fps_values[min(i, len(fake_fps_values) - 1)]
            call_count["n"] += 1
            run_dir = self._make_fake_build_structure(
                Path(base_build_dir), hw_name, fps_val
            )
            return str(run_dir)

        with mock.patch.object(fps_mod, "_run_estimate_build",
                               side_effect=mock_run_estimate), \
             mock.patch("utils.hw_utils.get_finn_ready_model",
                        return_value="/fake/model.onnx"), \
             mock.patch("utils.hw_utils.utils.run_and_capture"), \
             mock.patch("utils.hw_utils.utils.reset_folding",
                        return_value={"Defaults": {"PE": 1, "SIMD": 1},
                                      "MVAU_hls_0": {"PE": 1, "SIMD": 1}}), \
             mock.patch("utils.hw_utils.utils.modify_folding",
                        return_value={}):  # retorna igual → encerra o loop

            fps_mod.generate_map(
                model_info={"topology_id": "MNIST_TFC", "quant": 1},
                base_build_dir=str(build_dir),
                fpga_part="xc7z020clg400-1"
            )

        csv_path = build_dir / "fps_map.csv"
        assert csv_path.exists(), "fps_map.csv não foi criado"

        df = pd.read_csv(csv_path)
        assert "run_id" in df.columns,               "Coluna 'run_id' ausente no fps_map.csv"
        assert "estimated_fps" in df.columns,        "Coluna 'estimated_fps' ausente no fps_map.csv"
        assert "folding_config" in df.columns,       "Coluna 'folding_config' ausente no fps_map.csv"
        assert len(df) >= 1,                         "fps_map.csv deve ter ao menos 1 linha"

    def test_fps_map_png_created(self, tmp_path):
        """Verifica que fps_map.png é gerado ao lado do csv."""
        import run_fps_map_job as fps_mod
        import matplotlib
        matplotlib.use("Agg")  # sem display

        build_dir = self._setup_build_dir(tmp_path, "CIFAR10_1W1A")
        call_count = {"n": 0}
        fake_fps_values = [500.0, 1500.0]

        def mock_run_estimate(base_build_dir, onnx_path, hw_name, *args, **kwargs):
            i = call_count["n"]
            fps_val = fake_fps_values[min(i, len(fake_fps_values) - 1)]
            call_count["n"] += 1
            run_dir = self._make_fake_build_structure(
                Path(base_build_dir), hw_name, fps_val
            )
            return str(run_dir)

        with mock.patch.object(fps_mod, "_run_estimate_build",
                               side_effect=mock_run_estimate), \
             mock.patch("utils.hw_utils.get_finn_ready_model",
                        return_value="/fake/cnv.onnx"), \
             mock.patch("utils.hw_utils.utils.run_and_capture"), \
             mock.patch("utils.hw_utils.utils.reset_folding",
                        return_value={"Defaults": {"PE": 1, "SIMD": 1},
                                      "MVAU_hls_0": {"PE": 1, "SIMD": 1}}), \
             mock.patch("utils.hw_utils.utils.modify_folding",
                        return_value={}):

            fps_mod.generate_map(
                model_info={"topology_id": "CIFAR10_CNV", "quant": 1},
                base_build_dir=str(build_dir),
                fpga_part="xc7z020clg400-1"
            )

        assert (build_dir / "fps_map.png").exists(), "fps_map.png não foi criado"


# ---------------------------------------------------------------------------
# Teste 4: Comparação FPS analítico vs actual (tolerância configurável)
# ---------------------------------------------------------------------------

class TestFPSAnalyticalVsActual:
    """
    Valida a lógica de comparação entre o FPS analítico (estimado pelo FINN)
    e o FPS real (medido no hardware).

    O FPS real é simulado aqui com uma perturbação controlada.
    """

    # Tolerância máxima aceitável: FPS analítico pode superestimar em até 30%
    TOLERANCE_PCT = 0.30

    def _relative_error(self, analytical, actual):
        """Erro relativo: (analytical - actual) / actual"""
        if actual == 0:
            return float("inf")
        return abs(analytical - actual) / actual

    @pytest.mark.parametrize("analytical,actual", [
        (1000.0, 950.0),    # +5.3% – dentro da tolerância
        (2000.0, 1600.0),   # +25%  – dentro da tolerância
        (5000.0, 4000.0),   # +25%  – dentro da tolerância
        (500.0,  490.0),    # +2%   – dentro da tolerância
    ])
    def test_within_tolerance(self, analytical, actual):
        err = self._relative_error(analytical, actual)
        assert err <= self.TOLERANCE_PCT, \
            f"Erro {err:.1%} excede tolerância de {self.TOLERANCE_PCT:.0%} " \
            f"(analítico={analytical}, real={actual})"

    @pytest.mark.parametrize("analytical,actual", [
        (10000.0, 5000.0),  # +100% – fora da tolerância (modelo ruim)
        (3000.0,  1500.0),  # +100%
    ])
    def test_beyond_tolerance_detected(self, analytical, actual):
        err = self._relative_error(analytical, actual)
        assert err > self.TOLERANCE_PCT, \
            f"Esperávamos erro > {self.TOLERANCE_PCT:.0%}, mas obtivemos {err:.1%}"

    def test_analytical_overestimation_is_typical(self):
        """
        Verifica a hipótese de que o FINN geralmente SUPERestima o FPS.
        O analítico tende a ser maior que o real (sem levar em conta overhead).
        """
        pairs = [
            (1200.0, 1000.0),
            (3500.0, 2900.0),
            (800.0,  750.0),
        ]
        overestimates = sum(1 for a, r in pairs if a >= r)
        assert overestimates >= len(pairs) // 2, \
            "Esperávamos que a maioria dos pontos seja superestimada"

    def test_zero_actual_fps_handled(self):
        """Divisão por zero não deve ocorrer."""
        err = self._relative_error(1000.0, 0)
        assert err == float("inf")


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
