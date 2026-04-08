"""
orchestrator.py

HARAv2 — Orquestrador central do co-design de redes neurais e aceleradores
FPGA sob restrições definidas pelo usuário.

Fluxo principal (paper HARAv2):
  SETUP
    ↓
  FIRST-SHOT (encontra configuração viável para T* e A*)
    ↓
  GREEDY SEARCH (usa área restante para aumentar throughput)
    ↓
  VERIFICATION (build real via Vivado)
    ├─ sucesso → reporta resultado
    └─ falha   → RECOVERY (reduz paralelismo ou poda)

Módulos coordenados:
  - HardwareExplorer   (hardware_explorer.py)
  - MultiModuleLearner (ai/multi_module_learner.py)
  - ModelOptimizer     (model_optimizer.py) — stub, ativado quando disponível

CLI:
  python3 orchestrator.py \\
    --onnx <step_generate_estimate_reports.onnx> \\
    --target-fps 5000 \\
    --area-budget 1.0 \\
    [--simulate]
"""

import os
import sys
import json
import copy
import argparse
from datetime import datetime

# ---------------------------------------------------------------------------
# Importações dos módulos HARA
# ---------------------------------------------------------------------------

# Adiciona hara/ ao path para imports relativos
_HARA_ROOT = os.path.dirname(os.path.abspath(__file__))
if _HARA_ROOT not in sys.path:
    sys.path.insert(0, _HARA_ROOT)

from hardware_explorer import HardwareExplorer
from config import BUILD_CONFIG, HARA_LOOP_CONFIG

try:
    from ai.multi_module_learner import MultiModuleLearner
    _MML_AVAILABLE = True
except ImportError:
    _MML_AVAILABLE = False
    print("[Orchestrator] AVISO: MultiModuleLearner não disponível.")

try:
    from ai.predict_fifo_depths import predict_fifo_depths, inject_fifo_depths
    _FIFO_PRED_AVAILABLE = True
except ImportError:
    _FIFO_PRED_AVAILABLE = False

# FPGA padrão: PYNQ-Z1 (Zynq XC7Z020)
DEFAULT_FPGA_LIMITS = {
    "Total LUTs": 53200,
    "FFs":        106400,
    "BRAM (36k)": 140,
    "DSP Blocks": 220,
}

# Razão máxima de pruning (85% per paper)
MAX_PRUNING_RATIO = 0.85

# Gamma para recovery (paper: γ=2)
GAMMA = 2.0


# ===========================================================================
# Orchestrator
# ===========================================================================

class HARAv2Orchestrator:
    """
    Orquestrador central do HARAv2.

    Args:
        onnx_path        : caminho para step_generate_estimate_reports.onnx
        build_dir        : diretório raiz dos builds de hardware
        target_fps       : T* (0 = sem restrição)
        area_budget      : A* como fração do FPGA (ex: 0.25 = 25%)
        target_accuracy  : α* (mínimo aceitável de accuracy)
        fpga_limits      : recursos físicos máximos do FPGA
        models_dir       : dir com pkl de modelos por módulo
        simulate         : usa _run_fake_build em vez de Vivado real
        fpga_part        : string da part Xilinx
        topology_id      : id da topologia (ex: "CIFAR10_CNV")
        quant            : bitwidth (ex: 2)
    """

    def __init__(self,
                 onnx_path: str,
                 build_dir: str,
                 target_fps: float = 0.0,
                 area_budget: float = 1.0,
                 target_accuracy: float = 0.0,
                 fpga_limits: dict | None = None,
                 models_dir: str | None = None,
                 simulate: bool = False,
                 fpga_part: str = "xc7z020clg400-1",
                 topology_id: str = "unknown",
                 quant: int | None = None,
                 fixed_resources: dict | None = None):

        self.onnx_path        = onnx_path
        self.build_dir        = build_dir
        self.target_fps       = target_fps
        self.area_budget      = area_budget
        self.target_accuracy  = target_accuracy
        self.fpga_part        = fpga_part
        self.topology_id      = topology_id
        self.quant            = quant
        self.simulate         = simulate
        self.fixed_resources  = fixed_resources or {}

        # Limites de hardware
        self.fpga_limits = fpga_limits or DEFAULT_FPGA_LIMITS
        self.resource_limits = {
            k: int(v * area_budget) for k, v in self.fpga_limits.items()
        }

        # Pruning state
        self._current_pruning_ratio = 0.0
        self._current_onnx          = onnx_path

        # MultiModuleLearner
        if _MML_AVAILABLE and models_dir and os.path.isdir(models_dir):
            self.learner = MultiModuleLearner(models_dir)
        else:
            self.learner = None
            if not _MML_AVAILABLE:
                print("[Orchestrator] Rodando sem MultiModuleLearner.")

        # HardwareExplorer
        self.explorer = HardwareExplorer(
            build_dir=build_dir,
            config=BUILD_CONFIG,
            resource_limits=self.resource_limits,
            hara_loop_config=HARA_LOOP_CONFIG,
            simulation_mode=simulate,
            fixed_resources=self.fixed_resources,
            fpga_part=fpga_part,
            hardware_learner=self.learner,
        )

        # Estado do melhor resultado
        self.best_folding   = None
        self.best_resources = {}
        self.best_fps       = 0.0
        self.n_builds       = 0

        os.makedirs(build_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Utilidades
    # -----------------------------------------------------------------------

    def _resource_fits(self, predicted: dict) -> bool:
        """Verifica se os recursos preditos cabem no budget A*."""
        mappings = [
            ("Total LUTs",  self.resource_limits.get("Total LUTs", 1e9)),
            ("FFs",         self.resource_limits.get("FFs",         1e9)),
            ("BRAM (36k)",  self.resource_limits.get("BRAM (36k)",  1e9)),
            ("DSP Blocks",  self.resource_limits.get("DSP Blocks",  1e9)),
        ]
        for res, limit in mappings:
            if predicted.get(res, 0) > limit:
                return False
        return True

    def _predict_resources(self, folding: dict) -> dict:
        """
        Prediz recursos totais do accelerator para um folding.
        Injeta FIFO depths antes de predizer se o predictor estiver disponível.
        """
        if self.learner is None:
            return {}

        # Injeta depths previstas no folding para melhor estimativa de BRAM
        working_folding = folding
        if _FIFO_PRED_AVAILABLE:
            try:
                depths = predict_fifo_depths(self._current_onnx, folding)
                working_folding = inject_fifo_depths(folding, depths)
            except Exception as e:
                print(f"[Orchestrator] FIFOPredictor falhou (non-critical): {e}")

        preds = self.learner.predict(self._current_onnx, [working_folding])
        return preds[0] if preds else {}

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")

    def _save_state(self, tag: str = ""):
        """Salva o estado atual do melhor folding."""
        if self.best_folding is None:
            return
        out = {
            "best_folding":   self.best_folding,
            "best_resources": self.best_resources,
            "best_fps":       self.best_fps,
            "n_builds":       self.n_builds,
            "area_budget":    self.area_budget,
            "target_fps":     self.target_fps,
        }
        path = os.path.join(self.build_dir, f"orchestrator_state{tag}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        self._log(f"Estado salvo: {path}")

    # -----------------------------------------------------------------------
    # Analytical design space
    # -----------------------------------------------------------------------

    def _map_design_space(self) -> list[dict]:
        """Gera o espaço de foldings via FinnCycleEstimator (análitico, sem build)."""
        configs = self.explorer._map_theoretical_design_space(self._current_onnx)
        return configs  # lista de {'folding': dict, 'fps_estimated': float}

    # -----------------------------------------------------------------------
    # FIRST-SHOT
    # -----------------------------------------------------------------------

    def _first_shot(self, design_space: list[dict]) -> dict | None:
        """
        Encontra o primeiro folding que satisfaz T* e A*.

        1. Filtra candidatos com fps_estimated >= T*
        2. Prediz recursos e descarta os que excedem A*
        3. Pega o de maior FPS → faz o build real
        4. Se falhar → retorna None (trigger recovery/pruning)
        """
        self._log("=== FIRST-SHOT ===")

        # Filtra por T*
        candidates = [c for c in design_space
                      if c["fps_estimated"] >= self.target_fps]
        if not candidates:
            self._log(f"Nenhum folding atinge T*={self.target_fps:.0f} FPS. "
                      "Usando todos os candidatos.")
            candidates = design_space

        self._log(f"Candidatos após filtro T*: {len(candidates)}")

        # Filtra por A* com predictor
        if self.learner and self.learner.is_loaded():
            self._log("Filtrando por área predita (A*)...")
            feasible = []
            for c in candidates:
                pred = self._predict_resources(c["folding"])
                if self._resource_fits(pred):
                    feasible.append((c, pred))

            self._log(f"Candidatos viáveis por área: {len(feasible)}")
            if not feasible:
                self._log("[!] Nenhum candidato cabe no budget A*. "
                          "Considere aumentar A* ou podar a rede.")
                return None

            # Pega o de maior FPS estimado (último é o maior, lista é crescente)
            best_c, best_pred = feasible[-1]
        else:
            self._log("[AVISO] Learner não disponível. Usando candidato de maior FPS.")
            best_c    = candidates[-1]
            best_pred = {}

        self._log(f"First-shot candidate: FPS≈{best_c['fps_estimated']:.0f} | "
                  f"Pred LUT={best_pred.get('Total LUTs',0):,}")

        # Build real
        result = self._build(best_c["folding"], tag="first_shot")
        if result and result.get("status") == "success":
            self._log(f"[✓] First-shot sucesso!")
            return best_c["folding"]

        self._log("[!] First-shot falhou no hardware real.")
        return None

    # -----------------------------------------------------------------------
    # GREEDY SEARCH
    # -----------------------------------------------------------------------

    def _greedy_search(self, start_folding: dict,
                        design_space: list[dict]) -> dict:
        """
        A partir do first-shot, sobe no design space até o limite de A*.
        Para quando a predição indicar que excede o budget.
        Retorna o melhor folding encontrado.
        """
        self._log("=== GREEDY SEARCH ===")
        best = start_folding

        # Encontra o índice do first-shot no design space
        start_idx = 0
        for i, c in enumerate(design_space):
            if c["folding"] == start_folding:
                start_idx = i
                break

        for i in range(start_idx + 1, len(design_space)):
            c = design_space[i]
            pred = self._predict_resources(c["folding"])

            if not self._resource_fits(pred):
                self._log(f"  Step {i}: FPS≈{c['fps_estimated']:.0f} → excede área. Parando.")
                break

            luts = pred.get("Total LUTs", 0)
            ffs = pred.get("FFs", 0)
            bram = pred.get("BRAM (36k)", 0.0)
            dsp = pred.get("DSP Blocks", 0)
            
            self._log(f"  Step {i}: FPS≈{c['fps_estimated']:.0f} → dentro do budget. "
                      f"LUT={luts:,} | FF={ffs:,} | BRAM={bram:.1f} | DSP={dsp:,}")
            best = c["folding"]

        self._log(f"Greedy terminou. Candidato final selecionado.")
        return best

    # -----------------------------------------------------------------------
    # BUILD
    # -----------------------------------------------------------------------

    def _build(self, folding: dict, tag: str = "") -> dict | None:
        """Executa um build real via HardwareExplorer._run_single_build."""
        self.n_builds += 1
        hw_name = f"hara_v2_{tag}_build{self.n_builds}"

        folding_path = os.path.join(self.build_dir, f"{hw_name}.json")
        with open(folding_path, "w") as f:
            json.dump(folding, f, indent=2)

        result = self.explorer._run_single_build(
            onnx_model_path=self._current_onnx,
            hw_name=hw_name,
            quant=self.quant,
            topology_id=self.topology_id,
            steps=BUILD_CONFIG.get("hara_build", {}).get("steps", []),
            folding_path=folding_path,
            resource_limits=self.resource_limits,
        )

        if result and result.get("status") == "success":
            self.best_folding   = folding
            # Extrai FPS real do report se disponível
            perf_path = os.path.join(
                result.get("build_dir", ""),
                "report", "estimate_network_performance.json"
            )
            if os.path.exists(perf_path):
                with open(perf_path) as f:
                    perf = json.load(f)
                self.best_fps = perf.get("estimated_throughput_fps", 0)

        return result

    # -----------------------------------------------------------------------
    # RECOVERY
    # -----------------------------------------------------------------------

    def _phi_T(self) -> float:
        """Surplus de throughput atual."""
        if self.target_fps <= 0:
            return 1.0
        return self.best_fps / self.target_fps

    def _phi_alpha(self, current_accuracy: float) -> float:
        """Surplus de accuracy ponderado (γ=2)."""
        if self.target_accuracy <= 0:
            return 1.0
        return (current_accuracy / self.target_accuracy) ** GAMMA

    def _recovery(self, failed_folding: dict,
                   design_space: list[dict],
                   current_accuracy: float = 1.0) -> dict | None:
        """
        Procedimento de recovery após falha de síntese.

        T*=0: reduz paralelismo ao mínimo (index 0).
        T*>0: compara φ_T vs φ_α e decide entre pruning ou redução de paralelismo.
        """
        self._log("=== RECOVERY ===")

        if self.target_fps <= 0:
            self._log("T*=0 → reduzindo ao mínimo de paralelismo.")
            return design_space[0]["folding"]

        phi_t = self._phi_T()
        phi_a = self._phi_alpha(current_accuracy)
        self._log(f"φ_T={phi_t:.3f}  φ_α={phi_a:.3f} (γ={GAMMA})")

        if phi_a > phi_t:
            self._log("φ_α > φ_T → Pruning recomendado.")
            return self._prune_and_restart(design_space)
        else:
            self._log("φ_T >= φ_α → Reduzindo paralelismo.")
            # Volta um passo no design space
            for i, c in enumerate(design_space):
                if c["folding"] == failed_folding and i > 0:
                    return design_space[i - 1]["folding"]
            return design_space[0]["folding"]

    def _prune_and_restart(self, design_space: list[dict]) -> dict | None:
        """
        Stub: delega para ModelOptimizer (quando disponível).
        Por enquanto apenas loga e retorna None para sinalizar falha.
        """
        new_ratio = self._current_pruning_ratio + 0.05
        if new_ratio > MAX_PRUNING_RATIO:
            self._log("[!] Limite máximo de pruning atingido (85%). "
                      "Nenhuma configuração viável encontrada.")
            return None

        self._current_pruning_ratio = new_ratio
        self._log(f"[STUB] ModelOptimizer.prune({new_ratio*100:.0f}%) — "
                  "integração com Brevitas requer Docker/dataset.")
        self._log("       Após podar, re-execute o pipeline com o novo ONNX.")
        # Em uma integração completa, aqui chamaria:
        #   new_onnx = model_optimizer.prune(self._current_onnx, ratio=new_ratio)
        #   self._current_onnx = new_onnx
        #   return design_space[0]["folding"]
        return None

    # -----------------------------------------------------------------------
    # ENTRY POINT
    # -----------------------------------------------------------------------

    def run(self, current_accuracy: float = 1.0):
        """
        Executa o fluxo completo HARAv2.

        Args:
            current_accuracy: accuracy atual do modelo (0–1)
        """
        self._log("=" * 60)
        self._log("HARAv2 — Co-Design Flow")
        self._log(f"  ONNX:         {self._current_onnx}")
        self._log(f"  T*:           {self.target_fps:.0f} FPS")
        self._log(f"  A*:           {self.area_budget*100:.0f}% do FPGA")
        self._log(f"  α*:           {self.target_accuracy*100:.1f}%")
        self._log(f"  Simulate:     {self.simulate}")
        self._log("=" * 60)

        # --- 1. Mapeamento analítico do design space ---
        self._log("\n[FASE 1] Mapeando design space (análitico)...")
        design_space = self._map_design_space()

        if not design_space:
            self._log("[ERRO] Design space vazio. Verifique o ONNX.")
            return

        self._log(f"  {len(design_space)} configurações mapeadas. "
                  f"FPS range: [{design_space[0]['fps_estimated']:.0f}, "
                  f"{design_space[-1]['fps_estimated']:.0f}]")

        # --- 2. First-shot ---
        self._log("\n[FASE 2] First-Shot...")
        first_folding = self._first_shot(design_space)

        if first_folding is None:
            self._log("[FASE 2] First-shot falhou. Tentando recovery/pruning...")
            recovered = self._recovery(
                failed_folding=design_space[-1]["folding"],
                design_space=design_space,
                current_accuracy=current_accuracy,
            )
            if recovered is None:
                self._log("[ERRO] Recovery falhou. Encerrando.")
                return
            first_folding = recovered

        # --- 3. Greedy Search ---
        self._log("\n[FASE 3] Greedy Search...")
        best_greedy_folding = self._greedy_search(first_folding, design_space)

        # --- 4. Build final (verification) ---
        self._log("\n[FASE 4] Build final (Verificação)...")
        result = self._build(best_greedy_folding, tag="final")

        if result and result.get("status") == "success":
            self._log("\n✅ HARAv2 concluído com sucesso!")
            self._log(f"   FPS:  {self.best_fps:.0f}")
            self._log(f"   Builds totais: {self.n_builds}")
            self._save_state("_final")
        else:
            self._log("\n[FASE 4] Build final falhou. Iniciando recovery...")
            recovered = self._recovery(
                failed_folding=best_greedy_folding,
                design_space=design_space,
                current_accuracy=current_accuracy,
            )
            if recovered:
                self._log("Tentando build com folding reduzido...")
                final = self._build(recovered, tag="recovered")
                if final and final.get("status") == "success":
                    self._log("✅ Recovery bem-sucedido!")
                    self._save_state("_recovered")
                else:
                    self._log("[ERRO] Recovery não conseguiu produzir um build válido.")
            else:
                self._log("[ERRO] Recovery retornou None. Encerrando.")

        self._log(f"\nTotal de builds realizados: {self.n_builds}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HARAv2 — Co-Design FPGA/NN",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--onnx",        required=True,
                        help="Caminho para step_generate_estimate_reports.onnx")
    parser.add_argument("--build-dir",   default="hw/builds/harav2",
                        help="Diretório raiz dos builds")
    parser.add_argument("--target-fps",  type=float, default=0.0,
                        help="T* — throughput alvo (FPS). 0 = sem restrição")
    parser.add_argument("--area-budget", type=float, default=1.0,
                        help="A* — fração do FPGA (0–1). Ex: 0.25 = 25%%")
    parser.add_argument("--target-acc",  type=float, default=0.0,
                        help="α* — accuracy mínima (0–1)")
    parser.add_argument("--models-dir",  default=None,
                        help="Diretório com os modelos XGBoost por módulo (.pkl)")
    parser.add_argument("--topology-id", default="unknown",
                        help="ID da topologia (ex: CIFAR10_CNV)")
    parser.add_argument("--quant",       type=int, default=None,
                        help="Bitwidth (ex: 2)")
    parser.add_argument("--simulate",    action="store_true",
                        help="Usa builds simulados (sem Vivado real)")
    parser.add_argument("--accuracy",    type=float, default=1.0,
                        help="Accuracy atual do modelo (0–1)")
    args = parser.parse_args()

    # Models dir padrão: relativo ao hara/
    models_dir = args.models_dir
    if models_dir is None:
        models_dir = os.path.join(_HARA_ROOT, "ai", "retrieval", "results", "trained_models")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    build_dir = os.path.join(args.build_dir, f"run_{ts}")

    orch = HARAv2Orchestrator(
        onnx_path=args.onnx,
        build_dir=build_dir,
        target_fps=args.target_fps,
        area_budget=args.area_budget,
        target_accuracy=args.target_acc,
        models_dir=models_dir,
        simulate=args.simulate,
        topology_id=args.topology_id,
        quant=args.quant,
    )

    orch.run(current_accuracy=args.accuracy)


if __name__ == "__main__":
    main()
