#!/usr/bin/env python3
"""
run_fps_campaign.py — CLI para rodar fps_map em múltiplos request.json

Substitui o fluxo GUI (client_gui → server → run_fps_map_job) por uma
execução direta em linha de comando, dentro do container FINN.

Uso:
    # Rodar um único request:
    python3 run_fps_campaign.py requests/MNIST/req_fps_mnist_1w1a.json

    # Rodar múltiplos requests em sequência:
    python3 run_fps_campaign.py \
        requests/MNIST/req_fps_mnist_1w1a.json \
        requests/MNIST/req_fps_mnist_2w2a.json \
        requests/CIFAR10/req_fps_cifar10_1w1a.json \
        requests/CIFAR10/req_fps_cifar10_2w2a.json \
        requests/SAT6/req_fps_sat6_t2w4.json \
        requests/SAT6/req_fps_sat6_t2w8.json

    # Rodar todos os fps_requests de uma pasta:
    python3 run_fps_campaign.py requests/MNIST/req_fps_*.json

    # Escolher o diretório-pai de saída (padrão: ./fps_campaign_results):
    python3 run_fps_campaign.py --output-dir /tmp/fps_results \
        requests/MNIST/req_fps_mnist_1w1a.json

Saída:
    Para cada request, cria um subdiretório com o nome do model_id:
        <output_dir>/<model_id>/
            ├── request.json         (cópia do request usado)
            ├── fps_map.csv          (pontos run_id × FPS)
            ├── fps_map.png          (gráfico da curva de FPS)
            └── build_*.log          (logs de cada estimativa FINN)

    Ao final, exibe uma tabela resumindo os resultados de todos os datasets.
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Flags / Constantes
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "./fps_campaign_results"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_header(title: str):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def _print_section(title: str):
    print(f"\n--- {title} ---")


def _load_request(request_path: str) -> dict:
    with open(request_path, "r") as f:
        return json.load(f)


def _build_dir_for(output_dir: str, model_id: str, timestamp: str) -> str:
    safe_id = model_id.replace("/", "_").replace(" ", "_")
    return os.path.join(output_dir, f"{safe_id}_{timestamp}")


def _print_summary_table(results: list[dict]):
    _print_header("RESUMO DA CAMPANHA")
    col_w = [20, 12, 10, 45]
    header = (
        f"{'Model ID':<{col_w[0]}} "
        f"{'Status':<{col_w[1]}} "
        f"{'Dur.(s)':<{col_w[2]}} "
        f"{'Build Dir':<{col_w[3]}}"
    )
    print(header)
    print("-" * sum(col_w))
    for r in results:
        status_icon = "✓" if r["status"] == "success" else "✗"
        print(
            f"{r['model_id']:<{col_w[0]}} "
            f"{status_icon} {r['status']:<{col_w[1]-2}} "
            f"{r['duration']:<{col_w[2]}.1f} "
            f"{r['build_dir']:<{col_w[3]}}"
        )

    successes = sum(1 for r in results if r["status"] == "success")
    print(f"\nTotal: {successes}/{len(results)} concluídos com sucesso.")

    # Se houver fps_map.csv, mostra o FPS máximo atingido por dataset
    print()
    for r in results:
        csv_path = os.path.join(r["build_dir"], "fps_map.csv")
        if os.path.exists(csv_path):
            try:
                import csv as csv_mod
                with open(csv_path) as f:
                    rows = list(csv_mod.DictReader(f))
                if rows:
                    fps_col = "estimated_fps"
                    fps_values = [float(row[fps_col]) for row in rows if fps_col in row]
                    if fps_values:
                        print(f"  {r['model_id']:20s} → FPS mín={min(fps_values):.0f}  "
                              f"FPS máx={max(fps_values):.0f}  "
                              f"({len(fps_values)} pontos)")
            except Exception as e:
                print(f"  {r['model_id']:20s} → [Erro ao ler csv: {e}]")


# ---------------------------------------------------------------------------
# Núcleo: executa um único fps_map job
# ---------------------------------------------------------------------------

def run_single_fps_map(request_path: str, output_dir: str) -> dict:
    """
    Prepara o build_dir, copia o request.json e chama generate_map do
    run_fps_map_job.py diretamente (importado como módulo).

    Returns:
        dict com 'model_id', 'status', 'duration', 'build_dir'
    """
    request_path = os.path.abspath(request_path)
    request_data = _load_request(request_path)
    model_id = request_data.get("model_id", "UNKNOWN")
    fpga_part = request_data.get("fpga_part", "xc7z020clg400-1")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    build_dir = _build_dir_for(output_dir, model_id, timestamp)
    os.makedirs(build_dir, exist_ok=True)

    # Copia o request para dentro do build_dir (convenção do run_fps_map_job)
    shutil.copy(request_path, os.path.join(build_dir, "request.json"))

    _print_header(f"Iniciando: {model_id}")
    print(f"  Request:    {request_path}")
    print(f"  Build dir:  {build_dir}")
    print(f"  FPGA:       {fpga_part}")

    # Importa o registry de modelos
    import yaml
    registry_path = os.path.join(os.path.dirname(__file__), "hara", "models", "registry_models.yaml")
    if not os.path.exists(registry_path):
        # Tenta no diretório atual (caso rode de dentro de hara/)
        registry_path = os.path.join(os.path.dirname(__file__), "models", "registry_models.yaml")

    with open(registry_path, "r") as f:
        model_registry = yaml.safe_load(f)

    model_info = model_registry.get(model_id)
    if not model_info:
        print(f"[✗] model_id '{model_id}' não encontrado no registry. Pulando.")
        return {"model_id": model_id, "status": "skipped", "duration": 0.0, "build_dir": build_dir}

    # Importa e chama generate_map
    sys.path.insert(0, os.path.dirname(__file__))
    # Tenta import de dentro de hara/ ou do diretório atual
    try:
        import importlib, importlib.util
        fps_map_candidates = [
            os.path.join(os.path.dirname(__file__), "run_fps_map_job.py"),
            os.path.join(os.path.dirname(__file__), "hara", "run_fps_map_job.py"),
        ]
        fps_map_path = next((p for p in fps_map_candidates if os.path.exists(p)), None)
        if fps_map_path is None:
            raise FileNotFoundError("run_fps_map_job.py não encontrado")

        spec = importlib.util.spec_from_file_location("run_fps_map_job", fps_map_path)
        fps_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fps_mod)
    except Exception as e:
        print(f"[✗] Falha ao importar run_fps_map_job: {e}")
        return {"model_id": model_id, "status": "error", "duration": 0.0, "build_dir": build_dir}

    t_start = time.time()
    status = "error"
    try:
        fps_mod.generate_map(model_info, build_dir, fpga_part)
        status = "success"
        print(f"\n[✓] {model_id} concluído!")
    except Exception as e:
        print(f"\n[✗] {model_id} falhou: {e}")

    duration = time.time() - t_start
    return {"model_id": model_id, "status": status, "duration": duration, "build_dir": build_dir}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Campanha de fps_map para múltiplos datasets (MNIST, CIFAR10, SAT6, ...).",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Exemplos:
  # Rodar dois requests:
  python3 run_fps_campaign.py requests/MNIST/req_fps_mnist_1w1a.json requests/CIFAR10/req_fps_cifar10_1w1a.json

  # Todos os fps_requests do MNIST:
  python3 run_fps_campaign.py requests/MNIST/req_fps_*.json

  # Campanha completa MNIST + CIFAR10 + SAT6:
  python3 run_fps_campaign.py \\
      requests/MNIST/req_fps_mnist_1w1a.json \\
      requests/MNIST/req_fps_mnist_2w2a.json \\
      requests/CIFAR10/req_fps_cifar10_1w1a.json \\
      requests/CIFAR10/req_fps_cifar10_2w2a.json \\
      requests/SAT6/req_fps_sat6_t2w4.json \\
      requests/SAT6/req_fps_sat6_t2w8.json
"""
    )
    parser.add_argument(
        "requests",
        nargs="+",
        metavar="REQUEST_JSON",
        help="Um ou mais arquivos request.json a processar."
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Diretório-pai para os build dirs (padrão: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Interrompe a campanha se algum job falhar."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    _print_header(f"CAMPANHA FPS MAP — {len(args.requests)} request(s)")
    print(f"  Output dir: {os.path.abspath(args.output_dir)}")
    print(f"  Requests:   {len(args.requests)}")
    for rp in args.requests:
        print(f"    - {rp}")

    all_results = []
    for i, request_path in enumerate(args.requests, 1):
        print(f"\n[{i}/{len(args.requests)}] Processando: {request_path}")

        if not os.path.exists(request_path):
            print(f"[✗] Arquivo não encontrado: {request_path}. Pulando.")
            all_results.append({
                "model_id": os.path.basename(request_path),
                "status": "not_found",
                "duration": 0.0,
                "build_dir": "-"
            })
            continue

        result = run_single_fps_map(request_path, args.output_dir)
        all_results.append(result)

        if args.stop_on_error and result["status"] not in ("success", "skipped"):
            print("\n[!] --stop-on-error ativado. Interrompendo campanha.")
            break

    _print_summary_table(all_results)


if __name__ == "__main__":
    main()
