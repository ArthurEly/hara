#!/bin/bash
set -ex
echo "--- [DENTRO DO CONTÊINER] fps_map_service.sh iniciado ---"
REQUEST_FILE_PATH="$1"
BUILD_DIR_PATH="$2"
HARA_DIR="/home/arthurely/Desktop/finn_chi2p"
cd "$HARA_DIR"
echo "--- [DENTRO DO CONTÊINER] Executando run_fps_map_job.py... ---"
python3 ./hara/run_fps_map_job.py --request "$REQUEST_FILE_PATH" --build_dir "$BUILD_DIR_PATH"
echo "--- [DENTRO DO CONTÊINER] run_fps_map_job.py finalizado ---"