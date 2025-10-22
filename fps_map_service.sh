#!/bin/bash
set -ex # Para ver exatamente o que está sendo executado

echo "--- [DENTRO DO CONTÊINER] fps_map_service.sh iniciado ---"

# --- CORREÇÃO AQUI ---
# Apenas um argumento é passado para este script, que é o BUILD_DIR_PATH
BUILD_DIR_PATH="$1" 

# Validação para garantir que o BUILD_DIR_PATH foi recebido
if [ -z "$BUILD_DIR_PATH" ]; then
    echo "ERRO CRÍTICO: BUILD_DIR_PATH não foi recebido pelo script interno do contêiner."
    exit 1
fi

echo " -> BUILD_DIR_PATH recebido: $BUILD_DIR_PATH"

# Navega para o diretório raiz do FINN dentro do contêiner
# A variável FINN_ROOT é definida pelo run-docker.sh
cd "$FINN_ROOT" || exit 1 # Garante que o diretório existe

echo "--- [DENTRO DO CONTÊINER] Executando o script Python para o Mapa de FPS... ---"

# Chama o script Python, passando o BUILD_DIR_PATH
# O script Python (run_fps_map_job.py) é que vai ler o request.json dentro de BUILD_DIR_PATH
python3 hara/run_fps_map_job.py --build_dir "$BUILD_DIR_PATH"

echo "--- [DENTRO DO CONTÊINER] run_fps_map_job.py finalizado ---"