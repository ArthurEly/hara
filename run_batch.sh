#!/bin/bash

echo "=========================================================="
echo "Iniciando bateria de builds para coleta de depths (FIFOs)"
echo "=========================================================="

echo "Executando: SAT6_T2W2 (Auto)"
python run_exhaustive_hardware_v1.py --request requests/SAT6/pynq_greedy_req_fps_sat6_t2w2.json

echo "=========================================================="
echo "Todos os builds foram concluídos com sucesso!"
echo "=========================================================="
