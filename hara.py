#sys.path.append("/home/arthurely/Desktop/finn/notebooks/CIFAR10")
#from cnv import generate_hardware

import os
import torch

import subprocess
from cifar import generate_hardware
import torch
from finn.util.test import get_test_model_trained
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.fold_constants import FoldConstants
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

build_dir = "/home/arthurely/Desktop/finn/hara/builds"
def get_onnx_model(act,quant):
    tfc = get_test_model_trained("CNV", quant, act)
    export_onnx_path = build_dir+f"/cnv_w{quant}_a{quant}.onnx"
    export_qonnx(tfc, torch.randn(1, 3, 32, 32), build_dir+f"/cnv_w{quant}_a{act}.onnx"); # semicolon added to suppress log
    qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)

    model = ModelWrapper(build_dir+f"/cnv_w{quant}_a{act}.onnx")
    model = model.transform(ConvertQONNXtoFINN())

    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())

    model.save(build_dir+f"/cnv_w{quant}_a{act}_tidy.onnx")

    return build_dir+f"/cnv_w{quant}_a{act}_tidy.onnx"

model_path = get_onnx_model(1,1)
hw_name = "cnv_test"

first_steps = [
    'step_qonnx_to_finn',
    'step_tidy_up',
    'step_streamline',
    'step_convert_to_hw',
    'step_create_dataflow_partition',
    'step_specialize_layers',
    'step_target_fps_parallelization',
    'step_apply_folding_config',
    'step_minimize_bit_width',
    'step_generate_estimate_reports'
]

target_fps = 1
folding_json = None

generate_hardware(build_dir, model_path, hw_name, first_steps, folding_json, target_fps)

# vamos lidar apenas com MVAU
# cuidar com as regras de definição de PEs e SIMD

# agora, pega o JSON que ta dentro de build_dir/cnv_test/ e analisa se todos os PEs e SIMD estão ok
# fique analisando o que está dentro de build.log pra ver se deu erro
# se deu erro, pegue o último step que deu certo e copie todo o log para uma pasta. um erro pode ser um timeout definido pelo usuário
#   adicione um PE e um SIMD na primeira camada e tente novamente
#   se deu erro, adicione na proxima camada
# quando der certo, retire PEs e SIMDs das camadas passadas até dar erro
#   quando der erro, voltar para a configuração passada
# se deu certo, temos o menor hardware sintetizável

# ver o quanto esse hardware ocupa do total e tentar aumentar o hardware na mesma proporção que falta
# se der erro, diminuir a proporção. buscar sempre extrapolar pra dar erro
# quando der certo, analisar a distância entre o target e o real
#   se o erro for aceitável, finaliza
#   se não for, tenta novamente por n iterações de hardwares corretos
