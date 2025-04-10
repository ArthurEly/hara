#sys.path.append("/home/arthurely/Desktop/finn/notebooks/CIFAR10")
#from cnv import generate_hardware

import subprocess
from cifar import generate_hardware

# Exemplo de uso com parâmetros do DSE
model_path = "/home/arthurely/Desktop/finn/builds/cnv_w2_a2_tidy.onnx"
hw_name = "cnv_w2_a2_fps3000"
steps = [
    'step_qonnx_to_finn',
    'step_tidy_up',
    'step_streamline',
    'step_convert_to_hw',
    'step_create_dataflow_partition',
    'step_specialize_layers',
    'step_target_fps_parallelization',
    'step_apply_folding_config',
    'step_minimize_bit_width',
    'step_generate_estimate_reports',
    'step_hw_codegen',
    'step_hw_ipgen',
    'step_set_fifo_depths',
    'step_create_stitched_ip',
    'step_synthesize_bitfile',
    'step_make_pynq_driver',
    'step_deployment_package'
]
folding_json = "/home/arthurely/Desktop/finn/hara/folding.json"
target_fps = 3000

generate_hardware(model_path, hw_name, steps, folding_json, target_fps)