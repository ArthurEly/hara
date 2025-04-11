#!/usr/bin/env python
# coding: utf-8

# In[1]:


from cnns_classes import t1_quantizedCNN, t2_quantizedCNN
import torch
from utils import utils


# In[2]:


topologies = [
    {
        'id':2, 
        'tp_class':t2_quantizedCNN,
        'quant': [4]
    }
]

target_fps_list = [501]
device = torch.device('cpu')


# In[3]:


from finn.util.basic import make_build_dir
from finn.util.visualization import showInNetron
import os
    
build_dir = os.environ["FINN_BUILD_DIR"]


# In[4]:


from qonnx.core.datatype import DataType
import torch
import onnx
from finn.util.test import get_test_model_trained
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.transformation.streamline import Streamline
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.general import RemoveUnusedTensors
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hls
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_data_layouts import InferDataLayouts
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
import shutil


# In[5]:


def make_onnx(cnv,quant,topology):
    cnv.load_state_dict(torch.load(f"./pytorch_models/sat6-cnn-t{topology}w{quant}.pt"))
    onnx_output_filename = f"./hardware_onnxs/sat6-cnn-t{topology}w{quant}.onnx"
    
    export_onnx_path = build_dir + f"/end2end_cnv_t{topology}w{quant}_export.onnx"
    #tidy up
    export_qonnx(cnv, torch.randn(1, 4, 32, 32), export_onnx_path)    
    qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())

    #preprocessing
    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType["UINT8"])

    # postprocessing: insert Top-1 node at the end
    model = model.transform(InsertTopK(k=1))
    # tidy-up again
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())

    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    # absorb final add-mul nodes into TopK
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())

    # choose the memory mode for the MVTU units, decoupled or const
    model = model.transform(to_hls.InferBinaryMatrixVectorActivation())
    model = model.transform(to_hls.InferQuantizedMatrixVectorActivation())
    # TopK to LabelSelect
    model = model.transform(to_hls.InferLabelSelectLayer())
    # input quantization (if any) to standalone thresholding
    model = model.transform(to_hls.InferThresholdingLayer())
    model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(to_hls.InferStreamingMaxPool())
    # get rid of Reshape(-1, 1) operation between hlslib nodes
    model = model.transform(RemoveCNVtoFCFlatten())
    # get rid of Tranpose -> Tranpose identity seq
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    # infer tensor data layouts
    model = model.transform(InferDataLayouts())
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(build_dir + f"/end2end_cnv_t{topology}w{quant}_dataflow_parent.onnx")
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    # save the dataflow partition with a different name for easier access
    dataflow_model = ModelWrapper(dataflow_model_filename)
    dataflow_model.save(onnx_output_filename)
    return onnx_output_filename


# In[6]:


build_dir_name = "builds_pynq"


# In[7]:


def generate_hardware(topology,quant,target_fps):

    model_file = f"./hardware_onnxs/sat6-cnn-t{topology}w{quant}.onnx"
    
    hw_name = utils.get_hardware_config_name(quant=quant,topology=topology,target_fps=target_fps)
    estimates_output_dir = f"./{build_dir_name}/{hw_name}_u"
    
    #Delete previous run results if exist
    if os.path.exists(estimates_output_dir):
        shutil.rmtree(estimates_output_dir)
        print("Previous run results deleted!")
    
    cfg_estimates = build.DataflowBuildConfig(
        output_dir          = estimates_output_dir,
        mvau_wwidth_max     = 80, #tinha usado 80
        target_fps          = target_fps, #tinha usado 100
        synth_clk_period_ns = 10,
        rtlsim_batch_size   = 1000,
        verify_input_npy    = "input.npy",
        stitched_ip_gen_dcp = True,
        #enable_hw_debug = True,
        verify_expected_output_npy = "expected_output.npy",
        board = "Pynq-Z1",
        shell_flow_type = build_cfg.ShellFlowType.VIVADO_ZYNQ,
        specialize_layers_config_file = "impl.json",
        # verify_save_rtlsim_waveforms = True,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER
        ],
        steps=[
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
        ],
        verify_steps=[
            build_cfg.VerificationStepType.STITCHED_IP_RTLSIM
        ]
    )    
    
    build.build_dataflow_cfg(model_file, cfg_estimates)


# In[8]:


def build_hardware(topology,target_fps,topology_class,quant):
    cnv = topology_class(bit_quantization=quant)
    onnx_filename = make_onnx(cnv,quant,topology=topology)
    generate_hardware(target_fps=target_fps,quant=quant,topology=topology)
    


# In[9]:


import os
import shutil
finn_build_dir = os.environ["FINN_BUILD_DIR"] + '/'

def move_intermediate_outputs_dir(dest_folder_name):
    source_folder = finn_build_dir
    destination_folder = dest_folder_name
    
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
        
    # fetch all files
    for file_name in os.listdir(source_folder):
        # construct full file path
        source = source_folder + file_name
        destination = destination_folder + file_name
        # move only files
        if "pyverilator_ipstitched" in file_name:
            shutil.rmtree(source)
        else:
            try:
                shutil.move(source, destination)
                print("Arquivo movido com sucesso!")
            except FileNotFoundError:
                print("Erro: Arquivo de origem não encontrado.")
            except PermissionError:
                print("Erro: Permissão negada para mover o arquivo.")
            except Exception as e:
                print(f"Erro inesperado: {e}")

    print('All files moved successfully')


# In[10]:


def get_zynq_proj(src,dst):      
    for folder in os.listdir(src):
        if "vivado_zynq_proj" in folder:
            try:
                shutil.copytree(src + folder, dst)
                print("Arquivo movido com sucesso!")
            except FileNotFoundError:
                print("Erro: Arquivo de origem não encontrado.")
            except PermissionError:
                print("Erro: Permissão negada para mover o arquivo.")
            except Exception as e:
                print(f"Erro inesperado: {e}")
            break

    print('ZYNQ project successfully copied')


# In[11]:


get_ipython().run_cell_magic('time', '', 'for tp in topologies:\n    for quant in tp[\'quant\']:\n        for target_fps in target_fps_list:\n            build_hardware(topology=tp[\'id\'],target_fps=target_fps,topology_class=tp[\'tp_class\'],quant=quant)\n            hw_name = utils.get_hardware_config_name(quant=quant,topology=tp[\'id\'],target_fps=target_fps)\n            get_zynq_proj(src=finn_build_dir,dst=f"./{build_dir_name}/{hw_name}_u/zynq_proj/")\n            move_intermediate_outputs_dir(f"./{build_dir_name}/{hw_name}_sources_u/")\n')


# In[ ]:





# In[ ]:




