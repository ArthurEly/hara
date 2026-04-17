import os
import glob
import shutil
import warnings

# Use FINN and QONNX imports
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp

from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import (
    MakeMaxPoolNHWC,
    MoveScalarLinearPastInvariants,
)

import sys
# Path to "hara"
hara_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(hara_dir)

from run_fps_map_job import _run_estimate_build

INPUT_ONNX_DIR = os.path.join(hara_dir, "models", "SAT6")
BASE_BUILD_DIR = os.path.join(hara_dir, "configs", "sat6_sec")

onnx_files = glob.glob(os.path.join(INPUT_ONNX_DIR, "*.onnx"))

for fp in onnx_files:
    if "finn_ready" in fp or "estimate" in fp:
        continue
        
    model_name = os.path.basename(fp).replace(".onnx", "")
    print(f"\n================ Processing: {model_name} ================")
    
    # 1. Pipeline Streamlining do T2
    model_build_dir = os.path.join(BASE_BUILD_DIR, f"{model_name}_files")
    os.makedirs(model_build_dir, exist_ok=True)
    temp_export_path = os.path.join(model_build_dir, "temp.onnx")
    shutil.copy(fp, temp_export_path)
    
    qonnx_cleanup(temp_export_path, out_file=temp_export_path)
    model = ModelWrapper(temp_export_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    
    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
    model = model.transform(InsertTopK(k=1))
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
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())

    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(Streamline())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    
    # weights são de 2 bits no SAT6_T2_2W2A PRUNED (q_bits=2)
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(to_hw.InferStreamingMaxPool())

    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(InferDataLayouts())
    
    parent_model_path = os.path.join(model_build_dir, "parent.onnx")
    final_dataflow_model_path = os.path.join(INPUT_ONNX_DIR, f"{model_name}_finn_ready.onnx")
    
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(parent_model_path)
    
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename_in_build_dir = sdp_node.get_nodeattr("model")
    
    shutil.copy(dataflow_model_filename_in_build_dir, final_dataflow_model_path)
    print(f"-> Generated FINN Dataflow Model: {final_dataflow_model_path}")
    
    # 2. RUN FINN BUILD run0
    print("-> Running initial estimate build (run0_get_initial_fold) ...")
    est_dir_1 = _run_estimate_build(
        os.path.join(BASE_BUILD_DIR, "builds"), 
        final_dataflow_model_path, 
        f"run0_{model_name}", 
        "SAT6_T2", 
        2, 
        "xc7z020clg400-1", 
        target_fps=1
    )
    if est_dir_1 and os.path.exists(os.path.join(est_dir_1, "intermediate_models", "step_generate_estimate_reports.onnx")):
        report_onnx = os.path.join(est_dir_1, "intermediate_models", "step_generate_estimate_reports.onnx")
        dest_onnx = os.path.join(INPUT_ONNX_DIR, f"{model_name}_estimate.onnx")
        shutil.copy(report_onnx, dest_onnx)
        print(f"-> FINN Report saved to {dest_onnx}")
    else:
        print(f"-> ERROR: Failed to generate estimate reports for {model_name}.")

print("\nALL MODELS PROCESSED SUCCESFULLY!")
