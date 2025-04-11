import pickle
import pandas as pd
import os
import shutil
import json 
from datetime import datetime
import csv
import onnx
import onnx
from onnx import helper
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.fold_constants import FoldConstants
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.core.datatype import DataType
import torch
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

class utils():
    def __init__(self):
        super(utils, self).__init__()
        
    def save_object(filename, object):
        with open(filename, 'wb') as outp:
            pickle.dump(object, outp, pickle.HIGHEST_PROTOCOL)
    
    def read_object(filename):
        with open(filename, 'rb') as inp:
            return pickle.load(inp)

    def get_model_output_filename (topology,quant):
        return f"./pytorch_models/sat6-cnn-t{topology}w{quant}.pt"

    def get_hardware_config_name(topology, quant, target_fps, extra=''):
        fps_part = f"_{target_fps}fps" if target_fps is not None else ""
        return f"t{topology}w{quant}{fps_part}{extra}"


    def save_csv_table(results,csv_pathname):
        df = pd.DataFrame(results)
        print(df.to_string(header=None, index=False))
        df.to_csv(csv_pathname) 
        print(f"succesfully saved at {csv_pathname}")

    def move_intermediate_outputs_dir(src, dst):
        source_folder = src
        destination_folder = dst
        
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

    def read_build_log(build_dir):
        log_path = os.path.join(build_dir, "build_dataflow.log")
        if not os.path.exists(log_path):
            return ""
        with open(log_path, "r") as f:
            return f.read()

    def detect_synthesis_error(log_text, timeout_keywords=None):
        if timeout_keywords is None:
            timeout_keywords = ["ERROR", "failed", "timeout", "synthesis failed", "crash"]

        for line in log_text.splitlines():
            if any(keyword.lower() in line.lower() for keyword in timeout_keywords):
                return True
        return False

    def read_folding_config(build_dir):
        fold_path = os.path.join(build_dir, "auto_folding_config.json")
        if not os.path.exists(fold_path):
            return {}
        with open(fold_path, "r") as f:
            return json.load(f)

    def save_crash_report(build_dir, destination_base="./crash_reports"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dest_dir = os.path.join(destination_base, f"crash_{timestamp}")
        os.makedirs(dest_dir, exist_ok=True)

        files_to_copy = ["build_dataflow.log", "auto_folding_config.json"]
        for fname in files_to_copy:
            src_path = os.path.join(build_dir, fname)
            if os.path.exists(src_path):
                shutil.copy(src_path, os.path.join(dest_dir, fname))

        print(f"[!] Crash report saved at: {dest_dir}")
        return dest_dir
    
    def append_run_summary(file_path, hw_name, status, folding_config):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = {
            "date": now,
            "hw_name": hw_name,
            "status": status,
            "folding_summary": json.dumps(folding_config)
        }

        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=summary.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)

    def modify_folding(folding, onnx_path):
        def get_layer_features(onnx_path):
            model = onnx.load(onnx_path)
            features = {}

            for node in model.graph.node:
                layer_name = node.name
                op_type = node.op_type

                attr_dict = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}

                if op_type in ["MVAU_hls", "MVAU_rtl"]:  # MVAU
                    mw = attr_dict.get("MW")
                    mh = attr_dict.get("MH")
                    if mw is not None and mh is not None:
                        features[layer_name] = {"MW": int(mw), "MH": int(mh)}

                elif op_type in ["FMPadding_rtl", "FMPadding_hls"]:
                    num_channels = attr_dict.get("NumChannels")
                    if num_channels is not None:
                        features[layer_name] = {"NumChannels": int(num_channels)}

                elif op_type in ["ConvolutionInputGenerator_rtl", "ConvolutionInputGenerator_hls"]:
                    ifm_channels = attr_dict.get("IFMChannels")
                    if ifm_channels is not None:
                        features[layer_name] = {"IFMChannels": int(ifm_channels)}

                elif op_type in ["LabelSelect_rtl","LabelSelect_hls"]:
                    continue  # ignora esse

            return features

        feature_dims = get_layer_features(onnx_path)
        new_folding = {}

        for layer, cfg in folding.items():
            if layer == "Defaults":
                new_cfg = cfg.copy()
                new_folding[layer] = new_cfg
                continue

            new_cfg = cfg.copy()
            layer_features = feature_dims.get(layer, {})

            # MVAU: ajustar SIMD com MW
            if "SIMD" in new_cfg and any(k in layer_features for k in ["MW", "IFMChannels", "NumChannels"]):
                mw = layer_features.get("MW") or layer_features.get("IFMChannels") or layer_features.get("NumChannels")
                new_simd = new_cfg["SIMD"] * 2
                while new_simd > 1 and mw % new_simd != 0:
                    new_simd -= 1
                if mw % new_simd == 0:
                    new_cfg["SIMD"] = new_simd

            # MVAU: ajustar PE com MH
            if "PE" in new_cfg and "MH" in layer_features:
                mh = layer_features["MH"]
                new_pe = new_cfg["PE"] * 2
                while new_pe > 1 and mh % new_pe != 0:
                    new_pe -= 1
                if mh % new_pe == 0:
                    new_cfg["PE"] = new_pe

            new_folding[layer] = new_cfg

        return new_folding

    def build_hardware(build_dir,topology, target_fps, topology_class, quant, steps, folding_file, run, hw_name):
        def make_onnx(build_dir, cnv,quant,topology):
            cnv.load_state_dict(torch.load(f"../notebooks/sat6_cnn/pytorch_models/sat6-cnn-t{topology}w{quant}.pt"))
            onnx_output_filename = f"../notebooks/sat6_cnn/hardware_onnxs/sat6-cnn-t{topology}w{quant}.onnx"
            
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

        def generate_hardware(build_dir, topology, quant, target_fps, steps, folding_file, run, hw_name):
            model_file = f"../notebooks/sat6_cnn/hardware_onnxs/sat6-cnn-t{topology}w{quant}.onnx"
            estimates_output_dir = f"{build_dir}/{hw_name}"

            # Delete previous run results if they exist
            if os.path.exists(estimates_output_dir):
                shutil.rmtree(estimates_output_dir)
                print("Previous run results deleted!")

            cfg_estimates = build.DataflowBuildConfig(
                output_dir=estimates_output_dir,
                folding_config_file=folding_file,
                target_fps=target_fps,
                synth_clk_period_ns=10,
                board="Pynq-Z1",
                shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
                generate_outputs=[
                    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
                    build_cfg.DataflowOutputType.STITCHED_IP,
                    build_cfg.DataflowOutputType.BITFILE,
                    build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
                    build_cfg.DataflowOutputType.PYNQ_DRIVER
                ],
                steps=steps,
            )

            build.build_dataflow_cfg(model_file, cfg_estimates)

        cnv = topology_class(bit_quantization=quant)
        make_onnx(build_dir, cnv, quant, topology=topology)
        generate_hardware(
            build_dir=build_dir,
            target_fps=target_fps,
            quant=quant,
            topology=topology,
            steps=steps,
            folding_file=folding_file,
            run=run,
            hw_name=hw_name
    )


