# cnv.py
import os
import shutil
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

build_dir = os.environ["FINN_BUILD_DIR"]

def generate_hardware(build_dir, model, hw_name, steps_list, folding_config_path, target_fps):
    build_dir_name = build_dir
    estimates_output_dir = f"{build_dir_name}/{hw_name}"

    if os.path.exists(estimates_output_dir):
        shutil.rmtree(estimates_output_dir)
        print("Previous run results deleted!")

    cfg_estimates = build.DataflowBuildConfig(
        output_dir=estimates_output_dir,
        target_fps=target_fps,
        folding_config_file= folding_config_path,
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
        steps=steps_list
    )

    build.build_dataflow_cfg(model, cfg_estimates)
