import json, os, sys
sys.path.append(".")
from multi_module_learner import MultiModuleLearner
run_dir = "/home/arthurely/Desktop/finn_chi2p/hara/exhaustive_hw_builds/MNIST_1W1A_2026-04-10_08-53-31/run6_optimized"
onnx_file = os.path.join(run_dir, "intermediate_models", "step_generate_estimate_reports.onnx")
hw_config = os.path.join(run_dir, "final_hw_config.json")
m_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieval", "results", "trained_models")
learner = MultiModuleLearner(m_dir)
with open(hw_config, "r") as f: cfg = json.load(f)
p = learner.predict(onnx_file, [cfg])

for d, v in p[0]["_details"].items():
    if "FIFO" in d or "ConvolutionInputGenerator" in d:
        print(f"[{d}] BRAM={v.get('BRAM (36k eq.)', 0.0)} LUT={v.get('Total LUT', 0.0)}")
