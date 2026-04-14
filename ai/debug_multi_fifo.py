import json, os, sys
from multi_module_learner import MultiModuleLearner

sys.path.append(".")

run_dir = "/home/arthurely/Desktop/finn_chi2p/hara/exhaustive_hw_builds/SAT6_T2W2_2026-04-09_19-59-04/run1_baseline_folded"
onnx_file = os.path.join(run_dir, "intermediate_models", "step_generate_estimate_reports.onnx")
hw_config = os.path.join(run_dir, "final_hw_config.json")
m_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieval", "results", "trained_models")

learner = MultiModuleLearner(m_dir)
with open(hw_config, "r") as f: cfg = json.load(f)

original_predict_layer = learner._predict_layer

def debug_predict_layer(self, module_key, node_attrs, folding_cfg, depth):
    res = original_predict_layer(module_key, node_attrs, folding_cfg, depth)
    
    if module_key == "SplitFIFO_area":
        name = node_attrs.get("name", "Unknown")
        print(f"\n--- [DEBUG FIFO: {name}] ---")
        
        user_depth = folding_cfg.get(name, {}).get("depth")
        
        r_style = node_attrs.get("ram_style", "auto")
        if isinstance(r_style, bytes): r_style = r_style.decode("utf-8")
        
        i_style = node_attrs.get("impl_style", "rtl")
        if isinstance(i_style, bytes): i_style = i_style.decode("utf-8")
        
        # Como o FINN mapeia os 'auto' internamente na nossa heurística:
        if "auto" in r_style.lower() or r_style == "":
            decision_style = "block" if depth > 512 else "distributed"
        else:
            decision_style = r_style.lower()

        specialist_name = "SplitFIFO_block" if decision_style == "block" else "SplitFIFO_distributed"
        
        print(f"  Depth Selecionada: {depth}")
        print(f"  Depth do JSON: {user_depth}")
        print(f"  ram_style: {r_style} -> Decisão: {decision_style}")
        print(f"  impl_style: {i_style}")
        print(f"  Especialista Alvo: {specialist_name}")
        print(f"  => LUT Predita: {res.get('Total LUT')}")
        print(f"  => BRAM Predita: {res.get('BRAM (36k eq.)')}")
        
    return res

print("[*] Injetando debug e rodando predição...")
# Sobrescrevendo o método da instância da classe:
import types
learner._predict_layer = types.MethodType(debug_predict_layer, learner)
learner.predict(onnx_file, [cfg])
