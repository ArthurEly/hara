import os
import glob
import torch
import warnings
from brevitas.export import export_qonnx

import sys
sys.path.append("/home/arthurely/Desktop/finn_chi2p/deps/brevitas/src/brevitas_examples/bnn_pynq")

from models.cnns_classes import t2_quantizedCNN

# Find all pruned checkpoints
EXP_DIR = "/home/arthurely/Desktop/finn_chi2p/deps/brevitas/src/brevitas_examples/bnn_pynq/experiments/sat6_t2_2w2a_2W2A_20260415_154946"
OUTPUT_DIR = "/home/arthurely/Desktop/finn_chi2p/hara/models/SAT6"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Also grab the original best.tar
checkpoint_tars = glob.glob(os.path.join(EXP_DIR, "checkpoints", "best.tar"))
pruned_pths = glob.glob(os.path.join(EXP_DIR, "final_optimized_drop*_model.pth"))

all_models = checkpoint_tars + pruned_pths

for path in all_models:
    print(f"\nProcessing: {path}")
    
    # Load state dict
    if path.endswith(".tar"):
        package = torch.load(path, map_location="cpu")
        state_dict = package.get("state_dict", package)
        name = "sat6_t2_baseline"
    else:
        state_dict = torch.load(path, map_location="cpu")
        name = os.path.basename(path).replace(".pth", "")
        # The accuracy drops can be parsed from CSV, but we just generate ONNX for all of them
        
    # Infer architecture from state_dict
    c1 = state_dict['conv1.weight'].shape[0]
    c2 = state_dict['conv2.weight'].shape[0]
    c3 = state_dict['conv3.weight'].shape[0]
    c4 = state_dict['conv4.weight'].shape[0]
    fc_in = state_dict['fc1.weight'].shape[1]
    
    print(f"  Inferred sizes: channels=[{c1}, {c2}, {c3}, {c4}], fc_in={fc_in}")
    
    # Instantiate
    model = t2_quantizedCNN(bit_quantization=2, channels=[c1, c2, c3, c4], fc_in=fc_in)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # The input shape to SAT6 is (1, 4, 32, 32) (padded from 28x28)
    input_shape = (1, 4, 32, 32)
    fake_input = torch.randn(input_shape)

    out_path = os.path.join(OUTPUT_DIR, f"{name}.onnx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_qonnx(model, export_path=out_path, input_t=fake_input)
    print(f"  Saved QONNX to: {out_path}")

print("\nDone exporting all models.")
