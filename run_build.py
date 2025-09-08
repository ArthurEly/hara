# run_build.py
import argparse
import json
from utils.hw_utils import utils

# Argumentos que ainda são relevantes
parser = argparse.ArgumentParser()
parser.add_argument("--build_dir", type=str, required=True)
parser.add_argument("--hw_name", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True) 
parser.add_argument("--steps", type=str, required=True)

# Argumentos que se tornaram opcionais ou de controle
parser.add_argument("--folding_file", type=str, default=None)
parser.add_argument("--target_fps", type=str, default="None")

# Argumentos que são passados mas não são mais usados pela função de build
# Eles podem ser mantidos por enquanto para fins de log, ou removidos.
parser.add_argument("--topology", type=str)
parser.add_argument("--quant", type=int)
parser.add_argument("--run", type=int)

args = parser.parse_args()

# Converte os argumentos para os tipos corretos
target_fps_val = None if args.target_fps == "None" else int(args.target_fps)
folding_file_val = None if args.folding_file in ["", "None"] else args.folding_file
steps = json.loads(args.steps)

print("Começando a build...")
utils.build_hardware(
    model_path=args.model_path,
    build_dir=args.build_dir,
    hw_name=args.hw_name,
    steps=steps,
    folding_file=folding_file_val,
    target_fps=target_fps_val
)