# run_build.py
import argparse
import json
from cnns_classes import t2_quantizedCNN, t1_quantizedCNN
from utils.hw_utils import utils

parser = argparse.ArgumentParser()
parser.add_argument("--build_dir", type=str)
parser.add_argument("--topology", type=int, required=True)
parser.add_argument("--target_fps", type=str, required=True)
parser.add_argument("--quant", type=int, required=True)
parser.add_argument("--steps", type=str, required=True)
parser.add_argument("--folding_file", type=str)
parser.add_argument("--run", type=int, required=True)
parser.add_argument("--hw_name", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True) 

args = parser.parse_args()

if args.target_fps == "None":
    args.target_fps = None
else:
    args.target_fps = int(args.target_fps)

if args.folding_file == "":
    args.folding_file = None

steps = json.loads(args.steps)

topology_class = {1: t1_quantizedCNN, 2: t2_quantizedCNN}[args.topology]
print("Começando a build...")
utils.build_hardware(
    model_path=args.model_path,
    build_dir=args.build_dir,
    topology=args.topology,
    target_fps=args.target_fps,
    topology_class=topology_class,
    quant=args.quant,
    steps=steps,
    folding_file=args.folding_file,
    run=args.run,
    hw_name=args.hw_name
)