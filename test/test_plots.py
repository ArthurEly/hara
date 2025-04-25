import os
import pandas as pd
import matplotlib.pyplot as plt
from hw_utils import utils
csv_path = "/home/arthurely/Desktop/finn/hara/builds/run_2025-04-18_20-07-05/run_summary.csv"

# Teste da função de plotagem
utils.plot_area_usage_from_csv(csv_path, output_dir="./")
