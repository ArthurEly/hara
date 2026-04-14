import numpy as np
import math

def finn_partition_fifo(total_depth, user_ram_style="block"):
    """Simula o particionamento do FINN em potências de 2, imitando perfeitamente o SplitLargeFIFOs."""
    slices = []
    remaining_depth = total_depth
    
    # O FINN sempre particiona descendo pelas maiores potências de 2!
    while remaining_depth > 0:
        power = int(math.log2(remaining_depth))
        chunk_depth = int(math.pow(2, power))
        
        if chunk_depth >= 512:
            slices.append({"depth": chunk_depth, "ram_style": user_ram_style, "impl_style": "vivado"})
        else:
            slices.append({"depth": chunk_depth, "ram_style": "distributed", "impl_style": "rtl"})
            
        remaining_depth -= chunk_depth
        
    return slices

def prepare_fifo_features(depth, in_width, ram_style, impl_style, bits, simd):
    """Prepara o dicionário de features exatamente como o treino espera."""
    bit_capacity = depth * in_width
    return {
        "is_ram_style_auto": 0,
        "is_ram_style_block": 1 if ram_style == "block" else 0,
        "is_ram_style_distributed": 1 if ram_style == "distributed" else 0,
        "is_impl_style_rtl": 1 if impl_style == "rtl" else 0,
        "is_impl_style_vivado": 1 if impl_style == "vivado" else 0,
        "log_inWidth": np.log1p(in_width),
        "log_depth": np.log1p(depth),
        "log_bit_capacity": np.log1p(bit_capacity),
        "log_srl_complexity": np.log1p(bit_capacity / 32.0),
        "dataType_bits": bits,
        "simd": simd
    }