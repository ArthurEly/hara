import numpy as np

def finn_partition_fifo(total_depth, user_ram_style="block",
                        max_qsrl_depth=256, max_vivado_depth=32768):
    """Replica exatamente o get_fifo_split_configs do FINN (set_fifo_depths.py).

    Regras:
    - depth <= max_qsrl_depth (256): FIFO única RTL, sem split.
    - depth >  max_qsrl_depth: divide em potências de 2 (descendo), onde
      pedaços > 256 usam impl_style=vivado e pedaços <= 256 usam impl_style=rtl.
    """

    def floor_pow2(x):
        if (x & (x - 1) == 0) and x != 0:
            return x
        return 1 << ((x - 1).bit_length() - 1)

    def decompose_pow2(x):
        if x <= max_qsrl_depth:
            return [x]
        r = floor_pow2(x)
        if x == r:
            return [x]
        return [r, *decompose_pow2(x - r)]

    # Caso trivial: FIFO pequena → única, RTL
    if total_depth <= max_qsrl_depth:
        return [{"depth": total_depth, "ram_style": "distributed", "impl_style": "rtl"}]

    # 1ª passagem: respeitar max_vivado_depth (32k)
    pass1 = []
    remainder = total_depth
    while remainder:
        if remainder > max_vivado_depth:
            pass1.append(max_vivado_depth)
            remainder -= max_vivado_depth
        else:
            pass1.append(remainder)
            remainder = 0

    # 2ª passagem: decompor em potências de 2
    pass2 = [d for chunk in pass1 for d in decompose_pow2(chunk)]

    # Atribuir impl_style a cada fatia
    slices = []
    for d in pass2:
        if d <= max_qsrl_depth:
            slices.append({"depth": d, "ram_style": "distributed", "impl_style": "rtl"})
        else:
            slices.append({"depth": d, "ram_style": user_ram_style, "impl_style": "vivado"})
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