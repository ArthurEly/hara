# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dependencies import value

from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.quant.solver import ActQuantSolver
from brevitas.quant.solver import WeightQuantSolver
from brevitas.quant import Uint8ActPerTensorFloat
import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8ActPerTensorFloat

class CommonQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    @value
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT


class CommonWeightQuant(CommonQuant, WeightQuantSolver):
    scaling_const = 1.0


class CommonUintActQuant(Uint8ActPerTensorFloat):
    """
    Common unsigned act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    restrict_scaling_type = RestrictValueType.LOG_FP

class t1_quantizedCNN(nn.Module):
    def __init__(self, bit_quantization):
        super(t1_quantizedCNN, self).__init__()
        self.conv1 = qnn.QuantConv2d(
            4, 
            20, 
            kernel_size=3, stride=2, padding=1, 
            bias = False,
            weight_bit_width=bit_quantization, 
            weight_quant=CommonWeightQuant, 
        )
        self.relu1 = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=bit_quantization,
            return_quant_tensor=True
        )
        
        self.conv2 = qnn.QuantConv2d(
            20, 
            8, 
            kernel_size=1, stride=1,
            bias = False,
            weight_bit_width=bit_quantization, 
            weight_quant=CommonWeightQuant, 
        )
        self.relu2 = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=bit_quantization,
            return_quant_tensor=True
        )
        
        self.fc1 = qnn.QuantLinear(
            8*16*16, 
            6, 
            bias = False,
            weight_bit_width=bit_quantization, 
            weight_quant=CommonWeightQuant, 
        )

        if (bit_quantization == 2):
            for m in self.modules():
                if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                    nn.init.uniform_(m.weight.data, -1, 1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)      
        return x

class t2_quantizedCNN(nn.Module):
    def __init__(self,bit_quantization):
        super(t2_quantizedCNN, self).__init__()
        self.conv1 = qnn.QuantConv2d(
            4, 
            16, 
            kernel_size=3, stride=2, padding=1, 
            bias = False,
            weight_bit_width=bit_quantization, 
            weight_quant=CommonWeightQuant, 
        )
        self.relu1 = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=bit_quantization,
            return_quant_tensor=True
        )
        
        self.conv2 = qnn.QuantConv2d(
            16, 
            8, 
            kernel_size=3, stride=2, padding=1, 
            bias = False,
            weight_bit_width=bit_quantization, 
            weight_quant=CommonWeightQuant, 
        )
        self.relu2 = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=bit_quantization,
            return_quant_tensor=True
        )
        
        self.conv3 = qnn.QuantConv2d(
            8, 
            16, 
            kernel_size=3, stride=2, padding=1, 
            bias = False,
            weight_bit_width=bit_quantization, 
            weight_quant=CommonWeightQuant, 
        )
        self.relu3 = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=bit_quantization,
            return_quant_tensor=True
        )
        
        self.conv4 = qnn.QuantConv2d(
            16, 
            8, 
            kernel_size=3, stride=2, padding=1, 
            bias = False,
            weight_bit_width=bit_quantization, 
            weight_quant=CommonWeightQuant, 
        )
        self.relu4 = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=bit_quantization,
            return_quant_tensor=True
        )
        
        self.fc1 = qnn.QuantLinear(
            8*2*2, 
            6, 
            bias = False,
            weight_bit_width=bit_quantization, 
            weight_quant=CommonWeightQuant, 
        )

        if (bit_quantization == 2):
            for m in self.modules():
                if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                    nn.init.uniform_(m.weight.data, -1, 1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)      
        return x