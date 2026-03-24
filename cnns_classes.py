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

from brevitas_examples.bnn_pynq.models.mobilenet import mobilenet
import configparser


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

class CommonActQuant(CommonQuant, ActQuantSolver):
    min_val = -1.0
    max_val = 1.0

class t1_quantizedCNN(nn.Module):

    def __init__(self, weight_bit_width, act_bit_width, arch_config=None):
        super(t1_quantizedCNN, self).__init__()

        if arch_config is None: arch_config = {}

        conv1_out = arch_config.get('conv1_out', 20)
        conv2_out = arch_config.get('conv2_out', 8)

        self.conv1 = qnn.QuantConv2d(
            4, conv1_out, kernel_size=3, stride=2, padding=1, bias=False,
            weight_bit_width=weight_bit_width, weight_quant=CommonWeightQuant)

        self.relu1 = qnn.QuantReLU(
            act_quant=CommonUintActQuant, bit_width=act_bit_width, return_quant_tensor=True)

        self.conv2 = qnn.QuantConv2d(
            conv1_out, conv2_out, kernel_size=1, stride=1, bias=False,
            weight_bit_width=weight_bit_width, weight_quant=CommonWeightQuant)


        self.relu2 = qnn.QuantReLU(
            act_quant=CommonUintActQuant, bit_width=act_bit_width, return_quant_tensor=True)

        self.fc1 = qnn.QuantLinear(
            conv2_out * 16 * 16, 6, bias=False,
            weight_bit_width=weight_bit_width, weight_quant=CommonWeightQuant)
        
        for m in self.modules():
            if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                nn.init.uniform_(m.weight.data, -1, 1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)  

    def forward(self, x):
        x = self.conv1(x); x = self.relu1(x)
        x = self.conv2(x); x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class t2_quantizedCNN(nn.Module):
    def __init__(self, weight_bit_width, act_bit_width, arch_config=None):
        super(t2_quantizedCNN, self).__init__()

        if arch_config is None: arch_config = {}
        
        conv1_out = arch_config.get('conv1_out', 16)
        conv2_out = arch_config.get('conv2_out', 8)
        conv3_out = arch_config.get('conv3_out', 16)
        conv4_out = arch_config.get('conv4_out', 8)
        
        act_quant_class = CommonUintActQuant if act_bit_width > 1 else CommonActQuant
        act_layer = qnn.QuantReLU if act_bit_width > 1 else qnn.QuantIdentity

        self.conv1 = qnn.QuantConv2d(4, conv1_out, 3, 2, 1, bias=False, weight_bit_width=weight_bit_width, weight_quant=CommonWeightQuant)
        self.relu1 = act_layer(act_quant=act_quant_class, bit_width=act_bit_width, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(conv1_out, conv2_out, 3, 2, 1, bias=False, weight_bit_width=weight_bit_width, weight_quant=CommonWeightQuant)
        self.relu2 = act_layer(act_quant=act_quant_class, bit_width=act_bit_width, return_quant_tensor=True)
        self.conv3 = qnn.QuantConv2d(conv2_out, conv3_out, 3, 2, 1, bias=False, weight_bit_width=weight_bit_width, weight_quant=CommonWeightQuant)
        self.relu3 = act_layer(act_quant=act_quant_class, bit_width=act_bit_width, return_quant_tensor=True)
        self.conv4 = qnn.QuantConv2d(conv3_out, conv4_out, 3, 2, 1, bias=False, weight_bit_width=weight_bit_width, weight_quant=CommonWeightQuant)
        self.relu4 = act_layer(act_quant=act_quant_class, bit_width=act_bit_width, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(conv4_out * 2 * 2, 6, bias=False, weight_bit_width=weight_bit_width, weight_quant=CommonWeightQuant)
        
        for m in self.modules():
            if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                nn.init.uniform_(m.weight.data, -1, 1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)                    

    def forward(self, x):
        x = self.conv1(x); x = self.relu1(x); x = self.conv2(x); x = self.relu2(x)
        x = self.conv3(x); x = self.relu3(x); x = self.conv4(x); x = self.relu4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x 

class MobileNetWrapper(nn.Module):
    def __init__(self, weight_bit_width, act_bit_width, arch_config=None):
        super(MobileNetWrapper, self).__init__()
        
        # Use a ConfigParser to mimic the behavior of brevitas_examples config reading
        cfg = configparser.ConfigParser()
        cfg.add_section('QUANT')
        cfg.set('QUANT', 'WEIGHT_BIT_WIDTH', str(weight_bit_width))
        cfg.set('QUANT', 'ACT_BIT_WIDTH', str(act_bit_width))
        cfg.set('QUANT', 'IN_BIT_WIDTH', str(8)) # standard
        
        cfg.add_section('MODEL')
        # SAT6 typically has 6 classes, adjusting for the default dataset used in HARA
        cfg.set('MODEL', 'NUM_CLASSES', str(6))
        # Assuming SAT-6 images have 4 channels (RGB + NIR) as per other models
        cfg.set('MODEL', 'IN_CHANNELS', str(4))
        
        self.model = mobilenet(cfg)
        
    def forward(self, x):
        return self.model(x)