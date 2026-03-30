"""Simple quantization utilities."""
import torch
from torch import nn


def quantize_weights_int8(weight):
    """Quantize weights to INT8."""
    scale = weight.abs().max() / 127.0
    quantized = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def dequantize_weights_int8(quantized, scale):
    """Dequantize INT8 weights back to float."""
    return quantized.float() * scale


class QuantizedLinear(nn.Module):
    """Simple INT8 quantized linear layer."""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # store quantized weights
        self.register_buffer('weight_quantized', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    @classmethod
    def from_linear(cls, linear_layer):
        """Create quantized layer from existing linear layer."""
        has_bias = linear_layer.bias is not None
        quantized = cls(linear_layer.in_features, linear_layer.out_features, bias=has_bias)
        
        # quantize weights
        q_weight, scale = quantize_weights_int8(linear_layer.weight.data)
        quantized.weight_quantized = q_weight
        quantized.weight_scale = scale
        
        if has_bias:
            quantized.bias.data = linear_layer.bias.data.clone()
        
        return quantized
    
    def forward(self, x):
        # dequantize for computation (simple approach)
        weight = dequantize_weights_int8(self.weight_quantized, self.weight_scale)
        return nn.functional.linear(x, weight, self.bias)


def quantize_model(model):
    """
    Quantize all Linear layers in a model to INT8.
    Returns a new model with quantized weights.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            quantized_layer = QuantizedLinear.from_linear(module)
            setattr(model, name, quantized_layer)
        else:
            quantize_model(module)
    return model


def dequantize_model(model):
    """
    Convert quantized layers back to regular Linear layers.
    """
    for name, module in model.named_children():
        if isinstance(module, QuantizedLinear):
            linear = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
            linear.weight.data = dequantize_weights_int8(module.weight_quantized, module.weight_scale)
            if module.bias is not None:
                linear.bias.data = module.bias.data.clone()
            setattr(model, name, linear)
        else:
            dequantize_model(module)
    return model


def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024
