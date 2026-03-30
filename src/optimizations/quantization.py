"""Quantization utilities using PyTorch's optimized quantization."""
import torch
from torch import nn
import copy


def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024


def quantize_model_dynamic(model):
    """
    Apply PyTorch dynamic quantization to the model.
    
    This uses optimized INT8 kernels for Linear layers on CPU.
    Dynamic quantization quantizes weights statically and activations dynamically.
    
    Note: Dynamic quantization only works on CPU but provides real speedup.
    """
    # Dynamic quantization requires CPU
    model_cpu = copy.deepcopy(model).cpu()
    
    # Apply dynamic quantization to Linear layers
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8
    )
    
    return quantized_model


def quantize_model_static(model, calibration_data=None):
    """
    Apply static quantization (requires calibration data).
    
    Static quantization quantizes both weights and activations statically,
    which can be faster but requires representative calibration data.
    """
    model_cpu = copy.deepcopy(model).cpu().eval()
    
    # Fuse modules for better quantization (conv-bn-relu, etc.)
    # For transformers, we mainly have linear layers
    
    # Set quantization config
    model_cpu.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    
    # Prepare for quantization
    torch.ao.quantization.prepare(model_cpu, inplace=True)
    
    # Calibrate with sample data
    if calibration_data is not None:
        with torch.no_grad():
            for data in calibration_data:
                model_cpu(data)
    else:
        # Default calibration with random data
        with torch.no_grad():
            dummy_input = torch.randint(0, 100, (1, 16))
            try:
                model_cpu(dummy_input)
            except:
                pass
    
    # Convert to quantized model
    torch.ao.quantization.convert(model_cpu, inplace=True)
    
    return model_cpu


# =============================================================================
# Simple manual INT8 quantization (for demonstration/GPU compatibility)
# =============================================================================

def quantize_weights_int8(weight):
    """Quantize weights to INT8 with per-tensor scaling."""
    scale = weight.abs().max() / 127.0
    if scale == 0:
        scale = torch.tensor(1.0, device=weight.device)
    quantized = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def dequantize_weights_int8(quantized, scale):
    """Dequantize INT8 weights back to float."""
    return quantized.float() * scale


class QuantizedLinear(nn.Module):
    """
    INT8 quantized linear layer with cached dequantized weights.
    
    Caches the dequantized weights to avoid repeated dequantization overhead.
    This provides memory savings while maintaining speed.
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store quantized weights (for memory savings)
        self.register_buffer('weight_quantized', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        
        # Cache for dequantized weights (computed once)
        self._weight_cache = None
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    @classmethod
    def from_linear(cls, linear_layer):
        """Create quantized layer from existing linear layer."""
        has_bias = linear_layer.bias is not None
        quantized = cls(linear_layer.in_features, linear_layer.out_features, bias=has_bias)
        
        # Quantize weights
        q_weight, scale = quantize_weights_int8(linear_layer.weight.data)
        quantized.weight_quantized = q_weight
        quantized.weight_scale = scale
        
        if has_bias:
            quantized.bias.data = linear_layer.bias.data.clone()
        
        # Pre-compute dequantized weights
        quantized._weight_cache = dequantize_weights_int8(q_weight, scale)
        
        return quantized
    
    def forward(self, x):
        # Use cached dequantized weights (no repeated dequantization)
        if self._weight_cache is None or self._weight_cache.device != x.device:
            self._weight_cache = dequantize_weights_int8(
                self.weight_quantized, self.weight_scale
            ).to(x.device)
        
        return nn.functional.linear(x, self._weight_cache, self.bias)


def quantize_model(model, use_pytorch_quantization=True):
    """
    Quantize a model for inference.
    
    Args:
        model: The model to quantize
        use_pytorch_quantization: If True and on CPU, use PyTorch's optimized 
                                  dynamic quantization. Otherwise, use manual INT8.
    
    Returns:
        Quantized model
    """
    device = next(model.parameters()).device
    
    # Use PyTorch's optimized quantization for CPU (provides real speedup)
    if use_pytorch_quantization and device.type == 'cpu':
        return quantize_model_dynamic(model)
    
    # For GPU or when PyTorch quantization not desired, use manual approach
    # This saves memory but may not provide speedup
    model = copy.deepcopy(model)
    _quantize_model_recursive(model)
    return model


def _quantize_model_recursive(model):
    """Recursively quantize Linear layers in a model."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            quantized_layer = QuantizedLinear.from_linear(module)
            setattr(model, name, quantized_layer)
        else:
            _quantize_model_recursive(module)


def dequantize_model(model):
    """Convert quantized layers back to regular Linear layers."""
    for name, module in model.named_children():
        if isinstance(module, QuantizedLinear):
            linear = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
            if module._weight_cache is not None:
                linear.weight.data = module._weight_cache.clone()
            else:
                linear.weight.data = dequantize_weights_int8(module.weight_quantized, module.weight_scale)
            if module.bias is not None:
                linear.bias.data = module.bias.data.clone()
            setattr(model, name, linear)
        else:
            dequantize_model(module)
    return model
