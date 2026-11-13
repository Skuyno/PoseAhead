"""Inference-only convolution storage; preserve BatchNorm storage and the geometry path."""

CROP_PRECISIONS = ("autocast", "fp16_conv")


def set_convolution_storage(model, precision, torch):
    """Pre-store the same FP16 convolution operands CUDA autocast would create.

    Only Conv2d/Linear parameters change dtype; BatchNorm running statistics and
    affine parameters are intentionally untouched. Call before inference, never
    during training. Moving back to FP32 does not recover original precision:
    benchmarks must reload an original FP32 state, not round-trip FP16 weights.
    """
    if not isinstance(precision, str) or precision not in CROP_PRECISIONS:
        raise ValueError("unsupported crop_precision; use autocast or fp16_conv")
    if model.training:
        raise ValueError("convolution storage optimization requires an eval model")
    dtype = torch.float32 if precision == "autocast" else torch.float16
    count = 0
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            layer.to(dtype=dtype)
            count += 1
    if not count:
        raise ValueError("no convolution/linear layers found")
    return count
