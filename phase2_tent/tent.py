"""BatchNorm configuration for test-time entropy minimization."""

from torch import nn


def configure_tent(model: nn.Module) -> nn.Module:
    """Freeze the model except BatchNorm2d affine parameters.

    BatchNorm layers use the current batch's statistics and do not update
    their stored running statistics.
    """
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad = False

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            if module.weight is not None:
                module.weight.requires_grad = True
            if module.bias is not None:
                module.bias.requires_grad = True

    return model
