import torch.nn as nn
import torchvision.models as models

def get_model(model_type):
    """Returns the requested model with BatchNorm replaced by GroupNorm"""
    if model_type == "resnet152":
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    elif model_type == "efficientnet-b6":
        model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.DEFAULT)
    else:
        raise ValueError("❌ Invalid model type")

    return convert_batchnorm_to_groupnorm(model)

def convert_batchnorm_to_groupnorm(model):
    """Recursively replaces BatchNorm with GroupNorm, ensuring divisibility."""
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            num_groups = min(num_channels, 32)
            while num_channels % num_groups != 0:
                num_groups -= 1
            setattr(model, name, nn.GroupNorm(num_groups, num_channels))
        else:
            convert_batchnorm_to_groupnorm(module)  # Recursively check submodules
    return model
