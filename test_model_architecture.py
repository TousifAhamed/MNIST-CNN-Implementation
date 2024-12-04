import torch
import pytest
from MNIST_CNN import Net

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    model = Net()
    total_params = count_parameters(model)
    assert total_params < 20000, f"Model has {total_params} parameters, which exceeds the limit of 20,000"

def test_batch_normalization():
    model = Net()
    has_batch_norm = any(isinstance(module, torch.nn.BatchNorm2d) 
                        for module in model.modules())
    assert has_batch_norm, "Model must use Batch Normalization"

def test_dropout():
    model = Net()
    has_dropout = any(isinstance(module, torch.nn.Dropout) 
                     for module in model.modules())
    assert has_dropout, "Model must use Dropout"

def test_gap_or_fc():
    model = Net()
    has_gap = any(isinstance(module, torch.nn.AdaptiveAvgPool2d) 
                  for module in model.modules())
    has_fc = any(isinstance(module, torch.nn.Linear) 
                 for module in model.modules())
    assert has_gap or has_fc, "Model must use either Global Average Pooling or Fully Connected Layer"

def test_architecture_sequence():
    model = Net()
    
    # Check if BatchNorm follows Conv2d
    conv_followed_by_bn = False
    prev_layer = None
    for module in model.modules():
        if isinstance(prev_layer, torch.nn.Conv2d) and isinstance(module, torch.nn.BatchNorm2d):
            conv_followed_by_bn = True
            break
        prev_layer = module
    
    assert conv_followed_by_bn, "Conv2d layers should be followed by BatchNorm2d"