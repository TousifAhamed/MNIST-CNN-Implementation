import torch
import pytest
import torch.nn.functional as F
from MNIST_CNN import Net, test_loader, device

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    model = Net()
    total_params = count_parameters(model)
    print(f'\nModel Parameter Count Test:')
    print(f'Total trainable parameters: {total_params}')
    assert total_params < 20000, f"Model has {total_params} parameters, which exceeds the limit of 20,000"

def test_batch_normalization():
    model = Net()
    has_batch_norm = any(isinstance(module, torch.nn.BatchNorm2d) 
                        for module in model.modules())
    print('\nBatch Normalization Test:')
    print('Batch Normalization layers found:', has_batch_norm)
    assert has_batch_norm, "Model must use Batch Normalization"

def test_dropout():
    model = Net()
    dropout_layers = [module for module in model.modules() 
                     if isinstance(module, torch.nn.Dropout)]
    print('\nDropout Test:')
    print(f'Number of Dropout layers found: {len(dropout_layers)}')
    print(f'Dropout rates: {[layer.p for layer in dropout_layers]}')
    assert len(dropout_layers) > 0, "Model must use Dropout"

def test_gap_or_fc():
    model = Net()
    has_gap = any(isinstance(module, torch.nn.AdaptiveAvgPool2d) 
                  for module in model.modules())
    has_fc = any(isinstance(module, torch.nn.Linear) 
                 for module in model.modules())
    print('\nGAP/FC Layer Test:')
    print(f'Global Average Pooling found: {has_gap}')
    print(f'Fully Connected Layer found: {has_fc}')
    assert has_gap or has_fc, "Model must use either Global Average Pooling or Fully Connected Layer"

def test_architecture_sequence():
    model = Net()
    
    # Check if BatchNorm follows Conv2d
    conv_bn_pairs = []
    prev_layer = None
    for module in model.modules():
        if isinstance(prev_layer, torch.nn.Conv2d) and isinstance(module, torch.nn.BatchNorm2d):
            conv_bn_pairs.append((prev_layer, module))
        prev_layer = module
    
    print('\nArchitecture Sequence Test:')
    print(f'Number of Conv-BatchNorm pairs found: {len(conv_bn_pairs)}')
    assert len(conv_bn_pairs) > 0, "Conv2d layers should be followed by BatchNorm2d"

def test_model_performance():
    model = Net().to(device)
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nModel Performance Test:')
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    # Add assertions for minimum performance requirements
    assert accuracy > 90.0, f"Model accuracy ({accuracy:.2f}%) is below minimum requirement of 90%"
    assert test_loss < 0.5, f"Test loss ({test_loss:.4f}) is above maximum threshold of 0.5"

if __name__ == "__main__":
    print("Running Model Architecture Tests...")
    test_parameter_count()
    test_batch_normalization()
    test_dropout()
    test_gap_or_fc()
    test_architecture_sequence()
    test_model_performance()
    print("\nAll tests passed successfully!")