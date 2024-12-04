import torch
import pytest
import torch.nn.functional as F
from MNIST_CNN import Net, test_loader, device, train, train_loader, optimizer, scheduler
import MNIST_CNN  # Import the module to access the test function differently

def count_parameters(model):
    """Count the trainable parameters in the model using torchsummary"""
    from torchsummary import summary
    print("\nModel Summary:")
    summary(model, input_size=(1, 28, 28))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    """Test 1: Check if model has less than 20k parameters"""
    model = Net().to(device)
    total_params = count_parameters(model)
    print(f'\nModel Parameter Count Test:')
    print(f'Total trainable parameters: {total_params}')
    assert total_params < 20000, f"Model has {total_params} parameters, which exceeds the limit of 20,000"

def test_batch_normalization():
    """Test 2: Verify use of Batch Normalization"""
    model = Net()
    has_batch_norm = any(isinstance(module, torch.nn.BatchNorm2d) 
                        for module in model.modules())
    print('\nBatch Normalization Test:')
    print('Batch Normalization layers found:', has_batch_norm)
    assert has_batch_norm, "Model must use Batch Normalization"

def test_dropout():
    """Test 3: Verify use of Dropout"""
    model = Net()
    dropout_layers = [module for module in model.modules() 
                     if isinstance(module, torch.nn.Dropout)]
    print('\nDropout Test:')
    print(f'Number of Dropout layers found: {len(dropout_layers)}')
    print(f'Dropout rates: {[layer.p for layer in dropout_layers]}')
    assert len(dropout_layers) > 0, "Model must use Dropout"

def test_gap_or_fc():
    """Test 4: Verify use of GAP or FC layer"""
    model = Net()
    has_gap = any(isinstance(module, torch.nn.AdaptiveAvgPool2d) 
                  for module in model.modules())
    has_fc = any(isinstance(module, torch.nn.Linear) 
                 for module in model.modules())
    print('\nGAP/FC Layer Test:')
    print(f'Global Average Pooling found: {has_gap}')
    print(f'Fully Connected Layer found: {has_fc}')
    assert has_gap or has_fc, "Model must use either Global Average Pooling or Fully Connected Layer"

def test_depthwise_separable():
    """Test 5: Verify use of Depthwise Separable Convolutions"""
    from MNIST_CNN import DepthwiseSeparableConv
    model = Net()
    has_depthwise = any(isinstance(module, DepthwiseSeparableConv) 
                       for module in model.modules())
    print('\nDepthwise Separable Convolution Test:')
    print('Depthwise Separable Convolutions found:', has_depthwise)
    assert has_depthwise, "Model must use Depthwise Separable Convolutions"

def run_training_test():
    """Run one epoch of training and testing"""
    model = Net().to(device)
    print("\nRunning one epoch of training...")
    train(model, device, train_loader, optimizer, 1)
    MNIST_CNN.test(model, device, test_loader)  # Use the module to access the test function

if __name__ == "__main__":
    print("Running Model Architecture Tests...")
    test_parameter_count()
    test_batch_normalization()
    test_dropout()
    test_gap_or_fc()
    test_depthwise_separable()
    run_training_test()
    print("\nAll tests passed successfully!")