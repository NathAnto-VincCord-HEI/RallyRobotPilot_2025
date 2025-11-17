"""
Quick test script to verify the ProperCNN model works properly
Tests with different colored images to ensure output variation
"""
import torch
import numpy as np
from PIL import Image
from proper_cnn_model import ProperCNN


def test_model_variation(model_path='car_cnn_proper_best.pth'):
    """Test that model produces varying outputs for different inputs"""
    
    print("="*70)
    print("TESTING ProperCNN MODEL")
    print("="*70)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = ProperCNN()
    model.to(device)
    
    print(f"Loading model from: {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}")
        print("\nPlease train the model first:")
        print("  python scripts/train_proper_cnn.py")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Set to eval mode
    model.eval()
    
    # Force all BatchNorm layers to eval mode
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    
    print(f"✓ Model in eval mode")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {total_params:,}")
    
    # Test with different colored images
    print("\n" + "="*70)
    print("TESTING OUTPUT VARIATION")
    print("="*70)
    
    test_colors = [
        ('red', (255, 0, 0)),
        ('green', (0, 255, 0)),
        ('blue', (0, 0, 255)),
        ('yellow', (255, 255, 0)),
        ('cyan', (0, 255, 255)),
        ('magenta', (255, 0, 255)),
        ('white', (255, 255, 255)),
        ('black', (0, 0, 0)),
        ('gray', (128, 128, 128))
    ]
    
    all_outputs = []
    
    print("\nPredictions for different colored images:")
    print("-" * 70)
    
    with torch.no_grad():
        for color_name, rgb in test_colors:
            # Create solid color image (160x224 to match training)
            img = Image.new('RGB', (160, 224), color=rgb)
            img_np = np.array(img).astype(np.float32) / 255.0
            
            # Convert HWC to CHW
            img_np = np.transpose(img_np, (2, 0, 1))
            
            # Add batch dimension and convert to tensor
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)
            
            # Get predictions
            outputs = model(img_tensor)
            predictions = torch.sigmoid(outputs).squeeze(0).cpu().numpy()
            
            all_outputs.append(predictions)
            
            print(f"{color_name:8s}: F={predictions[0]:.4f}  B={predictions[1]:.4f}  "
                  f"L={predictions[2]:.4f}  R={predictions[3]:.4f}")
    
    # Calculate statistics
    print("\n" + "="*70)
    print("OUTPUT STATISTICS")
    print("="*70)
    
    all_outputs = np.array(all_outputs)
    
    for i, label in enumerate(['Forward', 'Back', 'Left', 'Right']):
        values = all_outputs[:, i]
        print(f"\n{label}:")
        print(f"  Mean:   {values.mean():.4f}")
        print(f"  Std:    {values.std():.4f}")
        print(f"  Min:    {values.min():.4f}")
        print(f"  Max:    {values.max():.4f}")
        print(f"  Range:  {values.max() - values.min():.4f}")
    
    # Overall assessment
    print("\n" + "="*70)
    print("ASSESSMENT")
    print("="*70)
    
    overall_std = all_outputs.std()
    print(f"\nOverall standard deviation: {overall_std:.4f}")
    
    if overall_std < 0.001:
        print("✗ FAIL: Model outputs are frozen (std < 0.001)")
        print("  The model is not producing varying outputs")
    elif overall_std < 0.01:
        print("⚠ WARNING: Model outputs have very low variation (std < 0.01)")
        print("  The model may not be working properly")
    elif overall_std < 0.05:
        print("⚠ CAUTION: Model outputs have low variation (std < 0.05)")
        print("  The model may need more training or adjustment")
    else:
        print("✓ PASS: Model outputs show good variation (std >= 0.05)")
        print("  The model appears to be working properly")
    
    print("\n" + "="*70)
    
    return overall_std >= 0.01


def test_with_random_noise(model_path='car_cnn_proper_best.pth'):
    """Test model with random noise patterns"""
    
    print("\n" + "="*70)
    print("TESTING WITH RANDOM NOISE")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProperCNN()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Force BatchNorm to eval
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    
    print("\nPredictions for random noise patterns:")
    print("-" * 70)
    
    with torch.no_grad():
        for i in range(5):
            # Generate random image
            img_np = np.random.rand(3, 224, 160).astype(np.float32)
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)
            
            # Get predictions
            outputs = model(img_tensor)
            predictions = torch.sigmoid(outputs).squeeze(0).cpu().numpy()
            
            print(f"Noise {i+1}: F={predictions[0]:.4f}  B={predictions[1]:.4f}  "
                  f"L={predictions[2]:.4f}  R={predictions[3]:.4f}")


if __name__ == '__main__':
    success = test_model_variation()
    
    if success:
        test_with_random_noise()
        print("\n✓ All tests completed successfully!")
    else:
        print("\n✗ Model testing failed")
        print("\nNext steps:")
        print("1. Train the model: python scripts/train_proper_cnn.py")
        print("2. Run this test again: python scripts/test_proper_model.py")
