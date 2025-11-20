"""
Visualize model predictions on sample images to understand what the model is learning.
This helps debug cases where the model makes mistakes.

Usage:
    python scripts/visualize_predictions.py models/cnn3b.pth

This will:
- Load a trained model
- Sample images from test data
- Show the image, true label, and predicted label
- Highlight misclassifications in red
"""

import sys
import numpy as np
import pickle
import lzma
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from cnn_model import CNNModel


def load_model(model_path, device='cpu'):
    """Load trained model"""
    model = CNNModel(dropout=0.4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def load_test_data(num_samples=50):
    """Load a sample of test data"""
    print("Loading test data...")
    
    all_images = []
    all_labels = []
    
    records = sorted(Path("./").glob("record_*.npz"))
    
    for record in records:
        with lzma.open(record, "rb") as file:
            import sys
            
            rallyrobopilot_modules = {k: v for k, v in sys.modules.items() 
                                     if k.startswith('rallyrobopilot')}
            for module_name in rallyrobopilot_modules:
                del sys.modules[module_name]
            
            try:
                class SafeUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if name == 'SensingSnapshot':
                            class SensingSnapshot:
                                def __init__(self):
                                    self.image = None
                                    self.current_controls = None
                            return SensingSnapshot
                        return super().find_class(module, name)
                
                data = SafeUnpickler(file).load()
                
                for snapshot in data:
                    if hasattr(snapshot, 'image') and hasattr(snapshot, 'current_controls'):
                        if snapshot.image is not None and snapshot.current_controls is not None:
                            all_images.append(snapshot.image)
                            all_labels.append(snapshot.current_controls)
                
            finally:
                sys.modules.update(rallyrobopilot_modules)
    
    # Sample random subset
    indices = np.random.choice(len(all_images), min(num_samples, len(all_images)), replace=False)
    
    X = np.array([all_images[i] for i in indices])
    y = np.array([all_labels[i] for i in indices])
    
    print(f"Loaded {len(X)} samples")
    return X, y


def predict_batch(model, images, device='cpu'):
    """Make predictions on a batch of images"""
    # Preprocess images
    X = images.transpose(0, 3, 1, 2)  # (N, C, H, W)
    X = X / 255.0  # Normalize
    X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
    
    return predictions


def visualize_predictions(model, X, y_true, save_path, device='cpu'):
    """Create visualization of predictions vs ground truth"""
    
    print("Making predictions...")
    y_pred = predict_batch(model, X, device)
    
    control_names = ['Fwd', 'Back', 'Left', 'Right']
    
    # Find interesting cases
    correct_indices = []
    incorrect_indices = []
    
    for i in range(len(y_pred)):
        if np.array_equal(y_pred[i], y_true[i]):
            correct_indices.append(i)
        else:
            incorrect_indices.append(i)
    
    print(f"Correct predictions: {len(correct_indices)}/{len(y_pred)}")
    print(f"Incorrect predictions: {len(incorrect_indices)}/{len(y_pred)}")
    
    # Show mix of correct and incorrect predictions
    num_samples = min(20, len(X))
    
    # Try to show mix: half incorrect, half correct
    num_incorrect = min(num_samples // 2, len(incorrect_indices))
    num_correct = num_samples - num_incorrect
    
    selected_indices = []
    if len(incorrect_indices) > 0:
        selected_indices.extend(np.random.choice(incorrect_indices, num_incorrect, replace=False))
    if len(correct_indices) > 0:
        remaining = num_samples - len(selected_indices)
        selected_indices.extend(np.random.choice(correct_indices, min(remaining, len(correct_indices)), replace=False))
    
    # Create visualization
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Model Predictions vs Ground Truth\n(Red border = Incorrect prediction)', 
                 fontsize=16)
    
    for plot_idx, idx in enumerate(selected_indices):
        if plot_idx >= 20:
            break
        
        row = plot_idx // 5
        col = plot_idx % 5
        ax = axes[row, col]
        
        # Show image
        ax.imshow(X[idx])
        
        # Check if prediction is correct
        is_correct = np.array_equal(y_pred[idx], y_true[idx])
        
        # Create label text
        true_labels = [control_names[i] for i in range(4) if y_true[idx][i] == 1]
        pred_labels = [control_names[i] for i in range(4) if y_pred[idx][i] == 1]
        
        true_str = '+'.join(true_labels) if true_labels else 'None'
        pred_str = '+'.join(pred_labels) if pred_labels else 'None'
        
        # Set title with color
        title = f"True: {true_str}\nPred: {pred_str}"
        color = 'green' if is_correct else 'red'
        ax.set_title(title, fontsize=10, color=color, weight='bold')
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        
        ax.axis('off')
    
    # Hide unused subplots
    for plot_idx in range(len(selected_indices), 20):
        row = plot_idx // 5
        col = plot_idx % 5
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Visualization saved to {save_path}")


def analyze_error_patterns(y_true, y_pred):
    """Analyze common error patterns"""
    print("\n" + "="*70)
    print("ERROR PATTERN ANALYSIS")
    print("="*70)
    
    control_names = ['Forward', 'Back', 'Left', 'Right']
    
    # Per-class errors
    print("\nPer-class error rates:")
    for i, name in enumerate(control_names):
        errors = np.sum(y_pred[:, i] != y_true[:, i])
        error_rate = errors / len(y_true) * 100
        
        # False positives and false negatives
        fp = np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 0))
        fn = np.sum((y_pred[:, i] == 0) & (y_true[:, i] == 1))
        
        print(f"  {name:8s}: {error_rate:5.1f}% errors (FP={fp}, FN={fn})")
    
    # Most common error patterns
    print("\nMost common error patterns:")
    from collections import Counter
    
    errors = []
    for i in range(len(y_true)):
        if not np.array_equal(y_true[i], y_pred[i]):
            true_str = ''.join(str(x) for x in y_true[i])
            pred_str = ''.join(str(x) for x in y_pred[i])
            errors.append(f"{true_str} → {pred_str}")
    
    if len(errors) > 0:
        error_counts = Counter(errors)
        for error, count in error_counts.most_common(10):
            pct = count / len(y_true) * 100
            print(f"  {error}: {count:3d} times ({pct:4.1f}%)")
    else:
        print("  No errors found!")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_predictions.py <model_path>")
        print("\nExample:")
        print("  python scripts/visualize_predictions.py models/cnn3b.pth")
        return
    
    model_path = sys.argv[1]
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    print("="*70)
    print("MODEL PREDICTION VISUALIZATION")
    print("="*70)
    print(f"\nModel: {model_path}")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(model_path, device)
    print("✓ Model loaded")
    
    # Load test data
    X, y_true = load_test_data(num_samples=100)
    
    # Make predictions
    y_pred = predict_batch(model, X, device)
    
    # Analyze errors
    analyze_error_patterns(y_true, y_pred)
    
    # Create output directory
    output_dir = Path("prediction_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Visualize predictions
    model_name = Path(model_path).stem
    save_path = output_dir / f"{model_name}_predictions.png"
    visualize_predictions(model, X, y_true, save_path, device)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nVisualization saved to: {save_path}")


if __name__ == "__main__":
    main()
