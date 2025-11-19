"""
Retrain with proper regularization to prevent overfitting.

Based on your training history, the model overfitted after epoch 10.
This script includes:
- Early stopping (stop when validation loss increases)
- Higher dropout (0.4 instead of 0.3)
- L2 regularization (weight decay)
- Learning rate scheduling
- Save best model based on validation loss

This version loads data WITHOUT importing Ursina!
"""

import numpy as np
import pickle
import lzma
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import metrics

from proper_cnn_model import ProperCNN


def get_next_model_id():
    """Find the next available model ID by scanning existing model files"""
    # Check both root directory and models folder
    root = Path("./")
    models_dir = root / "models"
    
    existing_ids = []
    
    # Pattern to match cnnXY.pth where X is digits, Y is b or f
    pattern = re.compile(r'cnn(\d+)[bf]\.pth')
    
    # Search in root directory
    for file in root.glob("cnn*.pth"):
        match = pattern.match(file.name)
        if match:
            existing_ids.append(int(match.group(1)))
    
    # Search in models directory if it exists
    if models_dir.exists():
        for file in models_dir.glob("cnn*.pth"):
            match = pattern.match(file.name)
            if match:
                existing_ids.append(int(match.group(1)))
    
    # Return next ID (or 1 if no models exist)
    return max(existing_ids) + 1 if existing_ids else 1


class CarDataset(Dataset):
    """Simple dataset for car images and controls"""
    def __init__(self, X, y):
        self.features = torch.from_numpy(X.astype(np.float32))
        self.labels = torch.from_numpy(y.astype(np.float32))
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.features)


def load_data():
    """Load and preprocess training data WITHOUT Ursina"""
    print("="*70)
    print("LOADING TRAINING DATA")
    print("="*70)
    
    # We'll extract data manually to avoid Ursina imports
    all_images = []
    all_labels = []
    
    records = sorted(Path("./").glob("record_*.npz"))
    
    for record in records:
        with lzma.open(record, "rb") as file:
            # Custom unpickler that skips Ursina classes
            import sys
            
            # Temporarily remove rallyrobopilot from sys.modules to prevent Ursina import
            rallyrobopilot_modules = {k: v for k, v in sys.modules.items() if k.startswith('rallyrobopilot')}
            for module_name in rallyrobopilot_modules:
                del sys.modules[module_name]
            
            try:
                # Create a custom unpickler
                class SafeUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        # For SensingSnapshot, create a simple substitute class
                        if name == 'SensingSnapshot':
                            class SensingSnapshot:
                                def __init__(self):
                                    self.image = None
                                    self.current_controls = None
                            return SensingSnapshot
                        return super().find_class(module, name)
                
                data = SafeUnpickler(file).load()
                
                # Extract images and controls
                valid_count = 0
                for snapshot in data:
                    if hasattr(snapshot, 'image') and hasattr(snapshot, 'current_controls'):
                        if snapshot.image is not None and snapshot.current_controls is not None:
                            all_images.append(snapshot.image)
                            all_labels.append(snapshot.current_controls)
                            valid_count += 1
                
                print(f"  {record.name}: {valid_count} valid snapshots")
                
            finally:
                # Restore modules
                sys.modules.update(rallyrobopilot_modules)
    
    print(f"\nTotal valid snapshots: {len(all_images)}")
    
    # Convert to numpy arrays
    X = np.array(all_images)  # (N, H, W, C)
    y = np.array(all_labels)  # (N, 4)
    
    # Normalize images
    X = X.transpose(0, 3, 1, 2)  # (N, C, H, W)
    X = X / 255.0  # Normalize to [0, 1]
    
    print(f"  Images shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    
    # Class distribution
    print(f"\nLabel distribution:")
    for i, name in enumerate(['Forward', 'Back', 'Left', 'Right']):
        count = np.sum(y[:, i])
        pct = 100 * count / len(y)
        print(f"  {name:8s}: {count:5d} ({pct:5.1f}%)")
    
    return X, y


def train_with_early_stopping():
    """Train model with proper regularization"""
    # Get next model ID and setup paths
    model_id = get_next_model_id()
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    best_model_path = models_dir / f"cnn{model_id}b.pth"
    final_model_path = models_dir / f"cnn{model_id}f.pth"
    history_plot_path = models_dir / f"cnn{model_id}_history.png"
    confusion_matrix_path = models_dir / f"cnn{model_id}_conf_matrix.png"
    
    print("\n" + "="*70)
    print("TRAINING SETUP")
    print("="*70)
    print(f"\nModel ID: {model_id}")
    print(f"Best model will be saved to: {best_model_path}")
    print(f"Final model will be saved to: {final_model_path}")
    print(f"History plot will be saved to: {history_plot_path}")
    print(f"Confusion matrix will be saved to: {confusion_matrix_path}")
    
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Val set:   {len(X_val)} samples")
    
    # Create datasets
    train_dataset = CarDataset(X_train, y_train)
    val_dataset = CarDataset(X_val, y_val)
    
    # DataLoaders with appropriate batch size
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    model = ProperCNN(dropout=0.4)
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    # Loss and optimizer with weight decay (L2 regularization)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    num_epochs = 5 # 30
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }
    
    # Storage for confusion matrix data
    train_all_preds = []
    train_all_labels = []
    val_all_preds = []
    val_all_labels = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Reset for this epoch
        epoch_train_preds = []
        epoch_train_labels = []
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.numel()
        
            # Store for confusion matrix (last epoch only)
            if epoch == num_epochs - 1 or patience_counter >= patience - 1:
                epoch_train_preds.append(preds.cpu().numpy())
                epoch_train_labels.append(labels.cpu().numpy())


        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Reset for this epoch
        epoch_val_preds = []
        epoch_val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()

                # Store for confusion matrix (last epoch only)
                if epoch == num_epochs - 1 or patience_counter >= patience - 1:
                    epoch_val_preds.append(preds.cpu().numpy())
                    epoch_val_labels.append(labels.cpu().numpy())
        
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        
        # Store final epoch predictions
        if epoch == num_epochs - 1 or patience_counter >= patience - 1:
            train_all_preds = np.concatenate(epoch_train_preds, axis=0)
            train_all_labels = np.concatenate(epoch_train_labels, axis=0)
            val_all_preds = np.concatenate(epoch_val_preds, axis=0)
            val_all_labels = np.concatenate(epoch_val_labels, axis=0)
        
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
            print(f"  Best validation loss: {best_val_loss:.4f}")
            break
    
    

    # Save final model
    torch.save(model.state_dict(), final_model_path)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History - Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training History - Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(history_plot_path)
    print(f"\n✓ Training history saved to {history_plot_path}")

    # Plot Confusion Matrices for each output class
    control_names = ['Forward', 'Back', 'Left', 'Right']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Confusion Matrices - Train (top) and Val (bottom)', fontsize=16)
    
    for i, control_name in enumerate(control_names):
        # Train confusion matrix
        train_cm = metrics.confusion_matrix(
            train_all_labels[:, i],
            train_all_preds[:, i]
        )
        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=train_cm,
            display_labels=['Off', 'On']
        )
        cm_display.plot(cmap='Blues')
        cm_display.plot(ax=axes[0, i], colorbar=False)
        axes[0, i].set_title(f'Train: {control_name}')
        
        # Val confusion matrix
        val_cm = metrics.confusion_matrix(
            val_all_labels[:, i],
            val_all_preds[:, i]
        )
        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=val_cm,
            display_labels=['Off', 'On']
        )
        cm_display.plot(cmap='Blues')
        cm_display.plot(ax=axes[1, i], colorbar=False)
        axes[1, i].set_title(f'Val: {control_name}')

    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    print(f"✓ Confusion matrices saved to {confusion_matrix_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest model saved as: {best_model_path}")
    print(f"Final model saved as: {final_model_path}")
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print("\nNext step:")
    print("  python scripts/test_proper_model.py")


if __name__ == "__main__":
    train_with_early_stopping()
    