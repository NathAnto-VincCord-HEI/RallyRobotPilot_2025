import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLargerModel(nn.Module):    
    def __init__(self, in_channels=3, num_classes=4, dropout=0.4):
        super(CNNLargerModel, self).__init__()
        
        # Conv block 1: 224x160 -> 112x80
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Conv block 2: 112x80 -> 56x40
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv block 3: 56x40 -> 28x20
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Conv block 4: 28x20 -> 14x10
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Conv block 5: 14x10 -> 7x5
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Flattened: 512 * 7 * 5 = 17,920
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512 * 7 * 5, 512)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 128)
        self.dropout3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Block 5
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        # Flatten
        x = x.reshape(x.shape[0], -1)
        
        # FC layers with dropout
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        
        x = self.dropout3(x)
        x = self.fc3(x)
        
        return x


if __name__ == "__main__":
    # Test the model
    model = CNNModel()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"CNN Model:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 160)
    output = model(dummy_input)
    print(f"\n  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"\n✓ Model architecture validated!")
