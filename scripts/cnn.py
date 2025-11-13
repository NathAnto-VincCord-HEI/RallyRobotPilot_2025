import numpy
import pickle
import lzma
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# !pip install torchvision
import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# !pip install torchmetrics
import torchmetrics

from sklearn.model_selection import train_test_split

from data import Data
from cnn_model import CNN


# Loading and Preprocessing the dataset

batch_size = 64

snapshots = []

# Get all files containing "record_" in the filename and ending with .npz
records = sorted(Path("./").glob("record_*.npz"))

for record in records:
    try:
        with lzma.open(record, "rb") as file:
            data = pickle.load(file)
            print("Processing file:", record.name, "with", len(data), "snapshots")
            snapshots.extend(data)
    except Exception as e:
        print(f"Error processing {record.name}: {e}")
        continue

print(f"Number of snapshots: {len(snapshots)}")

valid_snapshots = [
    s for s in snapshots
    if s.image is not None and s.current_controls is not None
]

print(f"Total original snapshots: {len(snapshots)}")
print(f"Snapshots remaining after filtering: {len(valid_snapshots)}")

X_list = [s.image for s in valid_snapshots]
y_list = [s.current_controls for s in valid_snapshots]

print(f"Length of X_list: {len(X_list)}")
print(f"Length of y_list: {len(y_list)}")

features = np.array(X_list)
labels = np.array(y_list, dtype=np.float32)

single_image = features[0]
print(f"Shape of a single image: {single_image.shape}")

# Transpose the last three dimensions: (H, W, C) -> (C, H, W)
features = np.transpose(features, (0, 3, 1, 2))

# The new shape should be (352, 3, 1024, 1280)
print(f"New features shape: {features.shape}")

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# # Further split training data into training (80%) and validation (20%)
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=0.2, random_state=42
# )

# Create Dataloaders
train_data = Data(X_train, y_train, augment=True)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# val_data = Data(X_val, y_val)
# val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

def imshow(img):
   npimg = img.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()

# get some random training images
dataiter = iter(train_dataloader)
images, labels = next(dataiter)
# labels
# show images
imshow(torchvision.utils.make_grid(images))

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(in_channels=3, num_classes=4).to(device)
print(model)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs=10
for epoch in range(num_epochs):
 # Iterate over training batches
   print(f"Epoch [{epoch + 1}/{num_epochs}]")

   for batch_index, (data, targets) in enumerate(tqdm(train_dataloader)):
       data = data.to(device)
       targets = targets.to(device)
       scores = model(data)
       loss = criterion(scores, targets)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

# Set up of multiclass accuracy metric
acc = torchmetrics.Accuracy(task="multiclass",num_classes=4)

# Iterate over the dataset batches
model.eval()
with torch.no_grad():
   for images, labels in test_dataloader:
       # Get predicted probabilities for test data batch
       outputs = model(images)
       _, preds = torch.max(outputs, 1)
       acc(preds, labels)
       torchmetrics.Precision(preds, labels)
       torchmetrics.Recall(preds, labels)

#Compute total test accuracy
test_accuracy = acc.compute()
print(f"Test accuracy: {test_accuracy}")