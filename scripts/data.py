import numpy as np
import torch
from torch.utils.data import Dataset

class Data(Dataset):
    # Add an 'augment' flag to the constructor
    def __init__(self, X, y, augment=False):
        self.features = torch.from_numpy(X.astype(np.float32))
        self.labels = torch.from_numpy(y.astype(np.float32))
        self.len = self.features.shape[0]
        self.augment = augment # Store the flag
       
    def __getitem__(self, idx):
        image = self.features[idx]
        label = self.labels[idx]

        # Crucial: Check if image or label is None right before returning
        # (This is a safeguard, but cleaning the input data is better)
        if image is None or label is None:
            # If you somehow missed a None, skip it or raise an error
            # A common fix is to clean the input data as shown in Step 1.
            # If you must handle it here, you need to return valid data,
            # or the DataLoader will break. The best solution is to clean the input.
            pass # The best approach is to ensure this code is never reached

        # ... (apply transforms, convert to tensor)
        return image, label
   
    def __len__(self):
        return self.len