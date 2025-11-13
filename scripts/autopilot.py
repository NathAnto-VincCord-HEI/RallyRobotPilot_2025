from PyQt6 import QtWidgets
from cnn_model import CNN
import numpy as np
import torch

from data_collector import DataCollectionUI
r"""
This file is provided as an example of what a simplistic controller could be done.
It simply uses the DataCollectionUI interface zo receive sensing_messages and send controls.

/!\ Be warned that if the processing time of NNMsgProcessor.process_message is superior to the message reception period, a lag between the images processed and commands sent.
One might want to only process the last sensing_message received, etc. 
Be warned that this could also cause crash on the client side if socket sending buffer overflows
ys
/!\ Do not work directly in this file (make a copy and rename it) to prevent future pull from erasing what you write here.
"""


class NNMsgProcessor:
    def __init__(self):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        
        # Load both models
        self.car_model = CNN(3, 4).to(self.device)
        self.car_model.load_state_dict(torch.load("car_cnn.pth", map_location=self.device))
        self.car_model.eval()

        # Initialize current_state (this was missing)
        self.current_state = {
            "forward": False,
            "back": False,
            "left": False,
            "right": False
        }

    def nn_infer(self, message):
        
        """
        Use the neural network to predict control commands from sensor data.
        
        message should have:
            - image: array
        """
        # 2. Convert to tensor and transpose to (C, H, W)
        image_tensor = torch.from_numpy(message.image).float()  # (H, W, C)
        image_tensor = image_tensor.permute(2, 0, 1)  # (C, H, W) = (3, 1024, 1280)

        # 3. Normalize to [0, 1]
        image_tensor = image_tensor / 255.0

        # 4. Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)  # (1, 3, 1024, 1280)

        # 5. Move to device
        image_tensor = image_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.car_model(image_tensor) # Get raw logits
            outputs = torch.sigmoid(logits) # Apply sigmoid to get probabilities
            predictions = (outputs >= 0.5).cpu().numpy()[0]  # Convert to binary and remove batch dim

        # Convert predictions to control dictionary
        return {
            "forward": bool(predictions[0]),
            "back": bool(predictions[1]),
            "left": bool(predictions[2]),
            "right": bool(predictions[3])
        }

    def process_message(self, message, data_collector):
        # Get desired control state from neural network        
        desired_state = self.nn_infer(message)
        
        # Only send commands when state changes
        for command, desired in desired_state.items():
            if self.current_state[command] != desired:
                data_collector.onCarControlled(command, desired)
                self.current_state[command] = desired

if  __name__ == "__main__":
    import sys
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    # Load your trained model (update this path to where you save your model)
    cnn_brain = NNMsgProcessor()
    data_window = DataCollectionUI(cnn_brain.process_message)
    data_window.show()

    app.exec()