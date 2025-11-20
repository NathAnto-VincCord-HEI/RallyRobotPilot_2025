import torch
import numpy as np
from PIL import Image
from cnn_model import CNNModel


class NNMsgProcessor:
    """Message processor using CNN model"""
    
    def __init__(self, model_path='cnnXY.pth', device='cpu', debug=False):
        self.model_path = model_path
        self.device = torch.device(device)
        self.debug = debug
        
        # Initialize model
        self.model = CNNModel(dropout=0.5)
        self.model.to(self.device)
        
        # Load trained weights
        print(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        # CRITICAL: Force eval mode to disable BatchNorm training behavior
        self.model.eval()
        
        # Double-check: explicitly set all BatchNorm layers to eval
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        
        if self.debug:
            print(f"✓ Model loaded on {self.device}")
            print(f"✓ Model in eval mode: {not self.model.training}")
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"✓ Model parameters: {total_params:,}")

    def process_msg(self, msg):
        """
        Process sensing message and return control predictions
        
        Args:
            msg: SensingSnapshot with .image attribute (PIL Image)
            
        Returns:
            dict with 'forward', 'back', 'left', 'right' predictions (0-1)
        """
        # Convert PIL image to numpy array
        if isinstance(msg.image, Image.Image):
            image_np = np.array(msg.image)
        else:
            image_np = msg.image
            
        # Normalize to [0, 1] and convert to float32
        if image_np.dtype == np.uint8:
            image_np = image_np.astype(np.float32) / 255.0
        
        # Convert from HWC to CHW format
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            image_np = np.transpose(image_np, (2, 0, 1))
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            predictions = torch.sigmoid(outputs).squeeze(0).cpu().numpy()
        
        # Convert to dictionary
        controls = {
            'forward': float(predictions[0]),
            'back': float(predictions[1]),
            'left': float(predictions[2]),
            'right': float(predictions[3])
        }
        
        if self.debug:
            print(f"Predictions: F={controls['forward']:.3f}, B={controls['back']:.3f}, "
                  f"L={controls['left']:.3f}, R={controls['right']:.3f}")
        
        return controls

    def nn_infer(self, message, threshold=0.5):
        """
        Inference method compatible with direct_controller.py
        Returns boolean controls based on prediction threshold
        
        Args:
            message: SensingSnapshot with .image attribute
            threshold: Prediction threshold for activating controls (default 0.5)
            
        Returns:
            dict with 'forward', 'back', 'left', 'right' as booleans
        """
        # Get float predictions
        predictions = self.process_msg(message)
        
        # Convert to boolean controls using threshold
        controls = {
            'forward': predictions['forward'] > threshold,
            'back': predictions['back'] > threshold,
            'left': predictions['left'] > threshold,
            'right': predictions['right'] > threshold
        }
        
        if self.debug:
            active = [k for k, v in controls.items() if v]
            print(f"Active controls: {active if active else 'none'}")
        
        return controls


def test_processor():
    """Test the processor with a dummy image"""
    print("Testing NNMsgProcessor...")
    
    # Create dummy sensing message
    class DummyMsg:
        def __init__(self):
            self.image = Image.new('RGB', (160, 224), color='red')
    
    # Test with CPU
    processor = NNMsgProcessor(
        model_path='cnnXY.pth',
        device='cpu',
        debug=True
    )
    
    msg = DummyMsg()
    controls = processor.process_msg(msg)
    
    print("\n✓ Test passed!")
    print(f"Controls: {controls}")
    
    # Test variation with different inputs
    print("\nTesting variation with different colored images...")
    colors = ['red', 'green', 'blue', 'black', 'white']
    for color in colors:
        msg.image = Image.new('RGB', (160, 224), color=color)
        controls = processor.process_msg(msg)
        print(f"{color:6s}: F={controls['forward']:.3f}, B={controls['back']:.3f}, "
              f"L={controls['left']:.3f}, R={controls['right']:.3f}")


if __name__ == '__main__':
    test_processor()
