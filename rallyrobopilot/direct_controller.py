"""
Direct Controller - Handles recording, autopilot, and controls directly in Ursina app
No socket communication needed - everything runs in the same process
"""

from ursina import *
import numpy as np
import os
import pickle
import lzma
import time

# Import sensing message for data structure
from .sensing_message import SensingSnapshot

# Check for PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DirectController(Entity):
    """
    Controller that manages recording, autopilot, and manual controls
    directly within the Ursina application without socket communication.
    """
    
    def __init__(self, car=None, enable_autopilot=False, enable_recording=False, 
                 autopilot_model_path="car_cnn.pth", record_images=True, 
                 autopilot_inference_hz=10):
        """
        Initialize the DirectController.
        
        Args:
            car: The Car instance to control
            enable_autopilot: Start with autopilot enabled
            enable_recording: Start with recording enabled
            autopilot_model_path: Path to the CNN model file
            record_images: Whether to record images in snapshots
            autopilot_inference_hz: Frequency of autopilot inference (default: 10Hz for performance)
        """
        super().__init__()
        
        self.car = car
        self.record_images = record_images
        
        # Recording state
        self.recording = False
        self.recorded_data = []
        self.recording_enabled_at_start = enable_recording
        self.sensing_period = 0.1  # Record at 10Hz (same as old remote controller)
        self.last_sensing_time = 0
        
        # Autopilot state
        self.autopilot_active = False
        self.autopilot_processor = None  # Will hold NNMsgProcessor instance
        self.autopilot_enabled_at_start = enable_autopilot
        self.autopilot_model_path = autopilot_model_path
        self.autopilot_inference_period = 1.0 / autopilot_inference_hz  # Convert Hz to period
        self.last_autopilot_inference_time = 0
        
        # Current autopilot control state
        self.autopilot_controls = {
            "forward": False,
            "back": False,
            "left": False,
            "right": False
        }
        
        # Load autopilot model if enabled
        if enable_autopilot:
            self.load_autopilot_model()
        
        # Status text displays
        self._create_status_displays()
        
        # Auto-start recording if enabled
        if enable_recording:
            self.start_recording()
    
    def _create_status_displays(self):
        """Create on-screen status indicators"""
        # Recording status (top-right)
        self.recording_text = Text(
            text="",
            position=(0.7, 0.35),
            origin=(0, 0),
            scale=1.5,
            color=color.red,
            enabled=False
        )
        
        # Autopilot status (top-right, below recording)
        self.autopilot_text = Text(
            text="",
            position=(0.7, 0.30),
            origin=(0, 0),
            scale=1.5,
            color=color.green,
            enabled=False
        )
    
    def load_autopilot_model(self):
        """Load the CNN autopilot model using NNMsgProcessor from autopilot.py or improved_autopilot.py"""
        try:
            import sys
            from pathlib import Path
            
            # Add scripts directory to path if not already there
            scripts_path = Path(__file__).parent.parent / "scripts"
            if str(scripts_path) not in sys.path:
                sys.path.insert(0, str(scripts_path))
            
            # Try to load autopilot based on model filename
            try:
                # import CNN model
                from autopilot import NNMsgProcessor
                self.autopilot_processor = NNMsgProcessor(
                    model_path=self.autopilot_model_path,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    debug=True
                )
                print("[+] Loaded CNN autopilot")
                
            except ImportError as e:
                # Fall back to regular autopilot
                print(f"[!] Autopilot not found: {e}")
                return False
                
            # Show device info
            device = self.autopilot_processor.device
            print(f"[+] Autopilot model: {self.autopilot_model_path}")
            print(f"[+] Running on device: {device}")
            if device == "cpu":
                print(f"[!] WARNING: Running on CPU. For better performance, install CUDA-enabled PyTorch")
            return True
            
        except Exception as e:
            print(f"[X] Failed to load autopilot: {e}")
            import traceback
            traceback.print_exc()
            self.autopilot_processor = None
            return False
    
    def start_recording(self):
        """Start recording sensor data"""
        if not self.recording:
            self.recording = True
            self.recorded_data = []
            self.recording_text.text = "● RECORDING"
            self.recording_text.enabled = True
            print("[+] Recording started")
    
    def stop_recording(self):
        """Stop recording and save data"""
        if self.recording:
            self.recording = False
            self.recording_text.enabled = False
            
            if len(self.recorded_data) > 0:
                self.save_recording()
            else:
                print("[!] No data recorded")
    
    def save_recording(self):
        """Save recorded data to file"""
        # Find next available filename
        record_name = "record_%d.npz"
        fid = 0
        while os.path.exists(record_name % fid):
            fid += 1
        
        filename = record_name % fid
        
        try:
            with lzma.open(filename, "wb") as f:
                pickle.dump(self.recorded_data, f)
            
            print(f"[+] Recorded {len(self.recorded_data)} snapshots saved to {filename}")
            self.recorded_data = []
            
        except Exception as e:
            print(f"[X] Failed to save recording: {e}")
    
    def start_autopilot(self):
        """Start autopilot mode"""
        if not hasattr(self, 'autopilot_processor') or self.autopilot_processor is None:
            if not self.load_autopilot_model():
                print("[X] Cannot start autopilot - processor not loaded")
                return
        
        self.autopilot_active = True
        self.autopilot_text.text = "[AUTO]"
        self.autopilot_text.enabled = True
        print("[+] Autopilot activated")
    
    def stop_autopilot(self):
        """Stop autopilot mode"""
        if self.autopilot_active:
            self.autopilot_active = False
            self.autopilot_text.enabled = False
            
            # Release all autopilot controls
            for key in ['w', 's', 'a', 'd']:
                if held_keys[key] and not self._is_manual_key_pressed(key):
                    held_keys[key] = False
            
            print("[+] Autopilot deactivated")
    
    def _is_manual_key_pressed(self, key):
        """Check if a key is actually being pressed manually (not just set by autopilot)"""
        # This is a workaround since Ursina doesn't distinguish between
        # programmatically set keys and physically pressed keys
        # In practice, we assume manual override if autopilot is active
        return False
    
    def capture_sensor_data(self):
        """Capture current sensor data snapshot"""
        if self.car is None:
            return None
        
        snapshot = SensingSnapshot()
        
        # Capture controls (current state)
        snapshot.current_controls = (
            held_keys['w'] or held_keys["up arrow"],
            held_keys['s'] or held_keys["down arrow"],
            held_keys['a'] or held_keys["left arrow"],
            held_keys['d'] or held_keys["right arrow"]
        )
        
        # Capture car state
        snapshot.car_position = self.car.world_position
        snapshot.car_speed = self.car.speed
        snapshot.car_angle = self.car.rotation_y
        snapshot.raycast_distances = self.car.multiray_sensor.collect_sensor_values()
        
        # Capture image if enabled
        if self.record_images:
            tex = base.win.getDisplayRegion(0).getScreenshot()
            arr = tex.getRamImageAs("RGB")
            data = np.frombuffer(arr, np.uint8)
            image = data.reshape(tex.getYSize(), tex.getXSize(), 3)
            image = image[::-1, :, :].copy()  # Invert Y axis and make contiguous copy for PyTorch
            snapshot.image = image
        else:
            snapshot.image = None
        
        return snapshot
    
    def run_autopilot_inference(self):
        """Run autopilot model inference and update controls using NNMsgProcessor"""
        if not self.autopilot_active or not hasattr(self, 'autopilot_processor') or self.autopilot_processor is None or self.car is None:
            return
        
        # Throttle inference to avoid performance issues
        current_time = time.time()
        if current_time - self.last_autopilot_inference_time < self.autopilot_inference_period:
            return  # Skip this frame
        
        self.last_autopilot_inference_time = current_time
        
        try:
            # Capture only the image for inference (lightweight)
            from .sensing_message import SensingSnapshot
            snapshot = SensingSnapshot()
            
            # Capture just the image (required for autopilot)
            tex = base.win.getDisplayRegion(0).getScreenshot()
            arr = tex.getRamImageAs("RGB")
            data = np.frombuffer(arr, np.uint8)
            image = data.reshape(tex.getYSize(), tex.getXSize(), 3)
            snapshot.image = image[::-1, :, :].copy()  # Invert Y axis and make contiguous copy
            
            # DEBUG: Check image properties
            if not hasattr(self, '_image_debug_shown'):
                print(f"[DEBUG] Captured image shape: {snapshot.image.shape}")
                print(f"[DEBUG] Image dtype: {snapshot.image.dtype}")
                print(f"[DEBUG] Image value range: [{snapshot.image.min()}, {snapshot.image.max()}]")
                print(f"[DEBUG] Image mean: {snapshot.image.mean():.2f}")
                print(f"[DEBUG] Image std: {snapshot.image.std():.2f}")
                self._image_debug_shown = True
            
            # Use the NNMsgProcessor's nn_infer method to get desired controls
            desired_controls = self.autopilot_processor.nn_infer(snapshot)
            
            # Apply controls only if state changed
            key_map = {"forward": "w", "back": "s", "left": "a", "right": "d"}
            
            for command, desired in desired_controls.items():
                key = key_map[command]
                
                # Only update if state changed
                if self.autopilot_controls[command] != desired:
                    held_keys[key] = desired
                    self.autopilot_controls[command] = desired
                    
        except Exception as e:
            print(f"[X] Autopilot inference error: {e}")
            import traceback
            traceback.print_exc()
    
    def input(self, key):
        """Handle keyboard input for recording and autopilot controls"""
        
        # Recording controls
        if key == 'r':
            if not self.recording:
                self.start_recording()
        
        elif key == 'e':
            if self.recording:
                self.stop_recording()
        
        # Autopilot controls
        elif key == 'p':
            if not self.autopilot_active:
                self.start_autopilot()
        
        elif key == 'o':
            if self.autopilot_active:
                self.stop_autopilot()
        
        # Manual controls override autopilot
        if self.autopilot_active and key in ['w', 'a', 's', 'd', 'up arrow', 'down arrow', 'left arrow', 'right arrow']:
            # User pressed a manual control key, temporarily override autopilot
            # The autopilot will resume control when the key is released
            pass
    
    def update(self):
        """Update controller state every frame"""
        
        # Run autopilot inference if active
        if self.autopilot_active:
            self.run_autopilot_inference()
        
        # Record data if recording (at specified rate)
        if self.recording and self.car is not None:
            current_time = time.time()
            if current_time - self.last_sensing_time >= self.sensing_period:
                snapshot = self.capture_sensor_data()
                if snapshot is not None:
                    self.recorded_data.append(snapshot)
                    
                    # Update recording text with count
                    self.recording_text.text = f"● REC ({len(self.recorded_data)})"
                    self.last_sensing_time = current_time
