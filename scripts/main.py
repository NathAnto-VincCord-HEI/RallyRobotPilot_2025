"""
Main script to launch the Rally Robot Pilot game with integrated controls.

This version removes the Flask server and socket-based remote controller
in favor of direct in-app controls for better performance.

Controls:
    - WASD / Arrow Keys: Manual driving
    - R: Start recording
    - E: Stop recording (and save)
    - P: Start autopilot
    - O: Stop autopilot
    - G: Reset car position
    - V: Toggle raycast visualization
    - ESC: Exit game

Configure settings below before running.
"""

from rallyrobopilot import prepare_game_app
 
# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================

# Track selection
TRACK_NAME = "SimpleTrack"  # Options: "SimpleTrack", "NotSoSimpleTrack", "SlightlyHarder", "VisualTrack/track_circuit2_metadata.json"

# Window settings
WINDOW_WIDTH = 160
WINDOW_HEIGHT = 224
TARGET_FPS = 60
# Note: Autopilot will automatically resize images to 1024x1280 for inference

# Autopilot settings
ENABLE_AUTOPILOT_AT_START = False  # Set to True to start with autopilot active
AUTOPILOT_MODEL_PATH = "models/cnn2b.pth"  # Path to trained CNN model
AUTOPILOT_INFERENCE_HZ = 5  # Inference frequency (Hz). Lower = better FPS, higher = more responsive
# Recommended: 5 Hz = smooth FPS, 10 Hz = balanced, 15+ Hz = needs good GPU

# Recording settings
ENABLE_RECORDING_AT_START = False  # Set to True to start recording immediately
RECORD_IMAGES = True  # Set to False to record only sensor data (no images)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Rally Robot Pilot - Integrated Controls")
    print("="*70)
    print(f"Track: {TRACK_NAME}")
    print(f"Window: {WINDOW_WIDTH}x{WINDOW_HEIGHT} @ {TARGET_FPS} FPS")
    print(f"Autopilot at start: {ENABLE_AUTOPILOT_AT_START}")
    print(f"Autopilot inference: {AUTOPILOT_INFERENCE_HZ} Hz")
    print(f"Recording at start: {ENABLE_RECORDING_AT_START}")
    print("="*70)
    print("\nControls:")
    print("  WASD / Arrows : Manual driving")
    print("  R             : Start recording")
    print("  E             : Stop recording & save")
    print("  P             : Start autopilot")
    print("  O             : Stop autopilot")
    print("  G             : Reset car")
    print("  V             : Toggle raycasts")
    print("  ESC           : Exit")
    print("="*70)
    print("\nStarting game...\n")
    
    # Prepare and run the game
    app, car, controller = prepare_game_app(
        track_name=TRACK_NAME,
        window_size=(WINDOW_WIDTH, WINDOW_HEIGHT),
        target_fps=TARGET_FPS,
        enable_autopilot=ENABLE_AUTOPILOT_AT_START,
        enable_recording=ENABLE_RECORDING_AT_START,
        autopilot_model_path=AUTOPILOT_MODEL_PATH,
        record_images=RECORD_IMAGES,
        autopilot_inference_hz=AUTOPILOT_INFERENCE_HZ
    )
    
    app.run()
