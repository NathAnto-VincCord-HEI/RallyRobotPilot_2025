from rallyrobopilot import Car, Track, SunLight, MultiRaySensor
from ursina import *

def prepare_game_app(track_name="SimpleTrack", 
                     window_size=(256, 256), 
                     target_fps=15,
                     enable_autopilot=False,
                     enable_recording=False,
                     autopilot_model_path="car_cnn.pth",
                     record_images=True,
                     autopilot_inference_hz=10):
    """
    Prepare and configure the game application with integrated controls.
    
    Args:
        track_name: Name of the track to load (default: "SimpleTrack")
        window_size: Tuple of (width, height) for window size (default: (256, 256))
        target_fps: Target frames per second (default: 15)
        enable_autopilot: Enable autopilot at startup (default: False)
        enable_recording: Enable recording at startup (default: False)
        autopilot_model_path: Path to the autopilot model file (default: "car_cnn.pth")
        record_images: Whether to record images in snapshots (default: True)
        autopilot_inference_hz: Autopilot inference frequency in Hz (default: 10)
    
    Returns:
        tuple: (app, car, controller) - Ursina app, Car instance, and DirectController
    """
    from ursina import window, Ursina
    application.target_fps = target_fps
    # Create Window
    window.vsync = False # Set to false to uncap FPS limit of 60
    app = Ursina(size=window_size)
    print("Asset folder")
    print(application.asset_folder)

    # Set assets folder. Here assets are one folder up from current location.
    application.asset_folder = application.asset_folder.parent
    print("Asset folder")
    print(application.asset_folder)

    window.title = "Rally"
    window.borderless = False
    window.show_ursina_splash = False
    window.cog_button.disable()
    window.fps_counter.enable()
    window.exit_button.disable()
    
    #   Global models & textures
    #                   car model       particle model    raycast model
    global_models = [ "assets/cars/sports-car.obj", "assets/particles/particles.obj",  "assets/utils/line.obj"]
    #                Car texture             Particle Textures
    global_texs = [ "assets/cars/garage/sports-car/sports-red.png", "sports-blue.png", "sports-green.png", "sports-orange.png", "sports-white.png", "particle_forest_track.png", "red.png"]
    
    # load asset
    track = Track(track_name)
    print("loading assets after track creation")
    track.load_assets(global_models, global_texs)
    
    # Car
    car = Car()
    car.sports_car()
    # Tracks
    car.set_track(track)
    
    
    car.multiray_sensor = MultiRaySensor(car, 15, 90)
    car.multiray_sensor.enable()
    
    # Lighting + shadows
    sun = SunLight(direction = (-0.7, -0.9, 0.5), resolution = 3072, car = car)
    ambient = AmbientLight(color = Vec4(0.5, 0.55, 0.66, 0) * 0.75)
    
    render.setShaderAuto()
    
    # Sky
    Sky(texture = "sky")
    
    car.visible = True
    
    mouse.locked = False
    mouse.visible = True
    
    car.enable()
    
    car.camera_angle = "top"
    car.change_camera = True
    car.camera_follow = True
    
    track.activate()
    track.played = True
    
    # Import and create DirectController
    from .direct_controller import DirectController
    controller = DirectController(
        car=car,
        enable_autopilot=enable_autopilot,
        enable_recording=enable_recording,
        autopilot_model_path=autopilot_model_path,
        record_images=record_images,
        autopilot_inference_hz=autopilot_inference_hz
    )
   
    return app, car, controller
