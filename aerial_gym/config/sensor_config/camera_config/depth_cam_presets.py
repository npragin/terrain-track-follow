from .base_depth_camera_config import BaseDepthCameraConfig

# ranges of possible rotations for the camera relative to the UAV’s body frame
# Roll: rotation around X-axis (camera “tilts left/right”)
# Pitch: rotation around Y-axis (camera “looks up/down”)
# Yaw: rotation around Z-axis (camera “turns left/right”)
# FD15 = Forward pitched downward 15 degrees
class DepthCamFD15Fixed(BaseDepthCameraConfig):
    # --- Intrinsics / rendering ---
    width  = 480
    height = 270
    horizontal_fov_deg_min = -45.0
    horizontal_fov_deg_max =  45.0
    # horizontal_fov_deg_min = -80.0
    # horizontal_fov_deg_max =  80.0
    near_plane = 0.05
    far_plane  = 30.0

    # --- Warp placement relative to robot body ---
    # Fixed mount: make min == max and keep randomize_placement False.
    randomize_placement = False
    min_translation = [0.10, 0.00, 0.03]      # x, y, z in meters
    max_translation = [0.10, 0.00, 0.03]
    min_euler_rotation_deg = [0.0, -15.0, 0.0]  # roll, pitch, yaw in degrees
    max_euler_rotation_deg = [0.0, -15.0, 0.0]

class DepthCamFDToleranceDR(DepthCamForwardDown15Fixed):
    # manufacturing-tolerance (Domain Randomization ranges in meters)
    randomize_placement = True
    min_translation = [0.09, -0.01, 0.02]
    max_translation = [0.11,  0.01, 0.04]
    min_euler_rotation_deg = [-2.0, -18.0, -3.0]   # roll, pitch, yaw
    max_euler_rotation_deg = [ 2.0, -12.0,  3.0]

class DepthCamFDWideDR(DepthCamForwardDown15Fixed):
    # wider (Domain Randomization ranges in meters)
    randomize_placement = True
    min_translation = [0.08, -0.02, 0.02]
    max_translation = [0.12,  0.02, 0.05]
    min_euler_rotation_deg = [-3.0, -20.0, -5.0]
    max_euler_rotation_deg = [ 3.0, -10.0,  5.0]

class DepthCamNoseLevel(BaseDepthCameraConfig):
    width  = 640
    height = 360
    horizontal_fov_deg_min = -50.0
    horizontal_fov_deg_max =  50.0
    near_plane = 0.05
    far_plane  = 40.0

    randomize_placement = False
    min_translation = [0.15, 0.00, 0.02]
    max_translation = [0.15, 0.00, 0.02]
    min_euler_rotation_deg = [0.0, 0.0, 0.0]
    max_euler_rotation_deg = [0.0, 0.0, 0.0]
