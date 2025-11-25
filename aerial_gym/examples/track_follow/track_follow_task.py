from aerial_gym.config.asset_config.env_object_config import TARGET_SEMANTIC_ID, target_asset_params
from aerial_gym.config.robot_config.lmf2_config import LMF2Cfg
from aerial_gym.task.navigation_task.navigation_task import NavigationTask, exponential_reward_function
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)

import math
import os
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from aerial_gym.utils.math import (
    get_euler_xyz_tensor,
    quat_rotate_inverse,
    torch_interpolate_ratio,
    torch_rand_float_tensor,
)


def compute_env_size(t, s, grid, max_env_size, scale=1.0):
    """
    Compute environment size based on episode length, drone max speed, and grid cell size.

    Args:
        t: episode length (time in seconds)
        s: drone max speed (m/s)
        grid: exploration_grid_cell_size (m)
        max_env_size: maximum environment size (m) - used to determine upper bound for iteration
        scale: scaling factor to apply to the computed environment width

    Returns:
        w: environment width (m) scaled by scale parameter

    """
    max_c = max(1, math.ceil(max_env_size / grid))

    for c in range(1, max_c + 1):
        w = t * s / (c + 1) + grid
        if c == math.ceil(w / grid):
            return w * scale
    logger.warning(
        f"No valid solution found for t={t}, s={s}, grid={grid}, max_env_size={max_env_size}. Returning max_env_size."
    )
    return max_env_size * scale


class TrackFollowTask(NavigationTask):
    def __init__(self, task_config, **kwargs):
        task_config.action_space_dim = 3
        task_config.curriculum.min_level = 36
        super().__init__(task_config=task_config, **kwargs)

        # Create directory for saving segmentation visualizations
        # Save in the track_follow examples directory for easy access
        script_dir = Path(__file__).parent.absolute()
        self.vis_save_dir = script_dir / "segmentation_visualizations"
        os.makedirs(self.vis_save_dir, exist_ok=True)
        logger.info(f"Segmentation visualizations will be saved to: {self.vis_save_dir}/")

        # Get target asset index from asset manager
        self.target_asset_idx = None
        if hasattr(self.sim_env, "asset_manager") and self.sim_env.asset_manager is not None:
            self.target_asset_idx = self.sim_env.asset_manager.target_asset_idx
            if self.target_asset_idx is not None:
                logger.info(f"TrackFollowTask: Found target asset at index {self.target_asset_idx}")
            else:
                logger.warning(
                    "TrackFollowTask: Target asset not found in asset manager. Rewards will use random target_position."
                )
        else:
            logger.warning("TrackFollowTask: Asset manager not available. Rewards will use random target_position.")

        # Cache for target bounding box to avoid duplicate extraction per step
        self.cached_target_bbox = torch.zeros((self.sim_env.num_envs, 4), device=self.device, requires_grad=False)

        # Cache for target visibility flag (computed once per step)
        self.cached_has_valid_bbox = torch.zeros(
            (self.sim_env.num_envs,), device=self.device, dtype=torch.bool, requires_grad=False
        )

        # Cache for previous actions (needed because robot_prev_actions gets updated during sim_env.step())
        self.cached_prev_actions = torch.zeros((self.sim_env.num_envs, 4), device=self.device, requires_grad=False)

        # Track previous episode state to detect episode endings for wandb logging
        self.prev_episode_done = torch.zeros((self.sim_env.num_envs,), device=self.device, dtype=torch.bool)

        # Store segmentation frames for env 0 to create episode GIF
        self.segmentation_frames_env0 = []

        # Grace period counter: tracks remaining frames of "target visible" rewards after losing visual contact
        # Resets to grace_period_frames when target becomes visible again
        # Initialized to zero (no grace period until target is first seen)
        self.grace_period_counter = torch.zeros(
            (self.sim_env.num_envs,), device=self.device, dtype=torch.int32, requires_grad=False
        )

        # Get camera parameters from robot configuration
        robot_cfg = self.sim_env.robot_manager.robot.cfg
        self.camera_max_range = robot_cfg.sensor_config.camera_config.max_range
        self.camera_horizontal_fov_deg = robot_cfg.sensor_config.camera_config.horizontal_fov_deg
        camera_width = robot_cfg.sensor_config.camera_config.width
        camera_height = robot_cfg.sensor_config.camera_config.height
        aspect_ratio = camera_width / camera_height

        # Get camera pitch based on sensor type
        if self.task_config.use_warp:
            min_euler_deg = robot_cfg.sensor_config.camera_config.min_euler_rotation_deg
            max_euler_deg = robot_cfg.sensor_config.camera_config.max_euler_rotation_deg
            camera_pitch_deg = (min_euler_deg[1] + max_euler_deg[1]) / 2.0
        else:
            camera_pitch_deg = robot_cfg.sensor_config.camera_config.nominal_orientation_euler_deg[1]
        camera_pitch_rad = np.deg2rad(camera_pitch_deg)

        desired_altitude = self.task_config.reward_parameters["min_desired_altitude_ratio"] * self.camera_max_range
        horizontal_fov_rad = np.deg2rad(self.camera_horizontal_fov_deg)
        vertical_fov_rad = 2.0 * np.arctan(np.tan(horizontal_fov_rad / 2.0) / aspect_ratio)

        pitch_from_vertical_rad = np.pi / 2.0 + camera_pitch_rad
        cos_pitch = np.clip(np.cos(pitch_from_vertical_rad), np.cos(np.deg2rad(85.0)), 1.0)
        distance_along_viewing_ray = desired_altitude / cos_pitch

        # Calculate exploration grid cell size: 2 * (altitude / cos(angle_from_vertical)) * tan(HFOV/2)
        horizontal_coverage = 2.0 * distance_along_viewing_ray * np.tan(horizontal_fov_rad / 2.0)
        vertical_coverage = 2.0 * distance_along_viewing_ray * np.tan(vertical_fov_rad / 2.0)
        grid_cell_size = min(horizontal_coverage, vertical_coverage)

        env_bounds_min_all = self.obs_dict["env_bounds_min"][:, 0:2]
        env_bounds_max_all = self.obs_dict["env_bounds_max"][:, 0:2]
        self.exploration_env_bounds_min = env_bounds_min_all.to(self.device)

        env_bounds_min = env_bounds_min_all[0].cpu().numpy()
        env_bounds_max = env_bounds_max_all[0].cpu().numpy()
        env_size_x = float(env_bounds_max[0] - env_bounds_min[0])
        env_size_y = float(env_bounds_max[1] - env_bounds_min[1])
        self.exploration_grid_size_x = int(np.ceil(env_size_x / grid_cell_size.item()))
        self.exploration_grid_size_y = int(np.ceil(env_size_y / grid_cell_size.item()))
        self.exploration_total_cells = self.exploration_grid_size_x * self.exploration_grid_size_y
        self.exploration_grid_cell_size = grid_cell_size

        self.exploration_visited_grid = torch.zeros(
            (self.sim_env.num_envs, self.exploration_grid_size_x, self.exploration_grid_size_y),
            device=self.device,
            dtype=torch.bool,
            requires_grad=False,
        )

        vertical_fov_deg = np.rad2deg(vertical_fov_rad)
        fov_used = "HFOV" if horizontal_coverage < vertical_coverage else "VFOV"
        logger.info(
            f"Exploration grid: {self.exploration_grid_size_x}x{self.exploration_grid_size_y} = "
            f"{self.exploration_total_cells} cells, cell_size={grid_cell_size:.2f}m "
            f"(HFOV={self.camera_horizontal_fov_deg:.1f}°={horizontal_coverage:.2f}m, "
            f"VFOV={vertical_fov_deg:.1f}°={vertical_coverage:.2f}m, using {fov_used}, "
            f"alt={desired_altitude:.2f}m, pitch={camera_pitch_deg:.1f}°)"
        )

        # Add privileged observations to observation space and task_obs
        # Privileged observations include vec_to_target (3D) and dist_to_target (1D) for the critic
        if self.task_config.privileged_observation_space_dim > 0:
            # Add privileged_obs to observation space
            from gym.spaces import Box, Dict

            # Update observation space to include privileged observations
            obs_dict = dict(self.observation_space.spaces)
            obs_dict["priviliged_obs"] = Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.task_config.privileged_observation_space_dim,),
                dtype=np.float32,
            )
            self.observation_space = Dict(obs_dict)
            # Add privileged_obs to task_obs dictionary
            self.task_obs["priviliged_obs"] = torch.zeros(
                (self.sim_env.num_envs, self.task_config.privileged_observation_space_dim),
                device=self.device,
                requires_grad=False,
            )
            logger.info(
                f"TrackFollowTask: Added privileged observations with dimension {self.task_config.privileged_observation_space_dim}"
            )

        # Curriculum-based bounds checking setup
        self.base_env_bounds_min = self.obs_dict["env_bounds_min"].clone()  # [num_envs, 3]
        self.base_env_bounds_max = self.obs_dict["env_bounds_max"].clone()  # [num_envs, 3]

        bounds_size = self.base_env_bounds_max[0] - self.base_env_bounds_min[0]
        env_size_x = float(bounds_size[0].item() if hasattr(bounds_size[0], "item") else bounds_size[0])
        env_size_y = float(bounds_size[1].item() if hasattr(bounds_size[1], "item") else bounds_size[1])
        self.max_env_size = max(env_size_x, env_size_y)

        self.enable_bounds_termination = self.task_config.curriculum.enable_bounds_termination

        self.dt = self.obs_dict.get("dt", self.sim_env.sim_config.sim.dt)
        if isinstance(self.dt, torch.Tensor):
            self.dt = self.dt.item() if self.dt.numel() == 1 else self.dt[0].item()

        self.drone_max_speed = self.task_config.max_speed
        self.base_episode_len_steps = self.task_config.episode_len_steps
        self.episode_len_fraction_min = self.task_config.curriculum.episode_len_fraction_min
        self.env_size_scale = self.task_config.curriculum.env_size_scale

        # Cache for current bounds to avoid recomputing on every call
        self._cached_bounds_min = None
        self._cached_bounds_max = None
        self._cached_curriculum_progress = None

        logger.info(
            f"TrackFollowTask: Curriculum bounds enabled={self.enable_bounds_termination}, "
            f"episode_len_fraction_min: {self.episode_len_fraction_min:.2f}, "
            f"drone_max_speed: {self.drone_max_speed:.2f} m/s, dt: {self.dt:.4f} s, "
            f"max_env_size: {self.max_env_size:.2f} m"
        )

        # Store target config values for resampling
        self.target_min_state_ratio_xyz = torch.tensor(
            target_asset_params.min_state_ratio[0:3], device=self.device, requires_grad=False
        )  # [x_ratio, y_ratio, z_ratio]
        self.target_max_state_ratio_xyz = torch.tensor(
            target_asset_params.max_state_ratio[0:3], device=self.device, requires_grad=False
        )  # [x_ratio, y_ratio, z_ratio]

        # Store LMF2 config init_state values for resampling (explicitly use LMF2Cfg instead of robot config)
        self.lmf2_min_init_state_xyz = torch.tensor(
            LMF2Cfg.init_config.min_init_state[0:3], device=self.device, requires_grad=False
        )  # [x_ratio, y_ratio, z_ratio]
        self.lmf2_max_init_state_xyz = torch.tensor(
            LMF2Cfg.init_config.max_init_state[0:3], device=self.device, requires_grad=False
        )  # [x_ratio, y_ratio, z_ratio]

    def get_current_episode_length_steps(self):
        """Compute current episode length in steps based on curriculum level."""
        current_fraction = (
            self.episode_len_fraction_min + (1.0 - self.episode_len_fraction_min) * self.curriculum_progress_fraction
        )
        return int(self.base_episode_len_steps * current_fraction)

    def get_current_bounds(self):
        """Compute current bounds based on curriculum level using episode length. Cached for performance."""
        if not self.enable_bounds_termination:
            return self.base_env_bounds_min.clone(), self.base_env_bounds_max.clone()

        if (
            self._cached_bounds_min is not None
            and self._cached_bounds_max is not None
            and self._cached_curriculum_progress == self.curriculum_progress_fraction
        ):
            return self._cached_bounds_min, self._cached_bounds_max

        current_episode_len_steps = self.get_current_episode_length_steps()
        t_seconds = current_episode_len_steps * self.dt

        grid_cell_size = float(
            self.exploration_grid_cell_size.item()
            if hasattr(self.exploration_grid_cell_size, "item")
            else self.exploration_grid_cell_size
        )
        env_width = compute_env_size(
            t_seconds, self.drone_max_speed, grid_cell_size, self.max_env_size, self.env_size_scale
        )

        bounds_center = (self.base_env_bounds_min + self.base_env_bounds_max) / 2.0
        bounds_size = self.base_env_bounds_max - self.base_env_bounds_min

        current_size = bounds_size.clone()
        current_size[:, 0] = env_width
        current_size[:, 1] = env_width

        current_bounds_min = bounds_center - current_size / 2.0
        current_bounds_max = bounds_center + current_size / 2.0

        self._cached_bounds_min = current_bounds_min
        self._cached_bounds_max = current_bounds_max
        self._cached_curriculum_progress = self.curriculum_progress_fraction

        return current_bounds_min, current_bounds_max

    def check_bounds_violation(self, position):
        """Check if position violates current curriculum bounds."""
        if not self.enable_bounds_termination:
            return torch.zeros(position.shape[0], device=self.device, dtype=torch.bool)

        current_bounds_min, current_bounds_max = self.get_current_bounds()
        below_min = (position < current_bounds_min).any(dim=1)
        above_max = (position > current_bounds_max).any(dim=1)
        return below_min | above_max

    def extract_target_bbox_from_segmentation(self, segmentation_mask):
        """
        Extract 2D bounding box of TARGET_SEMANTIC_ID from segmentation mask. Uses vectorized operations for efficiency.

        The bounding box is only registered if the number of target pixels is at least
        min_pixels_on_target (configured in task_config). This helps filter out noise
        and ensures only substantial target detections are used.

        Args:
            segmentation_mask: Tensor of shape [num_envs, num_sensors, height, width] with dtype int32

        Returns:
            bbox: Tensor of shape [num_envs, 4] with [x_min, y_min, x_max, y_max] normalized to [0, 1]
                  If target not found or has insufficient pixels, returns [0, 0, 0, 0]

        """
        num_envs = segmentation_mask.shape[0]
        num_cameras = segmentation_mask.shape[1] if segmentation_mask.ndim == 4 else 0
        bbox = torch.zeros((num_envs, 4), device=segmentation_mask.device, dtype=torch.float32)

        if num_cameras > 0:
            seg_mask = segmentation_mask[:, 0, :, :]  # [num_envs, height, width]
            height, width = seg_mask.shape[1], seg_mask.shape[2]

            # Create binary mask for target (TARGET_SEMANTIC_ID = 15)
            target_mask = seg_mask == TARGET_SEMANTIC_ID  # [num_envs, height, width]

            # Count number of target pixels per environment
            num_target_pixels = target_mask.sum(dim=(1, 2))  # [num_envs]

            # Check if target exists with sufficient pixels in each environment
            min_pixels = getattr(self.task_config, "min_pixels_on_target", 1)
            target_found = num_target_pixels >= min_pixels  # [num_envs]

            # Only compute bbox for environments where target is found with sufficient pixels
            if torch.any(target_found):
                # Find rows and columns where target is present for all environments at once
                y_has_target = torch.any(target_mask, dim=2)  # [num_envs, height] - True if any pixel in row has target
                x_has_target = torch.any(target_mask, dim=1)  # [num_envs, width] - True if any pixel in col has target

                # Create coordinate grids
                y_coords = torch.arange(height, device=segmentation_mask.device, dtype=torch.float32)  # [height]
                x_coords = torch.arange(width, device=segmentation_mask.device, dtype=torch.float32)  # [width]

                # Expand for broadcasting: [num_envs, height] and [num_envs, width]
                y_coords_expanded = y_coords.unsqueeze(0).expand(num_envs, -1)  # [num_envs, height]
                x_coords_expanded = x_coords.unsqueeze(0).expand(num_envs, -1)  # [num_envs, width]

                # Mask coordinates where target is present (use inf/-inf as sentinels)
                y_masked_min = torch.where(
                    y_has_target, y_coords_expanded, torch.full_like(y_coords_expanded, float("inf"))
                )
                y_masked_max = torch.where(
                    y_has_target, y_coords_expanded, torch.full_like(y_coords_expanded, float("-inf"))
                )
                x_masked_min = torch.where(
                    x_has_target, x_coords_expanded, torch.full_like(x_coords_expanded, float("inf"))
                )
                x_masked_max = torch.where(
                    x_has_target, x_coords_expanded, torch.full_like(x_coords_expanded, float("-inf"))
                )

                # Find min/max coordinates
                y_min = torch.min(y_masked_min, dim=1)[0]  # [num_envs]
                y_max = torch.max(y_masked_max, dim=1)[0]  # [num_envs]
                x_min = torch.min(x_masked_min, dim=1)[0]  # [num_envs]
                x_max = torch.max(x_masked_max, dim=1)[0]  # [num_envs]

                # Normalize by image dimensions (to range [0, 1]) and set to zero if target not found
                bbox[:, 0] = torch.where(target_found, x_min / width, torch.zeros_like(x_min))
                bbox[:, 1] = torch.where(target_found, y_min / height, torch.zeros_like(y_min))
                bbox[:, 2] = torch.where(target_found, x_max / width, torch.zeros_like(x_max))
                bbox[:, 3] = torch.where(target_found, y_max / height, torch.zeros_like(y_max))

        return bbox

    def create_segmentation_visualization(self, env_id=0, return_figure=False):
        """
        Create segmentation mask visualization with bounding box overlay.
        Can return the figure for wandb logging or save to disk.

        Args:
            env_id: Environment ID to visualize (default: 0)
            return_figure: If True, return the figure object instead of saving (default: False)

        Returns:
            If return_figure=True, returns the matplotlib figure. Otherwise returns None.

        """
        if "segmentation_pixels" not in self.obs_dict:
            return None

        cam_id = 0  # First camera

        # Get segmentation mask - only for the specified env_id
        seg_mask = self.obs_dict["segmentation_pixels"][env_id, cam_id].cpu().numpy()
        height, width = seg_mask.shape

        # Use cached bounding box (already computed in step() for all envs, but we only use env_id's)
        # This avoids recomputing bboxes for all environments
        bbox_normalized = self.cached_target_bbox[env_id].cpu().numpy()  # [x_min, y_min, x_max, y_max] in [0, 1]

        # Create visualization for segmentation mask
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Normalize segmentation mask for visualization (scale to 0-255)
        seg_vis = seg_mask.copy().astype(np.float32)
        if seg_vis.max() > seg_vis.min():
            seg_vis = (seg_vis - seg_vis.min()) / (seg_vis.max() - seg_vis.min()) * 255
        seg_vis = seg_vis.astype(np.uint8)

        # Create colormap for better visualization
        seg_colored = plt.cm.viridis(seg_vis / 255.0)[:, :, :3]  # Convert to RGB
        ax.imshow(seg_colored)

        # Draw bounding box if target is found (bbox not all zeros)
        if np.any(bbox_normalized != 0):
            # Convert normalized bbox to pixel coordinates
            x_min = int(bbox_normalized[0] * width)
            y_min = int(bbox_normalized[1] * height)
            x_max = int(bbox_normalized[2] * width)
            y_max = int(bbox_normalized[3] * height)

            # Draw rectangle
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min, linewidth=3, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)

            # Add text label
            ax.text(
                x_min,
                y_min - 5,
                f"Target BB: [{x_min}, {y_min}, {x_max}, {y_max}]",
                color="red",
                fontsize=10,
                weight="bold",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )
        else:
            # No target found
            ax.text(
                width // 2,
                height // 2,
                "No Target Detected",
                color="red",
                fontsize=16,
                weight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
            )

        ax.set_title(f"Segmentation Mask with Bounding Box - Step {self.num_task_steps}", fontsize=14)
        ax.axis("off")

        if return_figure:
            return fig
        else:
            # Save segmentation image
            save_path = self.vis_save_dir / f"seg_bbox_step_{self.num_task_steps:06d}.png"
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved segmentation visualization to {save_path}")
            return None

    def save_segmentation_visualization(self, save_every_n_steps=1):
        """
        Save segmentation mask visualization with bounding box overlay.
        Saves images from the first environment (env_id=0) every N steps.

        Args:
            save_every_n_steps: Save visualization every N steps (default: 200)

        """
        if "segmentation_pixels" not in self.obs_dict:
            return

        # Only save every N steps to avoid too many files
        if self.num_task_steps % save_every_n_steps != 0:
            return

        self.create_segmentation_visualization(env_id=0, return_figure=False)

        # Optionally save depth image alongside
        env_id = 0  # First environment
        cam_id = 0  # First camera
        if "depth_range_pixels" in self.obs_dict:
            depth_img = self.obs_dict["depth_range_pixels"][env_id, cam_id].cpu().numpy()

            # Create depth visualization
            fig_depth, ax_depth = plt.subplots(1, 1, figsize=(12, 8))

            # Process depth for visualization
            depth_vis = depth_img.copy()

            # Check if depth is normalized (values in [0, 1] or [-1, 1] range)
            # If normalized, we need to handle out-of-range markers differently
            # Out-of-range values are typically set to max_range (if normalized) or -1.0 (if not normalized)
            max_range = 10.0  # From camera config
            min_range = 0.2

            # Identify valid depth pixels (not out-of-range markers)
            # Out-of-range markers: negative values or values >= max_range (if normalized) or == -1.0 (if not normalized)
            if depth_vis.max() <= 1.0 and depth_vis.min() >= -1.0:
                # Likely normalized - out-of-range is max_range (1.0) or -max_range (-1.0)
                valid_mask = (depth_vis > -0.9) & (depth_vis < 0.9) & (depth_vis > 0)
            else:
                # Not normalized - out-of-range is -1.0
                valid_mask = depth_vis > 0

            if np.any(valid_mask):
                valid_depths = depth_vis[valid_mask]

                # Use percentile-based normalization to better show terrain variations
                # This helps when terrain has small variations compared to objects
                p1 = np.percentile(valid_depths, 1)
                p99 = np.percentile(valid_depths, 99)

                # Normalize depth for visualization
                depth_normalized = np.zeros_like(depth_vis)
                depth_normalized[valid_mask] = np.clip((depth_vis[valid_mask] - p1) / (p99 - p1 + 1e-6), 0, 1)

                # Use a colormap that shows depth well (inferno or turbo for better color variation)
                depth_colored = plt.cm.turbo(depth_normalized)[:, :, :3]
                # Set invalid pixels to black
                depth_colored[~valid_mask] = [0, 0, 0]

                # Get actual depth range for display
                depth_min = valid_depths.min()
                depth_max = valid_depths.max()
                depth_mean = valid_depths.mean()

                # If normalized, convert back to meters for display
                if depth_max <= 1.0:
                    depth_min_m = depth_min * max_range
                    depth_max_m = depth_max * max_range
                    depth_mean_m = depth_mean * max_range
                else:
                    depth_min_m = depth_min
                    depth_max_m = depth_max
                    depth_mean_m = depth_mean
            else:
                depth_colored = np.zeros((depth_vis.shape[0], depth_vis.shape[1], 3))
                depth_min_m = 0
                depth_max_m = 0
                depth_mean_m = 0

            # Display the depth image
            im = ax_depth.imshow(depth_colored)
            ax_depth.set_title(
                f"Depth Image - Step {self.num_task_steps}\n"
                f"Range: [{depth_min_m:.2f}, {depth_max_m:.2f}] m, Mean: {depth_mean_m:.2f} m",
                fontsize=14,
            )
            ax_depth.axis("off")

            # Add colorbar to show depth scale
            # Use ScalarMappable to create colorbar from the actual depth values
            if np.any(valid_mask):
                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize

                # Create a ScalarMappable with the actual depth range
                depth_min_val = depth_vis[valid_mask].min()
                depth_max_val = depth_vis[valid_mask].max()
                sm = ScalarMappable(cmap="turbo", norm=Normalize(vmin=depth_min_val, vmax=depth_max_val))
                sm.set_array([])  # Empty array, we just need the colormap

                # Create colorbar with proper positioning
                cbar = plt.colorbar(sm, ax=ax_depth, fraction=0.046, pad=0.04, aspect=40)
                # Label depends on whether normalized
                if depth_vis.max() <= 1.0:
                    cbar.set_label("Depth (normalized [0,1])", rotation=270, labelpad=20)
                else:
                    cbar.set_label("Depth (m)", rotation=270, labelpad=20)

            # Save depth image
            depth_save_path = self.vis_save_dir / f"depth_step_{self.num_task_steps:06d}.png"
            plt.savefig(str(depth_save_path), dpi=150, bbox_inches="tight")
            plt.close(fig_depth)

            logger.info(
                f"Saved depth visualization to {depth_save_path} (range: [{depth_min_m:.2f}, {depth_max_m:.2f}] m)"
            )

    def _ensure_env_ids_tensor(self, env_ids):
        """Convert env_ids to tensor on correct device."""
        if not isinstance(env_ids, torch.Tensor):
            return torch.tensor(env_ids, device=self.device, dtype=torch.long)
        return env_ids.to(self.device)

    def _resample_positions_within_bounds(self, env_ids, min_ratio_xyz, max_ratio_xyz):
        """
        Resample positions within curriculum bounds for specified environments.

        Args:
            env_ids: Environment IDs to resample positions for
            min_ratio_xyz: Minimum x, y, z ratios (shape: [num_envs, 3])
            max_ratio_xyz: Maximum x, y, z ratios (shape: [num_envs, 3])

        """
        env_ids_tensor = self._ensure_env_ids_tensor(env_ids)
        current_bounds_min, current_bounds_max = self.get_current_bounds()

        random_ratios = torch_rand_float_tensor(
            min_ratio_xyz,
            max_ratio_xyz,
        )

        return torch_interpolate_ratio(
            current_bounds_min[env_ids_tensor],
            current_bounds_max[env_ids_tensor],
            random_ratios,
        )

    def clamp_target_to_curriculum_bounds(self, env_ids=None):
        """Clamp target position to curriculum bounds."""
        if not self.enable_bounds_termination:
            return

        current_bounds_min, current_bounds_max = self.get_current_bounds()

        if env_ids is None:
            self.target_position[:] = torch.clamp(self.target_position, current_bounds_min, current_bounds_max)
        else:
            env_ids_tensor = self._ensure_env_ids_tensor(env_ids)
            self.target_position[env_ids_tensor] = torch.clamp(
                self.target_position[env_ids_tensor],
                current_bounds_min[env_ids_tensor],
                current_bounds_max[env_ids_tensor],
            )

    def resample_target_within_curriculum_bounds(self, env_ids):
        """Resample target position within curriculum bounds."""
        if not self.enable_bounds_termination:
            return

        env_ids_tensor = self._ensure_env_ids_tensor(env_ids)
        # Expand target config ratios to match number of environments
        target_min_ratio_xyz = self.target_min_state_ratio_xyz.unsqueeze(0).expand(len(env_ids_tensor), -1)
        target_max_ratio_xyz = self.target_max_state_ratio_xyz.unsqueeze(0).expand(len(env_ids_tensor), -1)
        new_positions = self._resample_positions_within_bounds(
            env_ids_tensor, target_min_ratio_xyz, target_max_ratio_xyz
        )
        self.target_position[env_ids_tensor] = new_positions

        if self.target_asset_idx is not None and hasattr(self.sim_env, "asset_manager"):
            asset_manager = self.sim_env.asset_manager
            if asset_manager is not None and hasattr(asset_manager, "env_asset_state_tensor"):
                asset_manager.env_asset_state_tensor[env_ids_tensor, self.target_asset_idx, 0:3] = new_positions

    def update_target_position_from_asset(self):
        """Update target_position from actual target object position in the environment."""
        if self.target_asset_idx is not None and hasattr(self.sim_env, "asset_manager"):
            asset_manager = self.sim_env.asset_manager
            if asset_manager is not None and hasattr(asset_manager, "env_asset_state_tensor"):
                self.target_position[:] = asset_manager.env_asset_state_tensor[:, self.target_asset_idx, 0:3]
            else:
                logger.warning("TrackFollowTask: Cannot access env_asset_state_tensor. Using previous target_position.")

    def compute_target_visibility_reward(self, crashes):
        """
        Compute reward for keeping the target visible in the camera frame.

        Args:
            crashes: Tensor indicating which environments have crashed

        Returns:
            visibility_reward: Tensor of rewards for target visibility

        """
        # Target is considered "visible" if actually visible OR still in grace period
        # Uses cached has_valid_bbox computed once per step
        target_visible_or_grace = self.cached_has_valid_bbox | (self.grace_period_counter > 0)

        # Reward for visible targets (only for non-terminated environments)
        visibility_reward = torch.where(
            target_visible_or_grace & ~crashes,  # Target visible (or grace period) AND not crashed
            self.task_config.reward_parameters["target_visibility_reward"],
            torch.zeros((self.sim_env.num_envs,), device=self.device),
        )

        return visibility_reward

    def compute_bbox_size_reward(self, crashes):
        """
        Encourage maintaining optimal distance to target using bounding-box size.
        Reward increases exponentially with bbox area

        Args:
        crashes: Tensor indicating which environments have crashed.

        Returns:
            bbox_reward: Tensor of per-env rewards based on bounding-box size.

        """
        # Only valid where target is visible or in grace
        target_visible_or_grace = self.cached_has_valid_bbox | (self.grace_period_counter > 0)

        # Extract normalized bbox area
        x_min = self.cached_target_bbox[:, 0]
        y_min = self.cached_target_bbox[:, 1]
        x_max = self.cached_target_bbox[:, 2]
        y_max = self.cached_target_bbox[:, 3]

        width = torch.clamp(x_max - x_min, min=0.0)
        height = torch.clamp(y_max - y_min, min=0.0)
        bbox_area = width * height  # already normalized to [0,1]

        params = self.task_config.reward_parameters
        w_bbox = params["bbox_size_reward_magnitude"]
        max_area = params["bbox_max_area_ratio"]
        rate = params["bbox_exponential_rate"]

        clamped_area = torch.clamp(bbox_area, max=max_area)

        normalized_area = clamped_area / (max_area + 1e-6)  # Normalize to [0, 1]
        bbox_reward_raw = 1.0 - torch.exp(-rate * normalized_area)

        bbox_reward = torch.where(
            target_visible_or_grace & (~crashes),
            w_bbox * bbox_reward_raw,
            torch.zeros_like(bbox_reward_raw),
        )

        return bbox_reward

    def compute_altitude_reward(self, obs_dict, crashes):
        """
        Compute reward for maintaining optimal altitude above terrain.

        When target is NOT visible:
            - If altitude is at or above min_desired_altitude_ratio, gives maximum reward
            - Otherwise, uses exponential reward that decays with distance below min desired altitude

        When target IS visible:
            - Maxes out the reward so altitude doesn't interfere with target tracking

        Penalty:
            - If altitude is above camera_max_range and target is not visible (or in grace period),
              applies a constant penalty

        Args:
            obs_dict: Dictionary containing robot observations
            crashes: Tensor indicating which environments have crashed

        Returns:
            altitude_reward: Tensor of rewards for altitude maintenance (penalty is subtracted)

        """
        terrain_gen = self.sim_env.shared_terrain_generator
        robot_positions = obs_dict["robot_position"]  # [num_envs, 3]

        # Get cached heightmap
        heightmap = terrain_gen.generate_heightmap(use_cache=True)

        # Sample terrain height for each drone's XY position
        terrain_heights = torch.zeros(self.sim_env.num_envs, device=self.device)
        for env_id in range(self.sim_env.num_envs):
            x = robot_positions[env_id, 0].item()
            y = robot_positions[env_id, 1].item()
            terrain_height = terrain_gen.sample_height(x, y, heightmap)
            terrain_heights[env_id] = terrain_height

        # Calculate altitude above terrain
        terrain_offset = terrain_gen.amplitude / 2.0
        altitude_above_terrain = robot_positions[:, 2] - (terrain_heights + terrain_offset)

        altitude_ratio = altitude_above_terrain / self.camera_max_range
        min_desired_altitude_ratio = self.task_config.reward_parameters["min_desired_altitude_ratio"]

        at_or_above_min = altitude_ratio >= min_desired_altitude_ratio
        above_range = altitude_above_terrain > self.camera_max_range
        altitude_above_range_penalty = self.task_config.reward_parameters["altitude_above_range_penalty"]

        # Compute exponential reward based on altitude error (only applies when below min)
        altitude_reward_value = exponential_reward_function(
            self.task_config.reward_parameters["altitude_reward_magnitude"],
            self.task_config.reward_parameters["altitude_reward_exponent"],
            min_desired_altitude_ratio - altitude_ratio,
        )

        # Target is considered "visible" if actually visible OR still in grace period
        # Uses cached has_valid_bbox computed once per step
        target_visible_or_grace = self.cached_has_valid_bbox | (self.grace_period_counter > 0)

        # Determine reward: max if at/above min OR target visible, otherwise use exponential
        # Apply penalty if above range and target not visible (or in grace period)
        # Zero out reward when crashed (no reward for crashed environments)
        altitude_reward = torch.where(
            ~crashes,  # Not crashed
            torch.where(
                target_visible_or_grace,  # Target visible or in grace period
                self.task_config.reward_parameters["altitude_reward_magnitude"],  # Max reward
                torch.where(
                    at_or_above_min,  # At or above minimum desired altitude
                    torch.where(
                        above_range,  # Above camera range
                        altitude_above_range_penalty,
                        self.task_config.reward_parameters["altitude_reward_magnitude"],
                    ),
                    altitude_reward_value,
                ),
            ),
            torch.zeros((self.sim_env.num_envs,), device=self.device),  # Zero reward if crashed
        )

        return altitude_reward

    def compute_exploration_reward(self, obs_dict, crashes):
        """
        Compute reward for exploring the entire region when target not visible.

        Uses grid-based coverage: divides environment into cells based on camera FOV at target altitude.
        Rewards visiting unvisited cells. Maxes out when target visible or in grace period.

        Args:
            obs_dict: Dictionary containing robot observations
            crashes: Tensor indicating which environments have crashed

        Returns:
            exploration_reward: Tensor of rewards for exploration

        """
        robot_positions = obs_dict["robot_position"]  # [num_envs, 3]

        pos_relative = robot_positions[:, 0:2] - self.exploration_env_bounds_min
        grid_x = torch.clamp(
            (pos_relative[:, 0] / self.exploration_grid_cell_size).long(),
            0,
            self.exploration_grid_size_x - 1,
        )
        grid_y = torch.clamp(
            (pos_relative[:, 1] / self.exploration_grid_cell_size).long(),
            0,
            self.exploration_grid_size_y - 1,
        )

        # Mark current cells as visited (vectorized for all environments)
        env_indices = torch.arange(self.sim_env.num_envs, device=self.device)
        self.exploration_visited_grid[env_indices, grid_x, grid_y] = True

        # Reward = magnitude * (num_visited_cells / total_cells) = magnitude * coverage_fraction
        num_visited_cells = self.exploration_visited_grid.sum(dim=(1, 2))  # [num_envs]
        coverage_fraction = num_visited_cells.float() / self.exploration_total_cells  # [num_envs]
        exploration_reward_value = (
            self.task_config.reward_parameters["exploration_reward_magnitude"] * coverage_fraction
        )

        # Max out exploration reward when target visible or in grace period (doesn't interfere with tracking)
        # Zero out reward when crashed (no reward for crashed environments)
        target_visible_or_grace = self.cached_has_valid_bbox | (self.grace_period_counter > 0)
        exploration_reward = torch.where(
            ~crashes,  # Not crashed
            torch.where(
                target_visible_or_grace,  # Target visible or in grace period
                self.task_config.reward_parameters["exploration_reward_magnitude"],  # Max reward
                exploration_reward_value,  # Reward based on unvisited cells
            ),
            torch.zeros((self.sim_env.num_envs,), device=self.device),  # Zero reward if crashed
        )

        return exploration_reward

    def compute_yaw_alignment_reward(self, obs_dict, crashes):
        """
        Compute raw yaw alignment reward (without crash handling and without multiplication factor).
        The multiplication factor is applied to all track_follow rewards together.
        """
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_orientation = obs_dict["robot_orientation"]

        # Compute yaw error: desired yaw angle to face target
        vec_to_target = target_position - robot_position
        desired_yaw = torch.atan2(vec_to_target[:, 1], vec_to_target[:, 0])
        current_yaw = ssa(get_euler_xyz_tensor(robot_orientation))[:, 2]
        yaw_error = ssa(desired_yaw - current_yaw)

        # Yaw alignment reward: exponential reward based on yaw error
        yaw_alignment_reward = exponential_reward_function(
            self.task_config.reward_parameters["yaw_alignment_reward_magnitude"],
            self.task_config.reward_parameters["yaw_alignment_reward_exponent"],
            yaw_error,
        )

        # Zero out reward when there's no bounding box or when crashed
        yaw_alignment_reward = torch.where(
            self.cached_has_valid_bbox & ~crashes,
            yaw_alignment_reward,
            torch.zeros((self.sim_env.num_envs,), device=self.device),
        )
        return yaw_alignment_reward

    def compute_track_follow_rewards(self, obs_dict, crashes):
        """
        Compute additional rewards for track follow task.
        All rewards are scaled by the curriculum multiplication factor.
        """
        visibility_reward = self.compute_target_visibility_reward(crashes)
        altitude_reward = self.compute_altitude_reward(obs_dict, crashes)
        exploration_reward = self.compute_exploration_reward(obs_dict, crashes)
        yaw_alignment_reward = self.compute_yaw_alignment_reward(obs_dict, crashes)
        bbox_size_reward = self.compute_bbox_size_reward(crashes)

        if "reward_components" not in self.infos:
            self.infos["reward_components"] = {}
        self.infos["reward_components"].update(
            {
                "visibility_reward": visibility_reward,
                "altitude_reward": altitude_reward,
                "exploration_reward": exploration_reward,
                "yaw_alignment_reward": yaw_alignment_reward,
                "bbox_size_reward": bbox_size_reward,
            }
        )

        # Apply curriculum multiplication factor to all track_follow rewards
        MULTIPLICATION_FACTOR_REWARD = 1.0 + (2.0) * self.curriculum_progress_fraction
        total_track_follow_rewards = (
            visibility_reward + altitude_reward + exploration_reward + yaw_alignment_reward + bbox_size_reward
        ) * MULTIPLICATION_FACTOR_REWARD

        return total_track_follow_rewards

    def compute_rewards_and_crashes(self, obs_dict):
        """Override parent to add track-follow rewards and bounds violation checks."""
        # Temporarily restore cached previous actions for reward computation
        # because robot_prev_actions was updated during sim_env.step()
        if "robot_prev_actions" in obs_dict:
            original_prev_actions = obs_dict["robot_prev_actions"].clone()
            obs_dict["robot_prev_actions"][:] = self.cached_prev_actions

        base_rewards, crashes = super().compute_rewards_and_crashes(obs_dict)

        # Restore the updated prev_actions after reward computation
        if "robot_prev_actions" in obs_dict:
            obs_dict["robot_prev_actions"][:] = original_prev_actions

        # Remove disabled reward components from logs
        if "reward_components" in self.infos:
            disabled_rewards = [
                "pos_reward",
                "very_close_to_goal_reward",
                "getting_closer_reward",
                "distance_from_goal_reward",
                "absolute_action_penalty",
            ]
            for reward_name in disabled_rewards:
                self.infos["reward_components"].pop(reward_name, None)

        if self.enable_bounds_termination:
            out_of_bounds = self.check_bounds_violation(obs_dict["robot_position"])
            crashes[:] = torch.where(out_of_bounds, torch.ones_like(crashes), crashes)

        track_follow_rewards = self.compute_track_follow_rewards(obs_dict, crashes)
        return base_rewards + track_follow_rewards, crashes

    def reset_idx(self, env_ids):
        """Override reset_idx to ensure robots and targets spawn within curriculum bounds."""
        super().reset_idx(env_ids)
        self.update_target_position_from_asset()

        if self.enable_bounds_termination:
            env_ids_tensor = self._ensure_env_ids_tensor(env_ids)

            # Check robot in bounds
            out_of_bounds_full = self.check_bounds_violation(self.obs_dict["robot_position"])
            out_of_bounds = out_of_bounds_full[env_ids_tensor]

            if torch.any(out_of_bounds):
                out_of_bounds_env_ids = env_ids_tensor[out_of_bounds]
                # Use LMF2Cfg init_state values for resampling (explicitly use LMF2Cfg instead of robot config)
                num_out_of_bounds = len(out_of_bounds_env_ids)
                lmf2_min_ratio_xyz = self.lmf2_min_init_state_xyz.unsqueeze(0).expand(
                    num_out_of_bounds, -1
                )  # [x_ratio, y_ratio, z_ratio]
                lmf2_max_ratio_xyz = self.lmf2_max_init_state_xyz.unsqueeze(0).expand(
                    num_out_of_bounds, -1
                )  # [x_ratio, y_ratio, z_ratio]
                new_positions = self._resample_positions_within_bounds(
                    out_of_bounds_env_ids, lmf2_min_ratio_xyz, lmf2_max_ratio_xyz
                )

                robot_state = self.sim_env.robot_manager.robot.robot_state
                robot_state[out_of_bounds_env_ids, 0:3] = new_positions
                self.obs_dict["robot_position"][out_of_bounds_env_ids] = new_positions

                self.sim_env.IGE_env.write_to_sim(refresh_robot_state=False)

                logger.debug(
                    f"TrackFollowTask: Resampled {len(out_of_bounds_env_ids)} robot positions to be within curriculum bounds"
                )

            # Check target in bounds
            out_of_bounds_full = self.check_bounds_violation(self.target_position)
            out_of_bounds = out_of_bounds_full[env_ids_tensor]

            if torch.any(out_of_bounds):
                out_of_bounds_env_ids = env_ids_tensor[out_of_bounds]
                self.resample_target_within_curriculum_bounds(out_of_bounds_env_ids)
                logger.debug(
                    f"TrackFollowTask: Resampled {len(out_of_bounds_env_ids)} target positions to be within curriculum bounds"
                )

        self.cached_target_bbox[env_ids] = 0.0
        self.cached_has_valid_bbox[env_ids] = False
        self.grace_period_counter[env_ids] = 0
        self.exploration_visited_grid[env_ids] = False
        self.cached_prev_actions[env_ids] = 0.0

        # Reset episode done tracking for reset environments
        env_ids_tensor = self._ensure_env_ids_tensor(env_ids)
        self.prev_episode_done[env_ids_tensor] = False

        if len(env_ids) > 0 and len(env_ids) <= 4:
            env_ids_list = env_ids.cpu().numpy().tolist() if isinstance(env_ids, torch.Tensor) else env_ids
            for env_id in env_ids_list:
                logger.debug(
                    f"TrackFollowTask: Reset env {env_id}, target_position = {self.target_position[env_id].cpu().numpy()}"
                )

    def step(self, actions):
        """Override step to update target position and enforce curriculum bounds."""
        transformed_action = self.action_transformation_function(actions)
        logger.debug(f"raw_action: {actions[0]}, transformed action: {transformed_action[0]}")

        # Cache previous actions BEFORE sim_env.step() updates them
        # This is needed because robot_prev_actions gets updated during sim_env.step()
        # via robot_manager.pre_physics_step(), which happens before we compute rewards
        if "robot_prev_actions" in self.obs_dict:
            self.cached_prev_actions[:] = self.obs_dict["robot_prev_actions"]

        self.sim_env.step(actions=transformed_action)
        self.update_target_position_from_asset()

        if self.enable_bounds_termination:
            self.clamp_target_to_curriculum_bounds()
            if self.target_asset_idx is not None and hasattr(self.sim_env, "asset_manager"):
                asset_manager = self.sim_env.asset_manager
                if asset_manager is not None and hasattr(asset_manager, "env_asset_state_tensor"):
                    asset_manager.env_asset_state_tensor[:, self.target_asset_idx, 0:3] = self.target_position

        if "segmentation_pixels" in self.obs_dict:
            self.cached_target_bbox[:] = self.extract_target_bbox_from_segmentation(
                self.obs_dict["segmentation_pixels"]
            )
        else:
            self.cached_target_bbox[:] = 0.0

        # Compute target visibility flag ONCE and cache it
        self.cached_has_valid_bbox[:] = (self.cached_target_bbox[:, 2] > self.cached_target_bbox[:, 0]) & (
            self.cached_target_bbox[:, 3] > self.cached_target_bbox[:, 1]
        )

        # Update grace period counter
        grace_period_frames = self.task_config.reward_parameters["target_visibility_grace_period_frames"]
        grace_period_active_before_update = self.grace_period_counter > 0

        # Reset counter to grace_period_frames when target becomes visible
        # Decrement counter when target not visible (but don't go below 0)
        self.grace_period_counter[:] = torch.where(
            self.cached_has_valid_bbox,
            torch.full_like(self.grace_period_counter, grace_period_frames),
            torch.clamp(self.grace_period_counter - 1, min=0),
        )

        # Compute rewards (includes base rewards + visibility + altitude)
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        current_episode_len_steps = self.get_current_episode_length_steps()
        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > current_episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )

        # Success criteria: Bounding box is valid or in grace period at timeout and no failure
        has_active_or_recent_bbox = self.cached_has_valid_bbox | grace_period_active_before_update
        no_failures = self.terminations == 0
        successes = self.truncations * no_failures * has_active_or_recent_bbox
        timeouts = torch.where(self.truncations > 0, torch.logical_not(successes), torch.zeros_like(successes))
        timeouts = torch.where(
            self.terminations > 0, torch.zeros_like(timeouts), timeouts
        )  # timeouts are not counted if there is a crash

        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = self.terminations

        self.logging_sanity_check(self.infos)
        self.check_and_update_curriculum_level(self.infos["successes"], self.infos["crashes"], self.infos["timeouts"])
        # rendering happens at the post-reward calculation step since the newer measurement is required to be
        # sent to the RL algorithm as an observation and it helps if the camera image is updated then
        reset_envs = self.sim_env.post_reward_calculation_step()

        # Log segmentation visualization to wandb when env 0's episode ends
        if len(reset_envs) > 0:
            reset_envs_tensor = (
                reset_envs if isinstance(reset_envs, torch.Tensor) else torch.tensor(reset_envs, device=self.device)
            )
            # Check if env 0 is in the reset list (episode ending)
            env_0_ending = False
            if isinstance(reset_envs_tensor, torch.Tensor):
                env_0_ending = (reset_envs_tensor == 0).any().item()
            else:
                env_0_ending = 0 in reset_envs_tensor

            if env_0_ending and not self.prev_episode_done[0]:
                # Episode just ended for env 0, log visualization to wandb
                self.log_segmentation_to_wandb(env_id=0)
                # Clear frames after logging (will be reset when new episode starts)
                self.segmentation_frames_env0 = []

            # Update previous episode done state for reset environments
            self.prev_episode_done[reset_envs_tensor] = True
            self.reset_idx(reset_envs)
            # Reset tracking after reset (episode has started fresh)
            self.prev_episode_done[reset_envs_tensor] = False
        self.num_task_steps += 1
        # do stuff with the image observations here
        self.process_image_observation()

        # Collect segmentation frame for env 0 to build episode GIF (only if episode is active)
        if "segmentation_pixels" in self.obs_dict and not self.prev_episode_done[0]:
            self._collect_segmentation_frame(env_id=0)
        self.post_image_reward_addition()
        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()
        return return_tuple

    def process_obs_for_task(self):
        self.task_obs["observations"][:, 0:4] = self.cached_target_bbox
        euler_angles = ssa(get_euler_xyz_tensor(self.obs_dict["robot_vehicle_orientation"]))
        self.task_obs["observations"][:, 4:7] = euler_angles
        self.task_obs["observations"][:, 7] = self.obs_dict["robot_position"][:, 2]
        self.task_obs["observations"][:, 8:11] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 11:14] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 14:18] = self.obs_dict["robot_actions"]
        self.task_obs["observations"][:, 18:82] = self.image_latents

        # Compute privileged observations for critic: vec_to_target (3D) and dist_to_target (1D)
        if self.task_config.privileged_observation_space_dim > 0 and "priviliged_obs" in self.task_obs:
            # Compute vector to target in vehicle frame (same as NavigationTask)
            vec_to_tgt = quat_rotate_inverse(
                self.obs_dict["robot_vehicle_orientation"],
                (self.target_position - self.obs_dict["robot_position"]),
            )
            dist_to_tgt = torch.norm(vec_to_tgt, dim=-1)

            # Normalize vector to get unit direction (to keep values bounded)
            # Add small epsilon to avoid division by zero
            dist_to_tgt_safe = torch.clamp(dist_to_tgt, min=1e-6)
            unit_vec_to_tgt = vec_to_tgt / dist_to_tgt_safe.unsqueeze(1)

            # Store in privileged observations: [unit_vec_x, unit_vec_y, unit_vec_z, distance]
            self.task_obs["priviliged_obs"][:, 0:3] = unit_vec_to_tgt
            self.task_obs["priviliged_obs"][:, 3] = (
                dist_to_tgt / 5.0
            )  # Normalize distance (divide by 5.0 like in NavigationTask)

        # Save segmentation visualization (only saves every N steps internally)
        # self.save_segmentation_visualization()
        nan_mask = torch.isnan(self.task_obs["observations"])
        inf_mask = torch.isinf(self.task_obs["observations"])
        if torch.any(nan_mask) or torch.any(inf_mask):
            logger.warning(
                f"Found NaN/Inf in observations. NaN count: {torch.sum(nan_mask)}, Inf count: {torch.sum(inf_mask)}"
            )
            self.task_obs["observations"][nan_mask | inf_mask] = 0.0

        # Check for NaN/Inf in privileged observations
        if "priviliged_obs" in self.task_obs:
            priv_nan_mask = torch.isnan(self.task_obs["priviliged_obs"])
            priv_inf_mask = torch.isinf(self.task_obs["priviliged_obs"])
            if torch.any(priv_nan_mask) or torch.any(priv_inf_mask):
                logger.warning(
                    f"Found NaN/Inf in privileged observations. NaN count: {torch.sum(priv_nan_mask)}, Inf count: {torch.sum(priv_inf_mask)}"
                )
                self.task_obs["priviliged_obs"][priv_nan_mask | priv_inf_mask] = 0.0

    def _collect_segmentation_frame(self, env_id=0):
        """
        Collect a segmentation frame for the specified environment to build an episode GIF.

        Args:
            env_id: Environment ID to collect frame from (default: 0)

        """
        if "segmentation_pixels" not in self.obs_dict:
            return

        cam_id = 0  # First camera

        # Get segmentation mask
        seg_mask = self.obs_dict["segmentation_pixels"][env_id, cam_id].cpu().numpy()
        height, width = seg_mask.shape

        # Get bounding box for this frame
        bbox_normalized = self.cached_target_bbox[env_id].cpu().numpy()

        # Create custom colored image: black for background, white for terrain, colors for other elements
        seg_colored = np.zeros((height, width, 3), dtype=np.uint8)

        # Background (negative values, typically -2): Black [0, 0, 0]
        background_mask = seg_mask < 0
        seg_colored[background_mask] = [0, 0, 0]

        # Terrain (value 0): White [255, 255, 255]
        terrain_mask = seg_mask == 0
        seg_colored[terrain_mask] = [255, 255, 255]

        # Other elements (positive values): Use colormap to distinguish different objects
        other_mask = seg_mask > 0
        if np.any(other_mask):
            # Get unique positive values for consistent coloring
            unique_values = np.unique(seg_mask[other_mask])
            # Use tab20 colormap for up to 20 different objects, then cycle
            colors = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]  # Get RGB values
            colors = (colors * 255).astype(np.uint8)

            # Map each unique value to a color
            for idx, val in enumerate(unique_values):
                color_idx = idx % 20  # Cycle through colors if more than 20 objects
                mask = seg_mask == val
                seg_colored[mask] = colors[color_idx]

        # Draw bounding box if target is found
        if np.any(bbox_normalized != 0):
            # Convert normalized bbox to pixel coordinates
            x_min = int(bbox_normalized[0] * width)
            y_min = int(bbox_normalized[1] * height)
            x_max = int(bbox_normalized[2] * width)
            y_max = int(bbox_normalized[3] * height)

            # Draw rectangle on the image
            seg_colored[y_min : y_min + 3, x_min:x_max] = [255, 0, 0]  # Top edge
            seg_colored[y_max - 3 : y_max, x_min:x_max] = [255, 0, 0]  # Bottom edge
            seg_colored[y_min:y_max, x_min : x_min + 3] = [255, 0, 0]  # Left edge
            seg_colored[y_min:y_max, x_max - 3 : x_max] = [255, 0, 0]  # Right edge

        # Convert to PIL Image and append to frames list
        frame = Image.fromarray(seg_colored)
        self.segmentation_frames_env0.append(frame)

    def log_segmentation_to_wandb(self, env_id=0):
        """
        Log segmentation visualization GIF to wandb for the specified environment.
        Creates a GIF from all frames collected during the episode.

        Args:
            env_id: Environment ID to log (default: 0)

        """
        try:
            import wandb

            if wandb.run is None:
                return
        except ImportError:
            logger.debug("wandb not available, skipping segmentation logging")
            return

        # Check if we have any frames collected
        if len(self.segmentation_frames_env0) == 0:
            logger.debug(f"No segmentation frames collected for env {env_id}, skipping GIF creation")
            return

        # Get episode info for logging context
        episode_info = {
            "step": self.num_task_steps,
            "env_id": env_id,
            "num_frames": len(self.segmentation_frames_env0),
        }
        if "successes" in self.infos:
            episode_info["success"] = bool(
                self.infos["successes"][env_id].item()
                if isinstance(self.infos["successes"], torch.Tensor)
                else self.infos["successes"][env_id]
            )
        if "crashes" in self.infos:
            episode_info["crashed"] = bool(
                self.infos["crashes"][env_id].item()
                if isinstance(self.infos["crashes"], torch.Tensor)
                else self.infos["crashes"][env_id]
            )
        if "timeouts" in self.infos:
            episode_info["timeout"] = bool(
                self.infos["timeouts"][env_id].item()
                if isinstance(self.infos["timeouts"], torch.Tensor)
                else self.infos["timeouts"][env_id]
            )

        # Create GIF from collected frames and log to wandb
        try:
            import tempfile

            # Save GIF to temporary file
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_file:
                gif_path = tmp_file.name

            # Save frames as GIF
            if len(self.segmentation_frames_env0) > 0:
                self.segmentation_frames_env0[0].save(
                    gif_path,
                    format="GIF",
                    save_all=True,
                    append_images=self.segmentation_frames_env0[1:],
                    duration=100,  # 100ms per frame
                    loop=0,  # Loop forever
                )

                # Log GIF to wandb using file path
                wandb.log(
                    {
                        "segmentation_visualization": wandb.Image(gif_path),
                        **{f"episode_info/{k}": v for k, v in episode_info.items()},
                    },
                    step=self.num_task_steps,
                )
                logger.debug(
                    f"Logged segmentation visualization GIF ({len(self.segmentation_frames_env0)} frames) to wandb for env {env_id} at step {self.num_task_steps}"
                )

                # Clean up temporary file
                import contextlib

                with contextlib.suppress(Exception):
                    os.unlink(gif_path)
        except Exception as e:
            logger.warning(f"Failed to log segmentation visualization GIF to wandb: {e}")
            import traceback

            logger.debug(traceback.format_exc())


@torch.jit.script
def ssa(a: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle"""
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi
