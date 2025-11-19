from aerial_gym.config.asset_config.env_object_config import TARGET_SEMANTIC_ID
from aerial_gym.task.navigation_task.navigation_task import NavigationTask
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)

import os
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from aerial_gym.utils.math import get_euler_xyz_tensor, quat_rotate_inverse


class TrackFollowTask(NavigationTask):
    def __init__(self, task_config, **kwargs):
        task_config.action_space_dim = 3
        task_config.curriculum.min_level = 36
        logger.critical("Hardcoding number of envs to 16 if it is greater than that.")
        task_config.num_envs = 16 if task_config.num_envs > 16 else task_config.num_envs
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
                logger.warning("TrackFollowTask: Target asset not found in asset manager. Rewards will use random target_position.")
        else:
            logger.warning("TrackFollowTask: Asset manager not available. Rewards will use random target_position.")

        # Cache for target bounding box to avoid duplicate extraction per step
        self.cached_target_bbox = torch.zeros(
            (self.sim_env.num_envs, 4), device=self.device, requires_grad=False
        )

        # Cache for target visibility flag (computed once per step)
        self.cached_has_valid_bbox = torch.zeros(
            (self.sim_env.num_envs,), device=self.device, dtype=torch.bool, requires_grad=False
        )

        # Grace period counter: tracks remaining frames of "target visible" rewards after losing visual contact
        # Resets to grace_period_frames when target becomes visible again
        # Initialized to zero (no grace period until target is first seen)
        self.grace_period_counter = torch.zeros(
            (self.sim_env.num_envs,), device=self.device, dtype=torch.int32, requires_grad=False
        )

        # Get camera max_range from robot configuration
        robot_cfg = self.sim_env.robot_manager.robot.cfg
        self.camera_max_range = robot_cfg.sensor_config.camera_config.max_range

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
            logger.info(f"TrackFollowTask: Added privileged observations with dimension {self.task_config.privileged_observation_space_dim}")

    def extract_target_bbox_from_segmentation(self, segmentation_mask):
        """
        Extract 2D bounding box of TARGET_SEMANTIC_ID from segmentation mask. Uses vectorized operations for efficiency.

        Args:
            segmentation_mask: Tensor of shape [num_envs, num_sensors, height, width] with dtype int32

        Returns:
            bbox: Tensor of shape [num_envs, 4] with [x_min, y_min, x_max, y_max] normalized to [0, 1]
                  If target not found, returns [0, 0, 0, 0]

        """
        num_envs = segmentation_mask.shape[0]
        num_cameras = segmentation_mask.shape[1] if segmentation_mask.ndim == 4 else 0
        bbox = torch.zeros((num_envs, 4), device=segmentation_mask.device, dtype=torch.float32)

        if num_cameras > 0:
            seg_mask = segmentation_mask[:, 0, :, :]  # [num_envs, height, width]
            height, width = seg_mask.shape[1], seg_mask.shape[2]

            # Create binary mask for target (TARGET_SEMANTIC_ID = 15)
            target_mask = seg_mask == TARGET_SEMANTIC_ID  # [num_envs, height, width]

            # Check if target exists in each environment
            target_found = torch.any(target_mask.view(num_envs, -1), dim=1)  # [num_envs]

            # Only compute bbox for environments where target is found
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

        env_id = 0  # First environment
        cam_id = 0  # First camera

        # Get segmentation mask
        seg_mask = self.obs_dict["segmentation_pixels"][env_id, cam_id].cpu().numpy()
        height, width = seg_mask.shape

        # Extract bounding box
        target_bbox = self.extract_target_bbox_from_segmentation(self.obs_dict["segmentation_pixels"])
        bbox_normalized = target_bbox[env_id].cpu().numpy()  # [x_min, y_min, x_max, y_max] in [0, 1]

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

        # Save segmentation image
        save_path = self.vis_save_dir / f"seg_bbox_step_{self.num_task_steps:06d}.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved segmentation visualization to {save_path}")

        # Optionally save depth image alongside
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

    def update_target_position_from_asset(self):
        """
        Update self.target_position from the actual target object position in the environment.
        This ensures rewards are based on the actual target location, not a random position.
        """
        if self.target_asset_idx is not None and hasattr(self.sim_env, "asset_manager"):
            asset_manager = self.sim_env.asset_manager
            if asset_manager is not None and hasattr(asset_manager, "env_asset_state_tensor"):
                # Get actual target position from asset state tensor
                # env_asset_state_tensor shape: [num_envs, num_assets, 13]
                # Position is at indices [0:3] (x, y, z)
                actual_target_pos = asset_manager.env_asset_state_tensor[:, self.target_asset_idx, 0:3]
                self.target_position[:] = actual_target_pos
            else:
                logger.warning("TrackFollowTask: Cannot access env_asset_state_tensor. Using previous target_position.")
        # If target_asset_idx is None, target_position remains as set in reset_idx (from parent class)

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

    def compute_altitude_reward(self, obs_dict, crashes):
        """
        Compute reward for maintaining optimal altitude above terrain.
        
        When target is NOT visible:
            - Incentivizes flying at desired_altitude (80% of max camera range) above terrain
            - Uses exponential reward that decays with distance from desired altitude
        
        When target IS visible:
            - Maxes out the reward so altitude doesn't interfere with target tracking
        
        Args:
            obs_dict: Dictionary containing robot observations
            crashes: Tensor indicating which environments have crashed
            
        Returns:
            altitude_reward: Tensor of rewards for altitude maintenance

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

        # Desired altitude based on camera max range
        desired_altitude = self.task_config.reward_parameters["desired_altitude_ratio"] * self.camera_max_range

        # Calculate altitude error
        altitude_error = torch.abs(altitude_above_terrain - desired_altitude)

        # Compute exponential reward based on altitude error
        altitude_reward_value = self.task_config.reward_parameters["altitude_reward_magnitude"] * torch.exp(
            -(altitude_error * altitude_error) * self.task_config.reward_parameters["altitude_reward_exponent"]
        )

        # Target is considered "visible" if actually visible OR still in grace period
        # Uses cached has_valid_bbox computed once per step
        target_visible_or_grace = self.cached_has_valid_bbox | (self.grace_period_counter > 0)

        # Max out altitude reward when target visible (or in grace period) (doesn't interfere with tracking)
        # Zero out reward when crashed (no reward for crashed environments)
        altitude_reward = torch.where(
            ~crashes,  # Not crashed
            torch.where(
                target_visible_or_grace,  # Target visible or in grace period
                self.task_config.reward_parameters["altitude_reward_magnitude"],  # Max reward
                altitude_reward_value,  # Exponential reward based on altitude error
            ),
            torch.zeros((self.sim_env.num_envs,), device=self.device),  # Zero reward if crashed
        )

        return altitude_reward

    def compute_track_follow_rewards(self, obs_dict, crashes):
        """
        Compute additional rewards for track follow task.
        """
        visibility_reward = self.compute_target_visibility_reward(crashes)
        altitude_reward = self.compute_altitude_reward(obs_dict, crashes)
        return visibility_reward + altitude_reward

    def compute_rewards_and_crashes(self, obs_dict):
        """
        Override parent method to add target visibility and altitude rewards.
        Computes base rewards from parent, then adds track-follow specific rewards.
        """
        # Get base rewards from parent class (distance, yaw, action penalties, etc.)
        base_rewards, crashes = super().compute_rewards_and_crashes(obs_dict)

        # Compute individual reward components
        track_follow_rewards = self.compute_track_follow_rewards(obs_dict, crashes)

        # Combine all rewards
        total_rewards = base_rewards + track_follow_rewards

        return total_rewards, crashes

    def reset_idx(self, env_ids):
        """
        Override reset_idx to set target_position from actual target object position
        instead of random position based on environment bounds.
        """
        # Call parent reset_idx first to set up environment
        # But we'll override the target_position afterwards
        super().reset_idx(env_ids)

        # Update target_position from actual target object position
        self.update_target_position_from_asset()

        # Clear cached bbox and visibility flag for reset environments (will be recomputed on next step)
        self.cached_target_bbox[env_ids] = 0.0
        self.cached_has_valid_bbox[env_ids] = False

        # Reset grace period counter for reset environments (start at zero for new episode)
        self.grace_period_counter[env_ids] = 0

        # Log for debugging
        if len(env_ids) > 0 and len(env_ids) <= 4:  # Only log for small number of envs
            env_ids_list = env_ids.cpu().numpy().tolist() if isinstance(env_ids, torch.Tensor) else env_ids
            for env_id in env_ids_list:
                logger.debug(
                    f"TrackFollowTask: Reset env {env_id}, target_position = {self.target_position[env_id].cpu().numpy()}"
                )

    def step(self, actions):
        """
        Override step to update target_position from actual target object before computing rewards.
        This handles cases where the target might move during the episode.
        """
        # Transform actions (same as parent)
        transformed_action = self.action_transformation_function(actions)
        logger.debug(f"raw_action: {actions[0]}, transformed action: {transformed_action[0]}")

        # Step the environment (this may update asset positions including target)
        self.sim_env.step(actions=transformed_action)

        # Update target position from actual asset position AFTER sim_env.step()
        # but BEFORE computing rewards. This ensures rewards are based on current target location.
        self.update_target_position_from_asset()

        # Extract target bounding box ONCE and cache it for use in both rewards and observations
        # This avoids duplicate extraction (was being done in both add_target_visibility_reward
        # and process_obs_for_task)
        if "segmentation_pixels" in self.obs_dict:
            self.cached_target_bbox[:] = self.extract_target_bbox_from_segmentation(
                self.obs_dict["segmentation_pixels"]
            )
        else:
            self.cached_target_bbox[:] = 0.0

        # Compute target visibility flag ONCE and cache it
        self.cached_has_valid_bbox[:] = (
            (self.cached_target_bbox[:, 2] > self.cached_target_bbox[:, 0]) &
            (self.cached_target_bbox[:, 3] > self.cached_target_bbox[:, 1])
        )

        # Update grace period counter
        grace_period_frames = self.task_config.reward_parameters["target_visibility_grace_period_frames"]

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

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )

        # successes are the sum of the environments which are to be truncated and have reached the target within a distance threshold
        successes = self.truncations * (torch.norm(self.target_position - self.obs_dict["robot_position"], dim=1) < 1.0)
        successes = torch.where(self.terminations > 0, torch.zeros_like(successes), successes)
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
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)
        self.num_task_steps += 1
        # do stuff with the image observations here
        self.process_image_observation()
        self.post_image_reward_addition()
        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()
        return return_tuple

    def process_obs_for_task(self):
        # Use cached target bounding box (extracted once per step in step() method)
        # This replaces vector to target (3D) + distance (1D) with bounding box (4D)
        self.task_obs["observations"][:, 0:4] = self.cached_target_bbox

        euler_angles = ssa(get_euler_xyz_tensor(self.obs_dict["robot_vehicle_orientation"]))
        self.task_obs["observations"][:, 4:6] = euler_angles[:, 0:2]
        self.task_obs["observations"][:, 6] = 0.0
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 13:17] = self.obs_dict["robot_actions"]
        self.task_obs["observations"][:, 17:81] = self.image_latents

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
            self.task_obs["priviliged_obs"][:, 3] = dist_to_tgt / 5.0  # Normalize distance (divide by 5.0 like in NavigationTask)

        # Save segmentation visualization (only saves every N steps internally)
        # self.save_segmentation_visualization()
        nan_mask = torch.isnan(self.task_obs["observations"])
        inf_mask = torch.isinf(self.task_obs["observations"])
        if torch.any(nan_mask) or torch.any(inf_mask):
            logger.warning(f"Found NaN/Inf in observations. NaN count: {torch.sum(nan_mask)}, Inf count: {torch.sum(inf_mask)}")
            self.task_obs["observations"][nan_mask | inf_mask] = 0.0

        # Check for NaN/Inf in privileged observations
        if "priviliged_obs" in self.task_obs:
            priv_nan_mask = torch.isnan(self.task_obs["priviliged_obs"])
            priv_inf_mask = torch.isinf(self.task_obs["priviliged_obs"])
            if torch.any(priv_nan_mask) or torch.any(priv_inf_mask):
                logger.warning(f"Found NaN/Inf in privileged observations. NaN count: {torch.sum(priv_nan_mask)}, Inf count: {torch.sum(priv_inf_mask)}")
                self.task_obs["priviliged_obs"][priv_nan_mask | priv_inf_mask] = 0.0


@torch.jit.script
def ssa(a: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle"""
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi
