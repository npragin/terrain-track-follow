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

from aerial_gym.utils.math import get_euler_xyz_tensor


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

    def post_image_reward_addition(self):
        """
        Override to provide dense reward: 1 if target is seen, 0 if not.
        This replaces the depth-based reward with a simple binary reward based on target visibility.
        """
        seg_mask = self.obs_dict["segmentation_pixels"][:, 0, :, :]  # [num_envs, height, width]
        target_mask = seg_mask == TARGET_SEMANTIC_ID  # [num_envs, height, width]
        target_seen = torch.any(target_mask.view(seg_mask.shape[0], -1), dim=1)  # [num_envs]
        target_reward = target_seen.float()  # [num_envs]
        self.rewards[self.terminations < 0] += target_reward[self.terminations < 0]

    def process_obs_for_task(self):
        # Extract target bounding box from segmentation camera
        if "segmentation_pixels" in self.obs_dict:
            target_bbox = self.extract_target_bbox_from_segmentation(self.obs_dict["segmentation_pixels"])
            # Replace vector to target (3D) + distance (1D) with bounding box (4D)
            self.task_obs["observations"][:, 0:4] = target_bbox
        else:
            # If segmentation not available, use zeros
            logger.warning("segmentation_pixels not available, using zero bounding box")
            self.task_obs["observations"][:, 0:4] = 0.0

        euler_angles = ssa(get_euler_xyz_tensor(self.obs_dict["robot_vehicle_orientation"]))
        self.task_obs["observations"][:, 4:6] = euler_angles[:, 0:2]
        self.task_obs["observations"][:, 6] = 0.0
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 13:17] = self.obs_dict["robot_actions"]
        self.task_obs["observations"][:, 17:81] = self.image_latents

        # Save segmentation visualization (only saves every N steps internally)
        # self.save_segmentation_visualization()


@torch.jit.script
def ssa(a: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle"""
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi
