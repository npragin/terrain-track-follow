from aerial_gym.task.navigation_task.navigation_task import NavigationTask
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)

from aerial_gym.utils.math import quat_rotate_inverse, get_euler_xyz_tensor
import torch
import numpy as np

class DCE_RL_Navigation_Task(NavigationTask):
    def __init__(self, task_config, **kwargs):
        task_config.action_space_dim = 3
        task_config.curriculum.min_level = 36
        logger.critical("Hardcoding number of envs to 16 if it is greater than that.")
        task_config.num_envs = 16 if task_config.num_envs > 16 else task_config.num_envs

        super().__init__(task_config=task_config, **kwargs)

        # Initialize moving object parameters
        self.movement_timer = torch.zeros(self.num_envs, device=self.device)
        # Initialize with random directions
        self.movement_direction = torch.randn((self.num_envs, 3), device=self.device)
        self.movement_direction[:, 2] = 0  # No vertical movement
        self.movement_direction = self.movement_direction / torch.norm(self.movement_direction, dim=1, keepdim=True)
        self.movement_speed = 0.7  # m/s (increased for more visible movement)
        self.direction_change_interval = 0.3  # seconds
        self.dt = self.sim_env.global_tensor_dict["dt"]
        
        # Initialize infos dictionary
        self.infos = {}
        
        # Debug logging
        logger.info(f"Initialized DCE navigation task with {self.num_envs} environments")
        if "obstacle_position" in self.obs_dict:
            num_obstacles = self.obs_dict["obstacle_position"].shape[1]
            logger.info(f"Found {num_obstacles} obstacles per environment")
        else:
            logger.warning("No obstacle_position found in obs_dict - moving objects may not work")

    def step(self, actions):
        # Generate obstacle actions for moving objects
        env_actions = self._generate_obstacle_actions()
        
        # Call the simulation step with obstacle actions
        transformed_action = self.action_transformation_function(actions)
        logger.debug(f"raw_action: {actions[0]}, transformed action: {transformed_action[0]}")
        self.sim_env.step(actions=transformed_action, env_actions=env_actions)
        
        # Continue with the rest of the step logic from the parent class
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)
        
        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()
        
        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )
        
        # successes are are the sum of the environments which are to be truncated and have reached the target within a distance threshold
        successes = self.truncations * (
            torch.norm(self.target_position - self.obs_dict["robot_position"], dim=1) < 1.0
        )
        successes = torch.where(self.terminations > 0, torch.zeros_like(successes), successes)
        timeouts = torch.where(
            self.truncations > 0, torch.logical_not(successes), torch.zeros_like(successes)
        )
        timeouts = torch.where(
            self.terminations > 0, torch.zeros_like(timeouts), timeouts
        )  # timeouts are not counted if there is a crash
        
        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = self.terminations
        
        self.logging_sanity_check(self.infos)
        self.check_and_update_curriculum_level(
            self.infos["successes"], self.infos["crashes"], self.infos["timeouts"]
        )
        
        # Handle resets
        envs_to_reset = self.sim_env.post_reward_calculation_step()
        if len(envs_to_reset) > 0:
            self.reset_idx(envs_to_reset)
        
        return self.get_return_tuple()
    
    def _generate_obstacle_actions(self):
        """Generate obstacle actions for moving objects"""
        if "obstacle_position" not in self.obs_dict:
            logger.warning("No obstacle_position found in obs_dict")
            return None
            
        # Update movement timer
        self.movement_timer += self.dt
        
        # Check if it's time to change direction for each environment
        change_direction = self.movement_timer >= self.direction_change_interval
        
        # Generate new random directions for environments that need direction change
        if torch.any(change_direction):
            # Generate random directions (only in x-y plane, z=0 for ground movement)
            new_directions = torch.randn((self.num_envs, 3), device=self.device)
            new_directions[:, 2] = 0  # No vertical movement
            new_directions = new_directions / torch.norm(new_directions, dim=1, keepdim=True)
            
            # Update directions for environments that need change
            self.movement_direction[change_direction] = new_directions[change_direction]
            self.movement_timer[change_direction] = 0.0
            logger.info(f"Changed direction for {torch.sum(change_direction)} environments")
        
        # Generate obstacle actions
        num_obstacles = self.obs_dict["obstacle_position"].shape[1]
        logger.debug(f"Generating actions for {num_obstacles} obstacles")
        
        # Create obstacle actions tensor: [num_envs, num_obstacles, 6]
        # First 3 dimensions are linear velocity, last 3 are angular velocity
        obstacle_actions = torch.zeros((self.num_envs, num_obstacles, 6), device=self.device)
        
        # Set linear velocities (x, y, z) - only x and y movement
        obstacle_actions[:, :, 0] = self.movement_direction[:, 0:1] * self.movement_speed  # x velocity
        obstacle_actions[:, :, 1] = self.movement_direction[:, 1:2] * self.movement_speed  # y velocity
        obstacle_actions[:, :, 2] = 0.0  # z velocity (no vertical movement)
        
        # Set angular velocities to zero (no rotation)
        obstacle_actions[:, :, 3:6] = 0.0
        
        # Debug logging for velocity commands
        # print(f"Movement direction (env 0): {self.movement_direction[0]}")
        # print(f"Movement speed: {self.movement_speed}")
        # print(f"Obstacle actions (env 0, obstacle 0): {obstacle_actions[0, 0]}")
        # print(f"Obstacle actions shape: {obstacle_actions.shape}")
        # print(f"Sample obstacle action (env 0): {obstacle_actions[0, 0]}")
        
        # Log obstacle positions for debugging
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 0
            
        if self._step_count % 100 == 0:  # Log every 100 steps
            obstacle_pos = self.obs_dict["obstacle_position"]
            logger.info(f"Step {self._step_count} - Obstacle positions (env 0): {obstacle_pos[0, :, :3]}")
            logger.info(f"Obstacle velocities (env 0): {obstacle_actions[0, :, :3]}")
        
        return obstacle_actions

    def reset_idx(self, env_ids):
        """Reset moving object parameters when environments are reset"""
        super().reset_idx(env_ids)
        
        # Reset movement parameters for reset environments
        self.movement_timer[env_ids] = 0.0
        
        # Generate new random directions for reset environments
        new_directions = torch.randn((len(env_ids), 3), device=self.device)
        new_directions[:, 2] = 0  # No vertical movement
        new_directions = new_directions / torch.norm(new_directions, dim=1, keepdim=True)
        self.movement_direction[env_ids] = new_directions


    def process_obs_for_task(self):
        vec_to_target = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            (self.target_position - self.obs_dict["robot_position"]),
        )
        dist_to_tgt = torch.norm(vec_to_target, dim=1)
        self.task_obs["observations"][:, 0:3] = vec_to_target / dist_to_tgt.unsqueeze(1)
        self.task_obs["observations"][:, 3] = dist_to_tgt / 5.0
        euler_angles = ssa(get_euler_xyz_tensor(self.obs_dict["robot_vehicle_orientation"]))
        self.task_obs["observations"][:, 4:6] = euler_angles[:, 0:2]
        self.task_obs["observations"][:, 6] = 0.0
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 13:17] = self.obs_dict["robot_actions"]
        self.task_obs["observations"][:, 17:81] = self.image_latents


@torch.jit.script
def ssa(a: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle"""
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi
