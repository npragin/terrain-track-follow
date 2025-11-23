# isaacgym must be imported before torch
import distutils
import os
from enum import IntEnum

import gym
import isaacgym  # noqa: F401
import numpy as np
import torch
import yaml
from gym import spaces
from rl_games.common import env_configurations, vecenv

from aerial_gym.registry.task_registry import task_registry

# Register custom asymmetric actor-critic network
from aerial_gym.rl_training.rl_games.asymmetric_actor_critic import register_asymmetric_network
from aerial_gym.utils.helpers import parse_arguments

register_asymmetric_network()

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# import warnings
# warnings.filterwarnings("error")


class TerminationState(IntEnum):
    NO_TERMINATION = 0
    SUCCESS = 1
    TERMINATION = 2
    TRUNCATION = 3


class EpisodeStatsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env_steps = 0
        self.prev_dones = None
        self.env_termination_states = None
        self.num_envs = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        if self.num_envs is None:
            self.num_envs = self.env.num_envs

        device = "cpu"
        if isinstance(observations, torch.Tensor):
            device = observations.device
        elif isinstance(observations, dict):
            for val in observations.values():
                if isinstance(val, torch.Tensor):
                    device = val.device
                    break

        if self.prev_dones is None:
            self.prev_dones = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        if self.env_termination_states is None:
            self.env_termination_states = torch.full(
                (self.num_envs,), TerminationState.NO_TERMINATION, dtype=torch.int32, device=device
            )

        return observations

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)
        self.env_steps += 1

        if not isinstance(terminated, torch.Tensor):
            device = (
                truncated.device
                if isinstance(truncated, torch.Tensor)
                else (self.prev_dones.device if self.prev_dones is not None else "cpu")
            )
            terminated = torch.tensor(terminated, dtype=torch.bool, device=device)
        if not isinstance(truncated, torch.Tensor):
            device = (
                terminated.device
                if isinstance(truncated, torch.Tensor)
                else (self.prev_dones.device if self.prev_dones is not None else "cpu")
            )
            truncated = torch.tensor(truncated, dtype=torch.bool, device=device)

        dones = terminated | truncated

        if self.prev_dones is not None:
            if self.prev_dones.device != dones.device:
                self.prev_dones = self.prev_dones.to(dones.device)
                if self.env_termination_states.device != dones.device:
                    self.env_termination_states = self.env_termination_states.to(dones.device)
            newly_done = ~self.prev_dones & dones

            if newly_done.any():
                successes_tensor = None
                if isinstance(infos, dict) and "successes" in infos:
                    successes_tensor = infos["successes"]
                elif isinstance(self.env.infos, dict) and "successes" in self.env.infos:
                    successes_tensor = self.env.infos["successes"]

                for env_id in torch.where(newly_done)[0]:
                    env_id = env_id.item()
                    if (
                        successes_tensor is not None
                        and isinstance(successes_tensor, torch.Tensor)
                        and successes_tensor[env_id]
                    ):
                        self.env_termination_states[env_id] = TerminationState.SUCCESS
                    elif terminated[env_id]:
                        self.env_termination_states[env_id] = TerminationState.TERMINATION
                    elif truncated[env_id]:
                        self.env_termination_states[env_id] = TerminationState.TRUNCATION

                self._log_stats()

        self.prev_dones = dones.clone()
        return observations, rewards, terminated, truncated, infos

    def _log_stats(self):
        import wandb

        if wandb.run is None:
            return

        success_count = (self.env_termination_states == TerminationState.SUCCESS).sum().item()
        termination_count = (self.env_termination_states == TerminationState.TERMINATION).sum().item()
        truncation_count = (self.env_termination_states == TerminationState.TRUNCATION).sum().item()
        total_terminated = success_count + termination_count + truncation_count

        if total_terminated > 0:
            success_rate = success_count / total_terminated
            truncation_rate = truncation_count / total_terminated
            termination_rate = termination_count / total_terminated
        else:
            success_rate = 0.0
            truncation_rate = 0.0
            termination_rate = 0.0

        metrics = {
            "success_rate/step": success_rate,
            "truncation_rate/step": truncation_rate,
            "termination_rate/step": termination_rate,
        }

        curriculum_fraction = self.env.curriculum_progress_fraction
        if isinstance(curriculum_fraction, torch.Tensor):
            if curriculum_fraction.numel() == 1:
                curriculum_fraction = curriculum_fraction.item()
            else:
                curriculum_fraction = curriculum_fraction[0].item()
        metrics["curriculum_fraction/step"] = curriculum_fraction

        wandb.log(metrics, step=self.env_steps)


class ExtractObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Check if environment has privileged observations for asymmetric actor-critic
        self.has_privileged_obs = hasattr(env, "task_config") and env.task_config.privileged_observation_space_dim > 0
        self.obs_dim = env.task_config.observation_space_dim if hasattr(env, "task_config") else None
        self.priv_dim = env.task_config.privileged_observation_space_dim if hasattr(env, "task_config") else 0

    def reset(self, **kwargs):
        observations, *_ = super().reset(**kwargs)

        # For asymmetric actor-critic with separate=True:
        # Concatenate all observations and let the network split them internally
        if self.has_privileged_obs and "priviliged_obs" in observations:
            # Return full concatenated observations [81 + 4 = 85]
            # Actor network will use [:81], critic network will use all 85
            full_obs = torch.cat([observations["observations"], observations["priviliged_obs"]], dim=-1)
            return full_obs
        else:
            return observations["observations"]

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)

        dones = torch.where(
            terminated | truncated,
            torch.ones_like(terminated),
            torch.zeros_like(terminated),
        )

        # For asymmetric actor-critic with separate=True:
        # Concatenate all observations and let the network split them internally
        if self.has_privileged_obs and "priviliged_obs" in observations:
            full_obs = torch.cat([observations["observations"], observations["priviliged_obs"]], dim=-1)
            return (full_obs, rewards, dones, infos)
        else:
            return (observations["observations"], rewards, dones, infos)


class AERIALRLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.env = EpisodeStatsWrapper(self.env)
        self.env = ExtractObsWrapper(self.env)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = spaces.Box(
            -np.ones(self.env.task_config.action_space_dim),
            np.ones(self.env.task_config.action_space_dim),
        )

        # For asymmetric actor-critic: observation_space is the FULL concatenated size
        # The custom network will split it internally (actor uses [:81], critic uses all 85)
        if hasattr(self.env, "task_config") and self.env.task_config.privileged_observation_space_dim > 0:
            full_dim = (
                self.env.task_config.observation_space_dim + self.env.task_config.privileged_observation_space_dim
            )
            info["observation_space"] = spaces.Box(
                np.ones(full_dim) * -np.inf,
                np.ones(full_dim) * np.inf,
            )
            print(f"Action space: {info['action_space']}")
            print(f"Observation space (full): {info['observation_space']}")
            print(f"  → Actor will use first {self.env.task_config.observation_space_dim} dims")
            print(f"  → Critic will use all {full_dim} dims (asymmetric)")
        else:
            info["observation_space"] = spaces.Box(
                np.ones(self.env.task_config.observation_space_dim) * -np.inf,
                np.ones(self.env.task_config.observation_space_dim) * np.inf,
            )
            print(info["action_space"], info["observation_space"])

        return info


env_configurations.register(
    "position_setpoint_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("position_setpoint_task", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_sim2real",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("position_setpoint_task_sim2real", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_acceleration_sim2real",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_acceleration_sim2real", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "navigation_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("navigation_task", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "track_follow_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("track_follow_task", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_reconfigurable",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("position_setpoint_task_reconfigurable", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_morphy",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("position_setpoint_task_morphy", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_sim2real_end_to_end",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("position_setpoint_task_sim2real_end_to_end", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

vecenv.register(
    "AERIAL-RLGPU",
    lambda config_name, num_actors, **kwargs: AERIALRLGPUEnv(config_name, num_actors, **kwargs),
)


def get_args():
    custom_parameters = [
        {
            "name": "--seed",
            "type": int,
            "default": 0,
            "required": False,
            "help": "Random seed, if larger than 0 will overwrite the value in yaml config.",
        },
        {
            "name": "--tf",
            "required": False,
            "help": "run tensorflow runner",
            "action": "store_true",
        },
        {
            "name": "--train",
            "required": False,
            "help": "train network",
            "action": "store_true",
        },
        {
            "name": "--play",
            "required": False,
            "help": "play(test) network",
            "action": "store_true",
        },
        {
            "name": "--checkpoint",
            "type": str,
            "required": False,
            "help": "path to checkpoint",
        },
        {
            "name": "--file",
            "type": str,
            "default": "ppo_aerial_quad.yaml",
            "required": False,
            "help": "path to config",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": "1024",
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--sigma",
            "type": float,
            "required": False,
            "help": "sets new sigma value in case if 'fixed_sigma: True' in yaml config",
        },
        {
            "name": "--track",
            "action": "store_true",
            "help": "if toggled, this experiment will be tracked with Weights and Biases",
        },
        {
            "name": "--wandb-project-name",
            "type": str,
            "default": "rl_games",
            "help": "the wandb's project name",
        },
        {
            "name": "--wandb-entity",
            "type": str,
            "default": None,
            "help": "the entity (team) of wandb's project",
        },
        {
            "name": "--task",
            "type": str,
            "default": "navigation_task",
            "help": "Override task from config file if provided.",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "False",
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--use_warp",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "True",
            "help": "Choose whether to use warp or Isaac Gym rendeing pipeline.",
        },
    ]

    # parse arguments
    args = parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def update_config(config, args):
    if args["task"] is not None:
        config["params"]["config"]["env_name"] = args["task"]
    if args["experiment_name"] is not None:
        config["params"]["config"]["name"] = args["experiment_name"]
    config["params"]["config"]["env_config"]["headless"] = args["headless"]
    config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    config["params"]["config"]["env_config"]["use_warp"] = args["use_warp"]
    if args["num_envs"] > 0:
        config["params"]["config"]["num_actors"] = args["num_envs"]
        # config['params']['config']['num_envs'] = args['num_envs']
        config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    if args["seed"] > 0:
        config["params"]["seed"] = args["seed"]
        config["params"]["config"]["env_config"]["seed"] = args["seed"]

    config["params"]["config"]["player"] = {"use_vecenv": True}
    return config


if __name__ == "__main__":
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    args = vars(get_args())

    config_name = args["file"]

    print("Loading config: ", config_name)
    with open(config_name) as stream:
        config = yaml.safe_load(stream)

        config = update_config(config, args)

        from rl_games.torch_runner import Runner

        runner = Runner()
        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            print(exc)

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if args["track"] and rank == 0:
        import wandb

        wandb.init(
            project=args["wandb_project_name"],
            entity=args["wandb_entity"],
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )
    runner.run(args)

    if args["track"] and rank == 0:
        wandb.finish()
