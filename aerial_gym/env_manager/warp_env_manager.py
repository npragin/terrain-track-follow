import numpy as np
import torch
import trimesh as tm
import warp as wp

from aerial_gym.env_manager.base_env_manager import BaseManager
from aerial_gym.utils.math import tf_apply

# intialize warp
wp.init()

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("warp_env_manager")


class WarpEnv(BaseManager):
    def __init__(self, global_sim_dict, device):
        logger.debug("Initializing WarpEnv")
        super().__init__(global_sim_dict["env_cfg"], device)
        self.num_envs = global_sim_dict["num_envs"]
        self.env_meshes = []
        self.warp_mesh_id_list = []
        self.warp_mesh_per_env = []
        self.global_vertex_to_asset_index_tensor = None
        self.vertex_maps_per_env_original = None
        self.global_env_mesh_list = []
        self.global_vertex_counter = 0
        self.global_vertex_segmentation_list = []
        self.global_vertex_to_asset_index_map = []

        self.CONST_WARP_MESH_ID_LIST = None
        self.CONST_WARP_MESH_PER_ENV = None
        self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR = None
        self.VERTEX_MAPS_PER_ENV_ORIGINAL = None
        logger.debug("[DONE] Initializing WarpEnv")

    def reset_idx(self, env_ids):
        if self.global_vertex_counter == 0:
            return
        # logger.debug("Updating vertex maps per env")
        # Terrain vertices use asset index -1 (static, no transform)
        # First, copy all original vertices (including terrain) to updated maps
        self.vertex_maps_per_env_updated[:] = self.VERTEX_MAPS_PER_ENV_ORIGINAL[:]

        # Then, only transform vertices that belong to dynamic assets (not terrain)
        # Filter out terrain vertices (index -1) and ensure indices are within bounds
        valid_mask = self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR >= 0
        if torch.any(valid_mask):
            valid_indices = self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR[valid_mask]
            # Ensure indices are within bounds of unfolded_env_vec_root_tensor
            max_index = self.unfolded_env_vec_root_tensor.shape[0] - 1
            in_bounds_mask = (valid_indices >= 0) & (valid_indices <= max_index)

            if torch.any(in_bounds_mask):
                # Get the positions in the full mask where we have valid, in-bounds indices
                valid_positions = torch.where(valid_mask)[0]
                safe_positions = valid_positions[in_bounds_mask]
                safe_indices = valid_indices[in_bounds_mask]

                self.vertex_maps_per_env_updated[safe_positions] = tf_apply(
                    self.unfolded_env_vec_root_tensor[safe_indices, 3:7],
                    self.unfolded_env_vec_root_tensor[safe_indices, 0:3],
                    self.VERTEX_MAPS_PER_ENV_ORIGINAL[safe_positions],
                )
        # Terrain vertices (index -1) remain at their original positions (static)
        # logger.debug("[DONE] Updating vertex maps per env")

        # logger.debug("Refitting warp meshes")
        for i in env_ids:
            self.warp_mesh_per_env[i].refit()
        # logger.debug("[DONE] Refitting warp meshes")

    def pre_physics_step(self, action):
        pass

    def post_physics_step(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        return self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_env(self, env_id):
        if len(self.env_meshes) <= env_id:
            self.env_meshes.append([])
        else:
            raise ValueError("Environment already exists")

    def add_terrain_mesh_to_env(self, terrain_gen, env_id, global_asset_counter, segmentation_counter):
        """
        Convert terrain heightmap to trimesh and add it to Warp environment for depth camera raycasting.

        Args:
            terrain_gen: TerrainGenerator instance with heightmap data
            env_id: Environment ID
            global_asset_counter: Global asset counter for indexing
            segmentation_counter: Segmentation counter for terrain (typically 0 or a fixed value)

        """
        if terrain_gen is None:
            return

        # Get heightmap from terrain generator
        heightmap = terrain_gen.generate_heightmap(use_cache=True)
        resolution = terrain_gen.resolution
        scale_x = terrain_gen.scale_x
        scale_y = terrain_gen.scale_y
        amplitude = terrain_gen.amplitude
        transform_x = getattr(terrain_gen, "transform_x", -scale_x / 2.0)
        transform_y = getattr(terrain_gen, "transform_y", -scale_y / 2.0)

        x = np.linspace(transform_x, transform_x + scale_x, resolution)
        y = np.linspace(transform_y, transform_y + scale_y, resolution)
        X, Y = np.meshgrid(x, y)

        # Heightmap is in range [-amplitude/2, amplitude/2] from terrain generator
        # Isaac Gym applies: heightmap_normalized = heightmap / amplitude, then offset by amplitude/2
        # So final z = (heightmap / amplitude) * amplitude + amplitude/2 = heightmap + amplitude/2
        # This gives range [0, amplitude] which matches Isaac Gym's transform.p.z = amplitude/2.0
        Z = heightmap + amplitude / 2.0
        vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

        # Create faces (triangles) for the heightmap
        # Each grid cell becomes 2 triangles
        faces = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                # Calculate vertex indices for this grid cell
                idx = i * resolution + j
                idx_right = i * resolution + (j + 1)
                idx_down = (i + 1) * resolution + j
                idx_down_right = (i + 1) * resolution + (j + 1)

                # Two triangles per cell
                # Triangle 1: (idx, idx_right, idx_down)
                faces.append([idx, idx_right, idx_down])
                # Triangle 2: (idx_right, idx_down_right, idx_down)
                faces.append([idx_right, idx_down_right, idx_down])

        faces = np.array(faces, dtype=np.int32)

        # Create trimesh from vertices and faces
        terrain_mesh = tm.Trimesh(vertices=vertices, faces=faces)

        # Add to environment meshes
        self.env_meshes[env_id].append(terrain_mesh)

        # Update global counters
        num_vertices = len(terrain_mesh.vertices)
        # Use segmentation_counter for terrain (typically 0 or a fixed semantic ID)
        terrain_segmentation_value = segmentation_counter
        # Use -1 as asset index for terrain (static, not in unfolded_env_vec_root_tensor)
        # This will be handled specially in reset_idx to keep terrain vertices static
        self.global_vertex_to_asset_index_map += [-1] * num_vertices
        self.global_vertex_counter += num_vertices
        self.global_vertex_segmentation_list += [terrain_segmentation_value] * num_vertices

        logger.debug(f"Added terrain mesh to environment {env_id} with {num_vertices} vertices")
        return None, 1  # Terrain is a single asset

    def add_asset_to_env(self, asset_info_dict, env_id, global_asset_counter, segmentation_counter):
        warp_asset = asset_info_dict["warp_asset"]
        # use the variable segmentation mask to set the segmentation id for each vertex
        updated_vertex_segmentation = (
            warp_asset.asset_vertex_segmentation_value + segmentation_counter * warp_asset.variable_segmentation_mask
        )
        logger.debug(
            f"Asset {asset_info_dict['filename']} has {len(warp_asset.asset_unified_mesh.vertices)} vertices. Segmentation mask: {warp_asset.variable_segmentation_mask} and updated segmentation: {updated_vertex_segmentation}"
        )
        self.env_meshes[env_id].append(warp_asset.asset_unified_mesh)

        self.global_vertex_to_asset_index_map += [global_asset_counter] * len(warp_asset.asset_unified_mesh.vertices)
        self.global_vertex_counter += len(warp_asset.asset_unified_mesh.vertices)
        self.global_vertex_segmentation_list += updated_vertex_segmentation.tolist()
        return None, len(np.unique(warp_asset.asset_vertex_segmentation_value * warp_asset.variable_segmentation_mask))

    def prepare_for_simulation(self, global_tensor_dict):
        logger.debug("Preparing for simulation")
        self.global_tensor_dict = global_tensor_dict
        if self.global_vertex_counter == 0:
            logger.warning("No assets have been added to the environment. Skipping preparation for simulation")
            self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"] = None
            self.global_tensor_dict["CONST_WARP_MESH_PER_ENV"] = None
            self.global_tensor_dict["CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR"] = None
            self.global_tensor_dict["VERTEX_MAPS_PER_ENV_ORIGINAL"] = None
            return 1

        self.global_vertex_to_asset_index_tensor = torch.tensor(
            self.global_vertex_to_asset_index_map,
            device=self.device,
            requires_grad=False,
        )
        self.vertex_maps_per_env_original = torch.zeros(
            (self.global_vertex_counter, 3), device=self.device, requires_grad=False
        )
        # updated vertex maps are used for the warp environment
        self.vertex_maps_per_env_updated = self.vertex_maps_per_env_original.clone()

        ## unify env meshes
        logger.debug("Unifying environment meshes")
        for i in range(len(self.env_meshes)):
            self.global_env_mesh_list.append(tm.util.concatenate(self.env_meshes[i]))
        logger.debug("[DONE] Unifying environment meshes")

        # prepare warp meshes
        logger.debug("Creating warp meshes")
        vertex_iterator = 0
        for env_mesh in self.global_env_mesh_list:
            self.vertex_maps_per_env_original[vertex_iterator : vertex_iterator + len(env_mesh.vertices)] = (
                torch.tensor(env_mesh.vertices, device=self.device, requires_grad=False)
            )
            faces_tensor = torch.tensor(
                env_mesh.faces,
                device=self.device,
                requires_grad=False,
                dtype=torch.int32,
            )
            vertex_velocities = torch.zeros(len(env_mesh.vertices), 3, device=self.device, requires_grad=False)
            segmentation_tensor = torch.tensor(
                self.global_vertex_segmentation_list[vertex_iterator : vertex_iterator + len(env_mesh.vertices)],
                device=self.device,
                requires_grad=False,
            )
            # we hijack this field and use it for segmentation
            vertex_velocities[:, 0] = segmentation_tensor

            vertex_vec3_array = wp.from_torch(
                self.vertex_maps_per_env_updated[vertex_iterator : vertex_iterator + len(env_mesh.vertices)],
                dtype=wp.vec3,
            )
            faces_wp_int32_array = wp.from_torch(faces_tensor.flatten(), dtype=wp.int32)
            velocities_vec3_array = wp.from_torch(vertex_velocities, dtype=wp.vec3)

            wp_mesh = wp.Mesh(
                points=vertex_vec3_array,
                indices=faces_wp_int32_array,
                velocities=velocities_vec3_array,
            )

            self.warp_mesh_per_env.append(wp_mesh)
            self.warp_mesh_id_list.append(wp_mesh.id)
            vertex_iterator += len(env_mesh.vertices)

        logger.debug("[DONE] Creating warp meshes")
        # define consts so that they can be accessed only after the environment has been prepared
        self.CONST_WARP_MESH_ID_LIST = self.warp_mesh_id_list
        self.CONST_WARP_MESH_PER_ENV = self.warp_mesh_per_env
        self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR = self.global_vertex_to_asset_index_tensor
        self.VERTEX_MAPS_PER_ENV_ORIGINAL = self.vertex_maps_per_env_original

        self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"] = self.CONST_WARP_MESH_ID_LIST
        self.global_tensor_dict["CONST_WARP_MESH_PER_ENV"] = self.CONST_WARP_MESH_PER_ENV
        self.global_tensor_dict["CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR"] = (
            self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR
        )
        self.global_tensor_dict["VERTEX_MAPS_PER_ENV_ORIGINAL"] = self.VERTEX_MAPS_PER_ENV_ORIGINAL

        self.unfolded_env_vec_root_tensor = self.global_tensor_dict["unfolded_env_asset_state_tensor"]
        return 1
