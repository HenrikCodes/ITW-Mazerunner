import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
from gymnasium import spaces

class TiltingPlateBallEnv(MujocoEnv):
    """Inherit from Gymnasium's MujocoEnv for built-in handling."""
    
    def __init__(self, render_mode=None, max_episode_steps=500):
        # Initialize MujocoEnv with your XML file
        super().__init__(
            model_path="tilting_plate.xml",
            frame_skip=1,
            observation_space=spaces.Box(
                low=np.array([-0.5236, -0.5236, -0.3, -0.3, -2.0, -2.0], dtype=np.float32),
                high=np.array([0.5236, 0.5236, 0.3, 0.3, 2.0, 2.0], dtype=np.float32),
                dtype=np.float32
            ),
            render_mode=render_mode
        )
        
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
    
    def step(self, action):
        # Action is automatically applied via self.data.ctrl
        obs, _, _, _, info = super().step(action)
        
        self.current_step += 1
        
        # Compute custom reward
        reward = self._compute_reward()
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """Extract observation from data."""
        return np.array([
            self.data.qpos[self.model.jnt_qposadr],
            self.data.qpos[self.model.jnt_qposadr],
            self.data.qpos[self.model.jnt_qposadr],
            self.data.qpos[self.model.jnt_qposadr],
            self.data.qvel[self.model.jnt_dofadr],
            self.data.qvel[self.model.jnt_dofadr],
        ], dtype=np.float32)
    
    def _compute_reward(self):
        obs = self._get_obs()
        ball_x, ball_y = obs, obs
        plate_x, plate_y = obs, obs
        
        pos_error = np.sqrt(ball_x**2 + ball_y**2)
        return -0.1 * pos_error - 0.01 * (plate_x**2 + plate_y**2)
    
    def _is_terminated(self):
        obs = self._get_obs()
        return abs(obs) > 0.15 or abs(obs) > 0.15
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), info
