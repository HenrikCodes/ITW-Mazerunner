"""
Custom Gymnasium environment for a dual-axis tilting plate with rolling ball.

This environment provides the foundation for training RL agents to control
a plate that tilts in two axes, with a ball rolling on its surface.

The physics simulation (ball dynamics) should be plugged into the `step` method
and the reward function can be customized based on ball position/velocity policy.

Author: Henrik Neibig
Date: January 2026
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TiltingPlateBallEnv(gym.Env):
    """
    Gymnasium environment for a tilting plate with a rolling ball.
    
    The plate can be tilted around two axes (θ_x and θ_y) controlled by motors.
    A ball rolls on the plate surface based on gravity and inertia.
    
    Observation Space (6 continuous values):
        - plate_angle_x: tilt angle around x-axis (radians)
        - plate_angle_y: tilt angle around y-axis (radians)
        - ball_pos_x: ball position on plate surface (normalized to [-1, 1])
        - ball_pos_y: ball position on plate surface (normalized to [-1, 1])
        - ball_vel_x: ball velocity x-direction (m/s)
        - ball_vel_y: ball velocity y-direction (m/s)
    
    Action Space (2 continuous values):
        - action[0]: motor command for x-axis tilt (rad/s or control signal)
        - action[1]: motor command for y-axis tilt (rad/s or control signal)
        
        Both actions are normalized to [-1, 1], you can rescale as needed.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(self, render_mode=None, max_episode_steps=500):
        """
        Initialize the environment.
        
        Args:
            render_mode: How to render ('human', 'rgb_array', or None)
            max_episode_steps: Maximum steps before episode truncates
        """
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # ==================== Physical Parameters ====================
        
        # Plate dimensions and constraints
        self.plate_width = 0.2  # meters
        self.plate_length = 0.2  # meters
        self.max_tilt_angle = np.pi / 6  # ±30 degrees max tilt
        
        # Motor dynamics parameters
        self.motor_response_time = 0.1  # seconds, for low-pass filtering
        self.motor_gain = 2.0  # rad/s per action unit
        
        # Ball parameters
        self.ball_radius = 0.01  # meters
        self.gravity = 9.81  # m/s²
        self.friction_coefficient = 0.2  # rolling friction
        
        # Physics simulation dt (you'll integrate your own physics here)
        self.dt = 0.01  # 10ms per step
        
        # ==================== State Variables ====================
        
        # Plate state
        self.plate_angle_x = 0.0  # current tilt around x-axis (radians)
        self.plate_angle_y = 0.0  # current tilt around y-axis (radians)
        self.plate_angular_vel_x = 0.0  # angular velocity (rad/s)
        self.plate_angular_vel_y = 0.0  # angular velocity (rad/s)
        
        # Ball state (relative to plate center)
        self.ball_pos_x = 0.0  # position on plate, normalized [-1, 1]
        self.ball_pos_y = 0.0  # position on plate, normalized [-1, 1]
        self.ball_vel_x = 0.0  # velocity (m/s)
        self.ball_vel_y = 0.0  # velocity (m/s)
        
        # ==================== Space Definitions ====================
        
        # Observation space: [plate_angle_x, plate_angle_y, 
        #                     ball_pos_x, ball_pos_y, 
        #                     ball_vel_x, ball_vel_y]
        self.observation_space = spaces.Box(
            low=np.array([
                -self.max_tilt_angle,  # plate_angle_x
                -self.max_tilt_angle,  # plate_angle_y
                -1.0,  # ball_pos_x normalized
                -1.0,  # ball_pos_y normalized
                -1.0,  # ball_vel_x max (m/s)
                -1.0   # ball_vel_y max (m/s)
            ], dtype=np.float32),
            high=np.array([
                self.max_tilt_angle,
                self.max_tilt_angle,
                1.0,
                1.0,
                1.0,
                1.0
            ], dtype=np.float32),
            dtype=np.float32,
            shape=(6,)
        )
        
        # Action space: [motor_x_command, motor_y_command], both in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
            shape=(2,)
        )
        
        self.rng = np.random.default_rng()
    
    def _get_obs(self):
        """
        Get the current observation as a numpy array.
        
        Returns:
            np.ndarray: [plate_angle_x, plate_angle_y, 
                        ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y]
        """
        return np.array([
            self.plate_angle_x,
            self.plate_angle_y,
            self.ball_pos_x,
            self.ball_pos_y,
            self.ball_vel_x,
            self.ball_vel_y
        ], dtype=np.float32)
    
    def _get_info(self):
        """
        Get auxiliary information about the environment state.
        
        Returns:
            dict: Additional info for debugging/analysis
        """
        return {
            "plate_angle_x": float(self.plate_angle_x),
            "plate_angle_y": float(self.plate_angle_y),
            "ball_pos_x": float(self.ball_pos_x),
            "ball_pos_y": float(self.ball_pos_y),
            "ball_vel_x": float(self.ball_vel_x),
            "ball_vel_y": float(self.ball_vel_y),
            "ball_on_plate": self._is_ball_on_plate(),
        }
    
    def _is_ball_on_plate(self):
        """Check if ball is still on the plate (within bounds)."""
        return abs(self.ball_pos_x) <= 1.0 and abs(self.ball_pos_y) <= 1.0
    
    def _update_plate_physics(self, action):
        """
        Update plate angles based on motor commands (action).
        
        This implements a simplified motor response model. You can replace
        this with your actual motor dynamics.
        
        Args:
            action: np.ndarray of shape (2,) with values in [-1, 1]
        """
        # Convert action to angular velocity commands
        angular_vel_target_x = action[0] * self.motor_gain
        angular_vel_target_y = action[1] * self.motor_gain
        
        # Simple first-order response (low-pass filter)
        alpha = 1.0 - np.exp(-self.dt / self.motor_response_time)
        self.plate_angular_vel_x += alpha * (angular_vel_target_x - self.plate_angular_vel_x)
        self.plate_angular_vel_y += alpha * (angular_vel_target_y - self.plate_angular_vel_y)
        
        # Update angles
        new_angle_x = self.plate_angle_x + self.plate_angular_vel_x * self.dt
        new_angle_y = self.plate_angle_y + self.plate_angular_vel_y * self.dt
        
        # Clamp to max tilt angle
        self.plate_angle_x = np.clip(new_angle_x, -self.max_tilt_angle, self.max_tilt_angle)
        self.plate_angle_y = np.clip(new_angle_y, -self.max_tilt_angle, self.max_tilt_angle)
    
    def _update_ball_physics(self):
        """
        Update ball position and velocity based on plate orientation.
        
        IMPORTANT: This is a PLACEHOLDER implementation showing the structure.
        You should replace this with your actual physics simulation that includes:
            - Gravity effects based on plate tilt
            - Rolling dynamics
            - Friction modeling
            - Collision with plate edges
            - Slip vs. roll conditions
        
        For a proper implementation, integrate your physics engine here.
        """
        # Placeholder: simple gravity-based acceleration
        # Real implementation should use your physics simulation
        
        accel_x = self.gravity * np.sin(self.plate_angle_x)
        accel_y = self.gravity * np.sin(self.plate_angle_y)
        
        # Simple friction damping
        friction_damping = 0.95
        self.ball_vel_x = self.ball_vel_x * friction_damping + accel_x * self.dt
        self.ball_vel_y = self.ball_vel_y * friction_damping + accel_y * self.dt
        
        # Update position
        self.ball_pos_x += self.ball_vel_x * self.dt
        self.ball_pos_y += self.ball_vel_y * self.dt
        
        # Bounce off plate edges (simple elastic collision)
        if abs(self.ball_pos_x) > 1.0:
            self.ball_pos_x = np.clip(self.ball_pos_x, -1.0, 1.0)
            self.ball_vel_x *= -0.8  # Energy loss on bounce
        
        if abs(self.ball_pos_y) > 1.0:
            self.ball_pos_y = np.clip(self.ball_pos_y, -1.0, 1.0)
            self.ball_vel_y *= -0.8
    
    def _compute_reward(self):
        """
        Compute the reward for the current step.
        
        This is a PLACEHOLDER showing common reward structures.
        You should customize this based on your control policy:
        
        Examples:
            - Reward ball to stay at center: -||ball_pos||²
            - Reward specific ball speed: -||ball_vel - target_vel||²
            - Reward combination: balance position + speed goals
            - Penalty for extreme plate angles
            - Penalty for ball falling off
        
        Returns:
            float: Reward value for this step
        """
        
        # Example 1: Simple penalty for ball position away from center
        position_penalty = -0.1 * (self.ball_pos_x**2 + self.ball_pos_y**2)
        
        # Example 2: Penalty for excessive plate tilting (energy cost)
        energy_cost = -0.01 * (self.plate_angle_x**2 + self.plate_angle_y**2)
        
        # Example 3: Penalty if ball falls off
        if not self._is_ball_on_plate():
            off_plate_penalty = -10.0
        else:
            off_plate_penalty = 0.0
        
        # Combine rewards (customize as needed)
        reward = position_penalty + energy_cost + off_plate_penalty
        
        return float(reward)
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (unused currently)
        
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Reset plate
        self.plate_angle_x = 0.0
        self.plate_angle_y = 0.0
        self.plate_angular_vel_x = 0.0
        self.plate_angular_vel_y = 0.0
        
        # Reset ball with small random perturbation for diversity
        self.ball_pos_x = self.rng.uniform(-0.2, 0.2)
        self.ball_pos_y = self.rng.uniform(-0.2, 0.2)
        self.ball_vel_x = self.rng.uniform(-0.5, 0.5)
        self.ball_vel_y = self.rng.uniform(-0.5, 0.5)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Execute one timestep of the environment.
        
        Args:
            action: np.ndarray of shape (2,), values in [-1, 1]
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation: Current state observation
                - reward: Reward for this step
                - terminated: Episode ended due to failure condition
                - truncated: Episode ended due to max steps
                - info: Additional information
        """
        self.current_step += 1
        
        # Validate action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Update plate based on motor commands
        self._update_plate_physics(action)
        
        # Update ball position and velocity
        self._update_ball_physics()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check termination conditions
        terminated = False
        
        # Episode terminates if ball falls off plate
        if not self._is_ball_on_plate():
            terminated = True
        
        # Episode truncates if max steps reached
        truncated = self.current_step >= self.max_episode_steps
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment.
        
        Currently only supports debug ANSI rendering.
        For full visualization, implement matplotlib or pygame rendering.
        """
        if self.render_mode is None:
            return None
        
        if self.render_mode == "human":
            # Print state to terminal for debugging
            print(f"\n--- Step {self.current_step} ---")
            print(f"Plate angles: ({self.plate_angle_x:.3f}, {self.plate_angle_y:.3f}) rad")
            print(f"Ball position: ({self.ball_pos_x:.3f}, {self.ball_pos_y:.3f})")
            print(f"Ball velocity: ({self.ball_vel_x:.3f}, {self.ball_vel_y:.3f}) m/s")
            print(f"Ball on plate: {self._is_ball_on_plate()}")
        
        elif self.render_mode == "rgb_array":
            # Return numpy array for rendering (not implemented yet)
            # You could implement matplotlib-based rendering here
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up resources."""
        pass


# ============================================================================
# Example usage and testing
# ============================================================================

if __name__ == "__main__":
    # Create environment
    env = TiltingPlateBallEnv(render_mode="human", max_episode_steps=200)
    
    # Test reset
    obs, info = env.reset(seed=42)
    print("Initial observation shape:", obs.shape)
    print("Initial observation:", obs)
    
    # Run a few random steps
    print("\nRunning 5 random steps:")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
        env.render()
        
        if terminated or truncated:
            print("Episode ended")
            break
    
    env.close()
    print("\nEnvironment closed successfully")
