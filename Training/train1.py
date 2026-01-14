from stable_baselines3 import PPO
from tilting_plate_env import TiltingPlateBallEnv

env = TiltingPlateBallEnv(render_mode=None)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    verbose=1,
    tensorboard_log="./mujoco_training/"
)

model.learn(total_timesteps=500000)
model.save("tilting_plate_mujoco_ppo")
