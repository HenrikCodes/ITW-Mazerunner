import numpy as np
from gymnasium.utils.play import play
from Simulation.tilting_plate_env import TiltingPlateBallEnv

env = TiltingPlateBallEnv(render_mode="human")

STEP = 0.3  # action magnitude in [-1, 1]

# Keys are tuples; use gymnasium docs pattern.
keys_to_action = {
    ("left",):  np.array([-STEP, 0.0], dtype=np.float32),
    ("right",): np.array([ STEP, 0.0], dtype=np.float32),
    ("down",):  np.array([0.0, -STEP], dtype=np.float32),
    ("up",):    np.array([0.0,  STEP], dtype=np.float32),

    # Optional diagonals (two keys at once)
    ("left", "up"):    np.array([-STEP,  STEP], dtype=np.float32),
    ("right", "up"):   np.array([ STEP,  STEP], dtype=np.float32),
    ("left", "down"):  np.array([-STEP, -STEP], dtype=np.float32),
    ("right", "down"): np.array([ STEP, -STEP], dtype=np.float32),

    # No key = do nothing (depends on helper; you can also handle this in env)
}

play(env, keys_to_action=keys_to_action)
