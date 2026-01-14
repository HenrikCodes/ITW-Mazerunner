import mujoco
import mujoco.viewer
from mujoco import MjData, MjModel

model = mujoco.MjModel.from_xml_path("ball_tilt_plate.xml")
data = MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Keyboard control for tilting (e.g., arrow keys or remap)
        # Rapid tilt: hold keys for acceleration
        action = [0.0, 0.0]  # Replace with viewer.get_key() logic if needed
        mujoco.mj_step(model, data)
        viewer.sync()
