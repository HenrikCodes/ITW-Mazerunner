from pathlib import Path
import mujoco
import mujoco.viewer
from mujoco import MjData, MjModel


HERE = Path(__file__).resolve().parent              # .../Simulation
xml_path = HERE / "ball_tilt_plate.xml"             # or "plate2.xml"

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Keyboard control for tilting (e.g., arrow keys or remap)
        # Rapid tilt: hold keys for acceleration
        action = [0.0, 0.0]  # Replace with viewer.get_key() logic if needed
        mujoco.mj_step(model, data)
        viewer.sync()
