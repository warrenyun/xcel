
import math
import yaml
import mujoco
from pathlib import Path

TRACKS_DIR = Path(__file__).parent.parent / "gazebo/models/track/tracks_yaml"
ASSETS_DIR = Path(__file__).parent.parent / "assets"

CAR_Z = 0.4

class World:
    """MuJoCo model. Car body is kinematically driven by JAX state."""

    def __init__(self, track: str):
        scene_path = ASSETS_DIR / "tracks" / f"{track}_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data  = mujoco.MjData(self.model)

        with open(TRACKS_DIR / f"{track}.yaml") as f:
            d = yaml.safe_load(f)
        self.start_pose  = d.get("starting_pose_front_wing", [0.0, 0.0, 0.0])
        self.cones_left  = d["cones_left"]
        self.cones_right = d["cones_right"]

        self._mocap_idx = 0
        mujoco.mj_forward(self.model, self.data)

    def set_car_pose(self, x: float, y: float, psi: float) -> None:
        hpsi = psi * 0.5
        self.data.mocap_pos[self._mocap_idx]  = [x, y, CAR_Z]
        self.data.mocap_quat[self._mocap_idx] = [math.cos(hpsi), 0.0, 0.0, math.sin(hpsi)]
        mujoco.mj_forward(self.model, self.data)
