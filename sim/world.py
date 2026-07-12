import math
from pathlib import Path
from typing import Any

import yaml
import mujoco

TRACKS_DIR: Path = Path(__file__).parent.parent / "gazebo/models/track/tracks_yaml"
ASSETS_DIR: Path = Path(__file__).parent.parent / "assets"
CAR_Z: float = 0.4

class World:
    """MuJoCo model. Car body is kinematically driven by JAX state."""

    model: mujoco.MjModel
    data: mujoco.MjData
    start_pose: list[float]
    cones_left: list[list[float]]
    cones_right: list[list[float]]

    def __init__(self, track: str) -> None:
        scene_path: Path = ASSETS_DIR / "tracks" / f"{track}_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        with open(TRACKS_DIR / f"{track}.yaml") as f:
            d: dict[str, Any] = yaml.safe_load(f)
        self.start_pose = d.get("starting_pose_front_wing", [0.0, 0.0, 0.0])
        self.cones_left = d["cones_left"]
        self.cones_right = d["cones_right"]

        self._mocap_idx: int = 0
        mujoco.mj_forward(self.model, self.data)

    def set_car_pose(self, x: float, y: float, psi: float) -> None:
        hpsi: float = psi * 0.5
        self.data.mocap_pos[self._mocap_idx] = [x, y, CAR_Z]
        self.data.mocap_quat[self._mocap_idx] = [math.cos(hpsi), 0.0, 0.0, math.sin(hpsi)]
        mujoco.mj_forward(self.model, self.data)
