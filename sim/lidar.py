
import struct
import time

import numpy as np
import mujoco
import zmq

LIDAR_ADDR  = "ipc:///tmp/hytech_sim_lidar"
_HEADER_FMT = "qI4x"  # int64 + uint32 + 4-byte trailing pad

N_AZIM = 512
N_ELEV = 32
ELEV_MIN = -0.261799 # -15 deg
ELEV_MAX =  0.261799 # +15 deg
MAX_RANGE = 30.0


class LidarSensor:
    def __init__(self, world_model, site_name: str = "lidar_site"):
        self._model = world_model.model
        self._data  = world_model.data   # shared reference; updated by WorldModel.set_car_pose

        self._site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if self._site_id < 0:
            raise ValueError(f"site '{site_name}' not found in MuJoCo model")

        self._body_exclude = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "car"
        )

        # Pre-compute ray directions in sensor LOCAL frame (constant across steps).
        # Convention: x=forward, y=left, z=up (standard sensor frame).
        azim = np.linspace(-np.pi, np.pi, N_AZIM, endpoint=False)
        elev = np.linspace(ELEV_MIN, ELEV_MAX, N_ELEV)
        e_grid, a_grid = np.meshgrid(elev, azim, indexing="ij")  # (32, 512)
        vx = np.cos(e_grid) * np.cos(a_grid)
        vy = np.cos(e_grid) * np.sin(a_grid)
        vz = np.sin(e_grid)
        self._rays_local = np.stack([vx.ravel(), vy.ravel(), vz.ravel()], axis=1)  # (N,3)
        self._n_rays = N_AZIM * N_ELEV

        self._geomid = np.full(self._n_rays, -1,  dtype=np.int32)
        self._dist = np.full(self._n_rays, -1.0, dtype=np.float64)
        self._vecs_buf = np.empty((self._n_rays, 3), dtype=np.float64)

        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.PUSH)
        self._sock.bind(LIDAR_ADDR)
        self._sock.setsockopt(zmq.SNDHWM, 1)

    def scan(self) -> np.ndarray:
        """
        Trace all rays. Returns hit points in sensor frame, shape (N, 3).
        Only rays that actually hit something are included.
        """
        pnt = self._data.site_xpos[self._site_id].copy()
        R = self._data.site_xmat[self._site_id].reshape(3, 3)

        np.dot(self._rays_local, R.T, out=self._vecs_buf)
        vecs_flat = self._vecs_buf.ravel()

        self._geomid[:] = -1
        self._dist[:] = -1.0

        mujoco.mj_multiRay(
            self._model, self._data,
            pnt, vecs_flat,
            None, True, self._body_exclude,
            self._geomid, self._dist, None,
            self._n_rays, MAX_RANGE,
        )

        hit = self._dist > 0
        if not np.any(hit):
            return np.empty((0, 3), dtype=np.float32)

        # World-frame hit points → back to sensor local frame
        pts_world = pnt + self._vecs_buf[hit] * self._dist[hit, None]
        pts_local = ((pts_world - pnt) @ R).astype(np.float32)
        return pts_local

    def scan_and_publish(self) -> np.ndarray:
        pts = self.scan()
        header = struct.pack(_HEADER_FMT, time.time_ns(), len(pts))
        try:
            self._sock.send(header + pts.tobytes(), zmq.NOBLOCK)
        except zmq.Again:
            pass
        return pts
