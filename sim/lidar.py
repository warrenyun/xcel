"""
Simulated Ouster OS1 128 Rev8 RGB LiDAR.
Datasheet: https://data.ouster.io/downloads/datasheets/datasheet-rev8-v4p0-os1.pdf
"""
import numpy as np
import mujoco

from .world import World

N_AZIM: int = 1024
N_ELEV: int = 128
ELEV_MIN: float = np.deg2rad(-22.0)
ELEV_MAX: float = np.deg2rad(22.0)
MAX_RANGE: float = 90.0

class LidarSensor:
    def __init__(self, world: World, site_name: str = "lidar_site") -> None:
        self._model: mujoco.MjModel = world.model
        self._data: mujoco.MjData = world.data

        self._site_id: int = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if self._site_id < 0:
            raise ValueError(f"site '{site_name}' not found in MuJoCo model")

        self._body_exclude: int = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "car"
        )

        azim: np.ndarray = np.linspace(-np.pi, np.pi, N_AZIM, endpoint=False)
        elev: np.ndarray = np.linspace(ELEV_MIN, ELEV_MAX, N_ELEV)
        e_grid, a_grid = np.meshgrid(elev, azim, indexing="ij")

        vx: np.ndarray = np.cos(e_grid) * np.cos(a_grid)
        vy: np.ndarray = np.cos(e_grid) * np.sin(a_grid)
        vz: np.ndarray = np.sin(e_grid)

        self._rays_local: np.ndarray = np.stack([vx.ravel(), vy.ravel(), vz.ravel()], axis=1)
        self._n_rays: int = N_AZIM * N_ELEV

        self._geomid: np.ndarray = np.full(self._n_rays, -1, dtype=np.int32)
        self._dist: np.ndarray = np.full(self._n_rays, -1.0, dtype=np.float64)
        self._vecs_buf: np.ndarray = np.empty((self._n_rays, 3), dtype=np.float64)

    def _apply_noise(self, distances: np.ndarray, hit: np.ndarray) -> None:
        """Apply range-dependent Gaussian noise to point cloud"""
        # Range percision graph from OS1 datasheet
        RANGE_POINTS = [0,     5,   10,  20,  30,   40,   50,   60,  70,  80,   90] # m
        SIGMA_POINTS = [0.25, 0.25, 0.3, 0.4, 0.49, 0.55, 0.65, 0.7, 1.0, 1.25, 1.5]  # cm

        r = distances[hit]
        sigma = np.interp(r, RANGE_POINTS, SIGMA_POINTS)
        distances[hit] = r + np.random.normal(0.0, sigma / 100.0)

    def scan(self) -> tuple[np.ndarray, np.ndarray]:
        """Trace all rays. Returns (pts_local (N,3) float32, rgba (N,4) uint8)."""
        pnt: np.ndarray = self._data.site_xpos[self._site_id].copy()
        R: np.ndarray = self._data.site_xmat[self._site_id].reshape(3, 3)

        np.dot(self._rays_local, R.T, out=self._vecs_buf)
        vecs_flat: np.ndarray = self._vecs_buf.ravel()

        self._geomid[:] = -1
        self._dist[:] = -1.0

        mujoco.mj_multiRay(
            self._model, self._data,
            pnt, vecs_flat,
            None, True, self._body_exclude,
            self._geomid, self._dist, None,
            self._n_rays, MAX_RANGE
        )

        hit: np.ndarray = self._dist > 0
        if not np.any(hit):
            empty = np.empty((0, 3), dtype=np.float32)
            return empty, np.empty((0, 4), dtype=np.uint8)

        self._apply_noise(self._dist, hit)

        rgba_f: np.ndarray = self._model.geom_rgba[self._geomid[hit]]
        rgba: np.ndarray = (np.clip(rgba_f, 0, 1) * 255).astype(np.uint8)

        pts_world: np.ndarray = pnt + self._vecs_buf[hit] * self._dist[hit, None]
        pts_local: np.ndarray = ((pts_world - pnt) @ R).astype(np.float32)
        return pts_local, rgba

