import math
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from sim.world import World, CAR_Z

CAR_HALF_L  = 1.2
CAR_HALF_W  = 0.6
CAR_HALF_H  = 0.2
CONE_RADIUS = 0.114
CONE_HALF_H = 0.1625
LIDAR_FWD   = 0.75
LIDAR_UP    = 0.15


def blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="world"),
            rrb.Vertical(
                rrb.TimeSeriesView(name="torques [Nm]",  contents=["scalars/torque_*"]),
                rrb.TimeSeriesView(name="steering [rad]", contents=["scalars/steering"]),
                rrb.TimeSeriesView(name="speed [m/s]",   contents=["scalars/speed"]),
                rrb.TimeSeriesView(name="yaw rate [r/s]", contents=["scalars/yaw_rate"]),
                rrb.TimeSeriesView(name="slip [rad]",    contents=["scalars/slip"]),
            ),
            column_shares=[2, 1],
        )
    )


def setup_static(world: World) -> None:
    # Blue cones
    if world.cones_left:
        pts = [[c[0], c[1], CONE_HALF_H] for c in world.cones_left]
        rr.log("world/cones/blue", rr.Points3D(pts, radii=CONE_RADIUS,
               colors=[[30, 100, 255]] * len(pts)), static=True)

    # Yellow cones
    if world.cones_right:
        pts = [[c[0], c[1], CONE_HALF_H] for c in world.cones_right]
        rr.log("world/cones/yellow", rr.Points3D(pts, radii=CONE_RADIUS,
               colors=[[255, 210, 0]] * len(pts)), static=True)

    # Series line colours
    rr.log("scalars/torque_fl", rr.SeriesLines(colors=[[255, 80,  80]],  names=["FL"]), static=True)
    rr.log("scalars/torque_fr", rr.SeriesLines(colors=[[80,  255, 80]],  names=["FR"]), static=True)
    rr.log("scalars/torque_rl", rr.SeriesLines(colors=[[80,  80,  255]], names=["RL"]), static=True)
    rr.log("scalars/torque_rr", rr.SeriesLines(colors=[[255, 200, 0]],   names=["RR"]), static=True)
    rr.log("scalars/steering",  rr.SeriesLines(colors=[[200, 200, 200]], names=["steer"]), static=True)
    rr.log("scalars/speed",     rr.SeriesLines(colors=[[0,   220, 220]], names=["speed"]), static=True)
    rr.log("scalars/yaw_rate",  rr.SeriesLines(colors=[[220, 100, 220]], names=["yaw_rate"]), static=True)
    rr.log("scalars/slip",      rr.SeriesLines(colors=[[255, 140, 0]],   names=["slip"]), static=True)


def log_frame(
    *,
    sim_t: float,
    x: float, y: float, psi: float,
    pts_local: np.ndarray,
    speed: float,
    yaw_rate: float,
    slip: float,
    torque_fl: float, torque_fr: float,
    torque_rl: float, torque_rr: float,
    steering: float,
    cmd_timed_out: bool,
) -> None:
    rr.set_time("sim_time", duration=sim_t)

    # Car box
    hpsi = psi * 0.5
    rr.log("world/car", rr.Boxes3D(
        centers=[[x, y, CAR_Z]],
        half_sizes=[[CAR_HALF_L, CAR_HALF_W, CAR_HALF_H]],
        quaternions=[rr.Quaternion(xyzw=[0.0, 0.0, math.sin(hpsi), math.cos(hpsi)])],
        colors=[[220, 50, 50] if not cmd_timed_out else [100, 100, 100]],
    ))

    # Point cloud: local → world
    if pts_local.shape[0] > 0:
        cp, sp = math.cos(psi), math.sin(psi)
        R_z = np.array([[cp, -sp, 0.0],
                        [sp,  cp, 0.0],
                        [0.0, 0.0, 1.0]], dtype=np.float32)
        sensor_pos = np.array([
            x + LIDAR_FWD * cp,
            y + LIDAR_FWD * sp,
            CAR_Z + LIDAR_UP,
        ], dtype=np.float32)
        pts_world = pts_local @ R_z.T + sensor_pos
        rr.log("world/lidar", rr.Points3D(pts_world, radii=0.05,
               colors=[[0, 255, 120]] * len(pts_world)))

    # Scalars
    rr.log("scalars/torque_fl", rr.Scalars(torque_fl))
    rr.log("scalars/torque_fr", rr.Scalars(torque_fr))
    rr.log("scalars/torque_rl", rr.Scalars(torque_rl))
    rr.log("scalars/torque_rr", rr.Scalars(torque_rr))
    rr.log("scalars/steering",  rr.Scalars(steering))
    rr.log("scalars/speed",     rr.Scalars(speed))
    rr.log("scalars/yaw_rate",  rr.Scalars(yaw_rate))
    rr.log("scalars/slip",      rr.Scalars(slip))
