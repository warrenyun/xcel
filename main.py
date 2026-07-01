"""
JAX vehicle sim.

ZMQ interface:
  PULL  ipc:///tmp/hytech_sim_cmd    — VehicleInput 
  PUSH  ipc:///tmp/hytech_sim_state  — VehicleState 
  PUSH  ipc:///tmp/hytech_sim_lidar  — LidarScan 
"""

import argparse
import math
import signal
import time

import numpy as np
import rerun as rr

import mujoco
import mujoco.viewer
from sim.vehicle.dynamics import (
    VehicleParams, make_state, make_input, steady_state_wheels, step_rk4,
    X, Y, PSI, VX, VY, OMEGA, slip_angle,
)
from sim.vehicle.state import jax_to_vehicle_state
from sim.comms import SimComms, CMD_ADDR, STATE_ADDR
from sim.world import World
from sim.lidar import LidarSensor, N_AZIM, N_ELEV
import visualize

VIEWER_HZ = 60.0
LIDAR_HZ = 20.0

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--track",    default="FSG", choices=["FSG", "FSI", "skidpad", "acceleration", "thin"])
    p.add_argument("--dt",       type=float, default=0.002, help="Sim timestep [s]")
    p.add_argument("--no-viz",    action="store_true",       help="Disable Rerun time-series")
    p.add_argument("--headless",  action="store_true",       help="Disable MuJoCo viewer")
    return p.parse_args()

def main():
    args = parse_args()
    params = VehicleParams()
    dt = args.dt

    vis_every = max(1, round(1.0 / (VIEWER_HZ * dt)))
    lidar_every = max(1, round(1.0 / (LIDAR_HZ * dt)))

    # JIT warm-up
    s0 = steady_state_wheels(0.0, params)
    u0 = make_input()
    step_rk4(s0, u0, params, dt).block_until_ready()

    world = World(args.track)
    lidar = LidarSensor(world)
    print(f"World: {args.track} track loaded  "
          f"({len(world.cones_left)} blue + {len(world.cones_right)} yellow cones)")
    print(f"LiDAR: {N_AZIM}×{N_ELEV} rays @ {LIDAR_HZ:.0f} Hz, 30m range")

    # Initial JAX state
    x0, y0, psi0 = [float(v) for v in world.start_pose]
    state = steady_state_wheels(0, params)
    state = state.at[X].set(x0).at[Y].set(y0).at[PSI].set(psi0)
    world.set_car_pose(x0, y0, psi0)

    # rerun
    if not args.no_viz:
        rr.init("vehicle_sim", spawn=True)
        visualize.setup_static(world)
        rr.send_blueprint(visualize.blueprint())

    # MuJoCo passive viewer
    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(world.model, world.data)
        viewer.cam.lookat[:] = [x0, y0, 0.5]
        viewer.cam.distance  = 20.0
        viewer.cam.elevation = -25.0

    # zmq
    comms = SimComms()

    running = True
    def stop(_sig, _frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT,  stop)
    signal.signal(signal.SIGTERM, stop)

    sim_t = 0.0
    step_i = 0
    wall_start = time.perf_counter()
    last_pts = np.empty((0, 3), dtype=np.float32)

    while running and (viewer is None or viewer.is_running()):
        comms.drain_commands()
        cmd = comms.current_input()

        jax_u = make_input(
            tau_fl=cmd.torque_fl, tau_fr=cmd.torque_fr,
            tau_rl=cmd.torque_rl, tau_rr=cmd.torque_rr,
            steer=cmd.wheel_steer_rad,
        )

        state = step_rk4(state, jax_u, params, dt)
        sim_t += dt
        step_i += 1

        x_i = float(state[X])
        y_i = float(state[Y])
        psi_i = float(state[PSI])

        world.set_car_pose(x_i, y_i, psi_i)
        comms.send_state(jax_to_vehicle_state(state, jax_u, params))

        if step_i % lidar_every == 0:
            last_pts = lidar.scan_and_publish()

        if step_i % vis_every == 0:
            if viewer is not None:
                viewer.sync()
            if not args.no_viz:
                visualize.log_frame(
                    sim_t=sim_t,
                    x=x_i, y=y_i, psi=psi_i,
                    pts_local=last_pts,
                    speed=math.hypot(float(state[VX]), float(state[VY])),
                    yaw_rate=float(state[OMEGA]),
                    slip=float(slip_angle(state)),
                    torque_fl=cmd.torque_fl, torque_fr=cmd.torque_fr,
                    torque_rl=cmd.torque_rl, torque_rr=cmd.torque_rr,
                    steering=cmd.wheel_steer_rad,
                    cmd_timed_out=comms.is_timed_out(),
                )

        drift = (wall_start + sim_t) - time.perf_counter()
        if drift > 0:
            time.sleep(drift)

    comms.close()
    print(f"\nStopped at t={sim_t:.2f}s  ({step_i} steps)")


if __name__ == "__main__":
    main()
