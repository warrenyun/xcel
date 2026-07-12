import struct
import time

import numpy as np
import zmq

from .vehicle import VehicleInput, VehicleState, INPUT_SIZE


CMD_ADDR: str = "ipc:///tmp/hytech_sim_cmd"
STATE_ADDR: str = "ipc:///tmp/hytech_sim_state"
LIDAR_ADDR: str = "ipc:///tmp/hytech_sim_lidar"
_LIDAR_HEADER_FMT: str = "qI4x"

INPUT_TIMEOUT: float = 1.0 # seconds

class SimComms:
    def __init__(self) -> None:
        self._ctx: zmq.Context = zmq.Context()

        self._cmd: zmq.Socket = self._ctx.socket(zmq.PULL)
        self._cmd.bind(CMD_ADDR)
        self._cmd.setsockopt(zmq.RCVHWM, 1)

        self._state: zmq.Socket = self._ctx.socket(zmq.PUSH)
        self._state.bind(STATE_ADDR)
        self._state.setsockopt(zmq.SNDHWM, 1)

        self._lidar: zmq.Socket = self._ctx.socket(zmq.PUSH)
        self._lidar.bind(LIDAR_ADDR)
        self._lidar.setsockopt(zmq.SNDHWM, 1)

        self._latest_input: VehicleInput = VehicleInput()
        self._last_cmd_time: float = time.monotonic()

    def drain_commands(self) -> None:
        while True:
            try:
                data: bytes = self._cmd.recv(zmq.NOBLOCK)
                if len(data) == INPUT_SIZE:
                    self._latest_input = VehicleInput.unpack(data)
                    self._last_cmd_time = time.monotonic()
            except zmq.Again:
                break

    def timed_out(self) -> bool:
        return time.monotonic() - self._last_cmd_time > INPUT_TIMEOUT

    def current_input(self) -> VehicleInput:
        """Return the latest input, zeroing torques if the controller timed out."""
        if self.timed_out():
            return VehicleInput(wheel_steer_rad=self._latest_input.wheel_steer_rad)
        return self._latest_input

    def send_state(self, state: VehicleState) -> None:
        try:
            self._state.send(state.pack(), zmq.NOBLOCK)
        except zmq.Again:
            pass

    def send_pointcloud(self, pts: np.ndarray) -> None:
        header: bytes = struct.pack(_LIDAR_HEADER_FMT, time.time_ns(), len(pts))
        try:
            self._lidar.send(header + pts.tobytes(), zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self) -> None:
        self._cmd.close()
        self._state.close()
        self._lidar.close()
        self._ctx.term()
