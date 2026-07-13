import time
import types

import numpy as np
import zmq

from .vehicle import VehicleInput, VehicleState, INPUT_SIZE

ZMQ_PREFIX: str = "ipc:///tmp/drivebrain_sim_"
SEND_SOCKET_PORT: int = 6767
RECV_SOCKET_PORT: int = 5940
LIDAR_SOCKET_PORT: int = 1155

INPUT_TIMEOUT: float = 1.0 # secs

POINT_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1'),
])
POINT_STRIDE: int = POINT_DTYPE.itemsize  # 16

class SimComms:
    def __init__(self, proto: types.ModuleType) -> None:
        self._proto = proto
        self._ctx: zmq.Context = zmq.Context()

        self._cmd: zmq.Socket = self._ctx.socket(zmq.PULL)
        self._cmd.bind(f"{ZMQ_PREFIX}{RECV_SOCKET_PORT}")
        self._cmd.setsockopt(zmq.RCVHWM, 1)

        self._state: zmq.Socket = self._ctx.socket(zmq.PUSH)
        self._state.bind(f"{ZMQ_PREFIX}{SEND_SOCKET_PORT}")
        self._state.setsockopt(zmq.SNDHWM, 1)

        self._lidar: zmq.Socket = self._ctx.socket(zmq.PUSH)

        self._lidar.bind(f"{ZMQ_PREFIX}{LIDAR_SOCKET_PORT}")
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

    def send_pointcloud(self, pts: np.ndarray, rgba: np.ndarray) -> None:
        n: int = len(pts)
        buf = np.empty(n, dtype=POINT_DTYPE)
        buf['x'] = pts[:, 0]
        buf['y'] = pts[:, 1]
        buf['z'] = pts[:, 2]
        buf['r'] = rgba[:, 0]
        buf['g'] = rgba[:, 1]
        buf['b'] = rgba[:, 2]
        buf['a'] = rgba[:, 3]

        PEF = self._proto.base_msgs_pb2.PackedElementField
        msg = self._proto.dv_msgs_pb2.PointCloud(
            frame_id="lidar",
            point_stride=POINT_STRIDE,
            fields=[PEF(name="x", offset=0,  type=PEF.FLOAT32),
                    PEF(name="y", offset=4,  type=PEF.FLOAT32),
                    PEF(name="z", offset=8,  type=PEF.FLOAT32),
                    PEF(name="red", offset=12, type=PEF.UINT8),
                    PEF(name="green", offset=13, type=PEF.UINT8),
                    PEF(name="blue", offset=14, type=PEF.UINT8),
                    PEF(name="alpha", offset=15, type=PEF.UINT8),
            ],
            data=buf.tobytes(),
        )
        try:
            self._lidar.send(msg.SerializeToString(), zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self) -> None:
        self._cmd.close()
        self._state.close()
        self._lidar.close()
        self._ctx.term()
