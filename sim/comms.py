"""
zmq command interface. exposes channels for vehicle state and commanding controls
"""

import time
import zmq

from .vehicle import VehicleInput, VehicleState, INPUT_SIZE

CMD_ADDR = "ipc:///tmp/hytech_sim_cmd"
STATE_ADDR = "ipc:///tmp/hytech_sim_state"

INPUT_TIMEOUT_S = 1.0 # max timeout before zero wheel torques are commanded

class SimComms:
    def __init__(self):
        self._ctx = zmq.Context()

        self._cmd = self._ctx.socket(zmq.PULL)
        self._cmd.bind(CMD_ADDR)
        self._cmd.setsockopt(zmq.RCVHWM, 1)

        # state should explicitly only be used for debug logging. in a simulated environment we should assume that we have no idea of the true vehicle state
        self._state = self._ctx.socket(zmq.PUSH)
        self._state.bind(STATE_ADDR)
        self._state.setsockopt(zmq.SNDHWM, 1) 

        self._latest_input = VehicleInput()
        self._last_cmd_time = time.monotonic()

    def drain_commands(self) -> None:
        while True:
            try:
                data = self._cmd.recv(zmq.NOBLOCK)
                if len(data) == INPUT_SIZE:
                    self._latest_input = VehicleInput.unpack(data)
                    self._last_cmd_time = time.monotonic()
            except zmq.Again:
                break

    def is_timed_out(self) -> bool:
        return time.monotonic() - self._last_cmd_time > INPUT_TIMEOUT_S

    def current_input(self) -> VehicleInput:
        """Return the latest input, zeroing torques if the controller timed out."""
        if self.is_timed_out():
            return VehicleInput(wheel_steer_rad=self._latest_input.wheel_steer_rad)
        return self._latest_input

    def send_state(self, state: VehicleState) -> None:
        try:
            self._state.send(state.pack(), zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self) -> None:
        self._cmd.close()
        self._state.close()
        self._ctx.term()
