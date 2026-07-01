"""
Wire-format dataclasses for ZMQ serialization (world-frame only).
"""

import struct
import math
from dataclasses import dataclass

from .dynamics import VehicleParams, dynamics, X, Y, PSI, VX, VY, OMEGA

_INPUT_FMT = "5d"
_STATE_FMT = "9d"

INPUT_SIZE = struct.calcsize(_INPUT_FMT)   # 40
STATE_SIZE = struct.calcsize(_STATE_FMT)   # 72


@dataclass
class VehicleInput:
    torque_fl: float = 0.0
    torque_fr: float = 0.0
    torque_rl: float = 0.0
    torque_rr: float = 0.0
    wheel_steer_rad: float = 0.0

    def pack(self) -> bytes:
        return struct.pack(
            _INPUT_FMT,
            self.torque_fl, self.torque_fr,
            self.torque_rl, self.torque_rr,
            self.wheel_steer_rad,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "VehicleInput":
        return cls(*struct.unpack(_INPUT_FMT, data))


@dataclass
class VehicleState:
    ax_world:       float = 0.0
    ay_world:       float = 0.0
    psi_ddot_world: float = 0.0
    vx_world:       float = 0.0
    vy_world:       float = 0.0
    psi_dot_world:  float = 0.0
    x_world:        float = 0.0
    y_world:        float = 0.0
    psi_world:      float = 0.0

    def pack(self) -> bytes:
        return struct.pack(
            _STATE_FMT,
            self.ax_world,  self.ay_world,  self.psi_ddot_world,
            self.vx_world,  self.vy_world,  self.psi_dot_world,
            self.x_world,   self.y_world,   self.psi_world,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "VehicleState":
        return cls(*struct.unpack(_STATE_FMT, data))


def jax_to_vehicle_state(jax_state, jax_input, params: VehicleParams) -> VehicleState:
    """Derive VehicleState (world-frame only) from the JAX state vector."""
    deriv = dynamics(jax_state, jax_input, params)

    return VehicleState(
        ax_world=float(deriv[3]),
        ay_world=float(deriv[4]),
        psi_ddot_world=float(deriv[5]),
        vx_world=float(jax_state[VX]),
        vy_world=float(jax_state[VY]),
        psi_dot_world=float(jax_state[OMEGA]),
        x_world=float(jax_state[X]),
        y_world=float(jax_state[Y]),
        psi_world=float(jax_state[PSI]),
    )
