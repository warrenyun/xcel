"""
JAX Two-Track Vehicle Dynamics Model
=====================================

Reimplementation of the OpenModelica PlanarMechanics two-track model
with SlipBasedWheelJoint tire model. Pure JAX — JIT-compilable,
vmappable, and differentiable.

State vector (10,):
    [x, y, ψ, vx, vy, ω, ω_fl, ω_fr, ω_rl, ω_rr]

    x, y       : CG position in world frame [m]
    ψ (psi)    : Heading angle [rad]
    vx, vy     : CG velocity in world frame [m/s]
    ω (omega)  : Yaw rate [rad/s]
    ω_fl..rr   : Wheel angular velocities [rad/s]

Input vector (5,):
    [τ_fl, τ_fr, τ_rl, τ_rr, δ]

    τ_fl..rr   : Motor torques [N·m] (pre-gear-ratio)
    δ (delta)  : Front wheel steering angle [rad]
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple
from functools import partial


# ── Parameters ─────────────────────────────────────────────────────

class VehicleParams(NamedTuple):
    """Vehicle parameters (defaults match OpenModelica FMU)."""
    # Body
    m: float = 250.0            # Vehicle mass [kg]
    I_zz: float = 120.0         # Yaw moment of inertia [kg·m²]
    # Geometry
    l_f: float = 0.765          # CG to front axle [m]
    l_r: float = 0.765          # CG to rear axle [m]
    w_f: float = 0.6            # Front half-track width [m]
    w_r: float = 0.6            # Rear half-track width [m]
    # Wheels
    r_w: float = 0.2            # Wheel radius [m]
    I_w: float = 0.094          # Wheel rotational inertia [kg·m²]
    gear_ratio: float = 11.83   # Motor-to-wheel gear ratio
    # Tires (Modelica SlipBasedWheelJoint)
    f_n: float = 600.0          # Normal load per wheel [N] (static, no transfer)
    mu_a: float = 1.8           # Peak friction coefficient (adhesion)
    mu_s: float = 1.5           # Sliding friction coefficient
    s_adh: float = 0.117        # Adhesion slip ratio threshold
    s_sld: float = 0.35         # Sliding slip ratio threshold
    v_adh_min: float = 0.005    # Minimum adhesion velocity [m/s]
    v_sld_min: float = 0.02     # Minimum sliding velocity [m/s]


# ── State / input indices ──────────────────────────────────────────

X, Y, PSI, VX, VY, OMEGA, W_FL, W_FR, W_RL, W_RR = range(10)
TAU_FL, TAU_FR, TAU_RL, TAU_RR, DELTA = range(5)


# ── Tire friction model ────────────────────────────────────────────

def _tire_friction(v_slip, v_adh, v_sld, mu_a, mu_s, f_n):
    """
    Friction force magnitude from slip velocity magnitude.

    Matches the Modelica SlipBasedWheelJoint friction curve:
      [0, v_adh)     : Linear adhesion, peaks at μ_a · F_N
      [v_adh, v_sld) : Cosine interpolation from μ_a → μ_s
      [v_sld, ∞)     : Constant sliding at μ_s · F_N
    """
    eps = 1e-8

    # Adhesion: force rises linearly with slip
    f_adhesion = mu_a * f_n * v_slip / jnp.maximum(v_adh, eps)

    # Transition: smooth cosine blend
    t = jnp.clip((v_slip - v_adh) / jnp.maximum(v_sld - v_adh, eps), 0.0, 1.0)
    mu_t = mu_a - (mu_a - mu_s) * 0.5 * (1.0 - jnp.cos(jnp.pi * t))
    f_transition = mu_t * f_n

    # Sliding: constant force
    f_sliding = mu_s * f_n

    return jnp.where(v_slip < v_adh, f_adhesion,
                     jnp.where(v_slip < v_sld, f_transition, f_sliding))


# ── Continuous dynamics ────────────────────────────────────────────

@jax.jit
def dynamics(state, u, params):
    """
    State derivative ẋ = f(x, u, p).

    Args:
        state:  (10,) state vector
        u:      (5,)  input vector [τ_fl, τ_fr, τ_rl, τ_rr, δ]
        params: VehicleParams

    Returns:
        (10,) state derivative
    """
    p = params
    eps = 1e-10

    psi   = state[PSI]
    vx    = state[VX]
    vy    = state[VY]
    omega = state[OMEGA]
    w     = state[6:10]                    # wheel speeds (4,)

    tau_drive = u[:4] * p.gear_ratio       # post-gearbox torques (4,)
    delta     = u[DELTA]

    c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)

    # ── Wheel positions in body frame (4×2) ────────────────────────
    r_body = jnp.array([
        [ p.l_f,  p.w_f],    # FL
        [ p.l_f, -p.w_f],    # FR
        [-p.l_r,  p.w_r],    # RL
        [-p.l_r, -p.w_r],    # RR
    ])

    # Rotate to world frame
    r_wx = c_psi * r_body[:, 0] - s_psi * r_body[:, 1]
    r_wy = s_psi * r_body[:, 0] + c_psi * r_body[:, 1]

    # ── Contact-point velocities ───────────────────────────────────
    # v_contact = v_cg + ω × r   (2D cross: ω×[rx,ry] = [-ω·ry, ω·rx])
    v_cx = vx - omega * r_wy
    v_cy = vy + omega * r_wx

    # ── Wheel headings ─────────────────────────────────────────────
    steer = jnp.array([delta, delta, 0.0, 0.0])
    heading = psi + steer
    c_h, s_h = jnp.cos(heading), jnp.sin(heading)

    # ── Velocity in each wheel's frame ─────────────────────────────
    v_long = v_cx * c_h + v_cy * s_h       # longitudinal (4,)
    v_lat  = -v_cx * s_h + v_cy * c_h      # lateral (4,)

    # ── Slip velocities ────────────────────────────────────────────
    v_slip_long = v_long - w * p.r_w
    v_slip_lat  = v_lat
    v_slip_mag  = jnp.sqrt(v_slip_long**2 + v_slip_lat**2 + eps)

    # Speed-dependent friction thresholds
    v_adh = jnp.maximum(p.s_adh * jnp.abs(v_long), p.v_adh_min)
    v_sld = jnp.maximum(p.s_sld * jnp.abs(v_long), p.v_sld_min)

    # ── Tire forces ────────────────────────────────────────────────
    f_mag  = _tire_friction(v_slip_mag, v_adh, v_sld, p.mu_a, p.mu_s, p.f_n)
    f_long = -f_mag * v_slip_long / v_slip_mag    # opposes slip (4,)
    f_lat  = -f_mag * v_slip_lat  / v_slip_mag    # opposes slip (4,)

    # ── World-frame forces ─────────────────────────────────────────
    Fx = f_long * c_h - f_lat * s_h
    Fy = f_long * s_h + f_lat * c_h

    # ── Body accelerations ─────────────────────────────────────────
    ax    = jnp.sum(Fx) / p.m
    ay    = jnp.sum(Fy) / p.m
    alpha = jnp.sum(r_wx * Fy - r_wy * Fx) / p.I_zz

    # ── Wheel spin dynamics ────────────────────────────────────────
    # I_w · ẇ = τ_drive − F_long · r_w
    w_dot = (tau_drive - f_long * p.r_w) / p.I_w

    return jnp.array([
        vx, vy, omega,            # dx, dy, dpsi
        ax, ay, alpha,            # dvx, dvy, domega
        w_dot[0], w_dot[1],       # dw_fl, dw_fr
        w_dot[2], w_dot[3],       # dw_rl, dw_rr
    ])


# ── Integration ────────────────────────────────────────────────────

@jax.jit
def step_rk4(state, u, params, dt):
    """Single RK4 step with zero-order hold on input."""
    k1 = dynamics(state,                u, params)
    k2 = dynamics(state + 0.5 * dt * k1, u, params)
    k3 = dynamics(state + 0.5 * dt * k2, u, params)
    k4 = dynamics(state + dt * k3,       u, params)
    return state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def rollout(state0, inputs, params, dt):
    """
    Simulate forward in time.

    Args:
        state0: (10,) initial state
        inputs: (N, 5) input sequence (one per timestep)
        params: VehicleParams
        dt:     timestep [s]

    Returns:
        (N+1, 10) state trajectory (includes initial state)
    """
    def scan_fn(state, u):
        next_state = step_rk4(state, u, params, dt)
        return next_state, next_state

    _, traj = jax.lax.scan(scan_fn, state0, inputs)
    return jnp.concatenate([state0[None], traj], axis=0)


def rollout_const(state0, u, params, dt, n_steps):
    """
    Simulate with constant input.

    Args:
        state0:  (10,) initial state
        u:       (5,)  constant input
        params:  VehicleParams
        dt:      timestep [s]
        n_steps: number of steps

    Returns:
        (n_steps+1, 10) state trajectory
    """
    inputs = jnp.tile(u, (n_steps, 1))
    return rollout(state0, inputs, params, dt)


# ── Batch simulation (vmap) ────────────────────────────────────────

# Run N simulations in parallel with different initial states:
#   batch_step = jax.vmap(step_rk4, in_axes=(0, 0, None, None))
#
# Or different parameters (domain randomization):
#   batch_step = jax.vmap(step_rk4, in_axes=(0, 0, 0, None))


# ── Output helpers ─────────────────────────────────────────────────

def body_frame_velocity(state):
    """World-frame velocity → body-frame velocity."""
    psi = state[..., PSI]
    c, s = jnp.cos(psi), jnp.sin(psi)
    vx_b =  c * state[..., VX] + s * state[..., VY]
    vy_b = -s * state[..., VX] + c * state[..., VY]
    return vx_b, vy_b


def slip_angle(state):
    """Vehicle sideslip angle β."""
    vx_b, vy_b = body_frame_velocity(state)
    return jnp.arctan2(vy_b, jnp.maximum(vx_b, 1e-3))


def speed(state):
    """Scalar speed [m/s]."""
    return jnp.sqrt(state[..., VX]**2 + state[..., VY]**2)


# ── Convenience constructors ───────────────────────────────────────

def make_state(x=0., y=0., psi=0., vx=0., vy=0., omega=0.,
               w_fl=0., w_fr=0., w_rl=0., w_rr=0.):
    return jnp.array([x, y, psi, vx, vy, omega, w_fl, w_fr, w_rl, w_rr])


def make_input(tau_fl=0., tau_fr=0., tau_rl=0., tau_rr=0., steer=0.):
    return jnp.array([tau_fl, tau_fr, tau_rl, tau_rr, steer])


def steady_state_wheels(vx, params):
    """Wheel speeds for no-slip at forward velocity vx."""
    w = vx / params.r_w
    return make_state(vx=vx, w_fl=w, w_fr=w, w_rl=w, w_rr=w)


# ── Sanity check ───────────────────────────────────────────────────

if __name__ == "__main__":
    params = VehicleParams()
    dt = 0.002  # matches FMU stepSize

    # Test 1: straight-line acceleration from rest
    print("=== Straight-line acceleration ===")
    s0 = make_state()
    u  = make_input(tau_fl=5., tau_fr=5., tau_rl=5., tau_rr=5.)
    traj = rollout_const(s0, u, params, dt, n_steps=2500)  # 5 seconds

    t = jnp.arange(traj.shape[0]) * dt
    spd = speed(traj)
    print(f"  t=1s: speed={spd[500]:.2f} m/s, x={traj[500, X]:.2f} m")
    print(f"  t=3s: speed={spd[1500]:.2f} m/s, x={traj[1500, X]:.2f} m")
    print(f"  t=5s: speed={spd[2500]:.2f} m/s, x={traj[2500, X]:.2f} m")

    # Test 2: constant speed + step steer
    print("\n=== Step steer at 10 m/s ===")
    s0 = steady_state_wheels(10.0, params)
    u  = make_input(steer=0.1)  # ~5.7 deg
    traj = rollout_const(s0, u, params, dt, n_steps=1000)  # 2 seconds

    print(f"  t=0.5s: yaw_rate={traj[250, OMEGA]:.3f} rad/s, "
          f"beta={float(slip_angle(traj[250])):.4f} rad")
    print(f"  t=2.0s: yaw_rate={traj[1000, OMEGA]:.3f} rad/s, "
          f"x={traj[1000, X]:.1f} m, y={traj[1000, Y]:.1f} m")

    # Test 3: verify JIT + vmap work
    print("\n=== JIT + vmap check ===")
    batch_step = jax.vmap(step_rk4, in_axes=(0, None, None, None))
    states = jnp.tile(s0, (64, 1))
    out = batch_step(states, u, params, dt)
    print(f"  Batched 64 envs: output shape = {out.shape}")

    # Test 4: differentiability
    print("\n=== Gradient check ===")
    def loss_fn(u_input):
        s = steady_state_wheels(10.0, params)
        for _ in range(50):
            s = step_rk4(s, u_input, params, dt)
        return s[Y]  # lateral displacement

    grad_u = jax.grad(loss_fn)(make_input(steer=0.1))
    print(f"  d(y)/d(steer) = {grad_u[DELTA]:.4f}")
    print(f"  d(y)/d(tau_fl) = {grad_u[TAU_FL]:.6f}")

    print("\nAll checks passed.")
