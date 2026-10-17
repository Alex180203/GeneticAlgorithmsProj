"""Discrete 1 kHz simulation of the decoupled joint dynamics

    J_i q_i'' + B_i q_i' + d_i(q_i') = tau_i,
    d_i = bias_i + coulomb_i * tanh(q_i' / v_s)        (gravity/static + Coulomb)

with torque saturation, an in-loop transport delay (feedback over UDP) and an
optional zero-order hold on the feedback/reference at the UDP rate.

Every simulator is vectorised: gains of shape (N, M) simulate N candidate
controllers on M joints simultaneously.
"""

from dataclasses import dataclass

import numpy as np

from .controllers import DiscretePID
from . import kinematics as kin


@dataclass
class JointPlant:
    J: np.ndarray
    B: np.ndarray
    bias: np.ndarray
    coulomb: np.ndarray
    tau_max: np.ndarray
    tau_cont: np.ndarray
    v_s: float = 0.01

    @classmethod
    def from_config(cls, cfg, joints=None):
        d = cfg["dynamics"]
        idx = list(range(3)) if joints is None else list(np.atleast_1d(joints))
        pick = lambda key: np.asarray(d[key], dtype=float)[idx]
        return cls(J=pick("J"), B=pick("B"), bias=pick("bias"), coulomb=pick("coulomb"),
                   tau_max=pick("tau_max"), tau_cont=pick("tau_cont"),
                   v_s=float(d["coulomb_smoothing"]))

    def disturbance(self, qd):
        return self.bias + self.coulomb * np.tanh(qd / self.v_s)

    def accel(self, qd, tau, with_disturbance=True):
        d = self.disturbance(qd) if with_disturbance else 0.0
        return (tau - self.B * qd - d) / self.J


def plant_frequency_response(J, B, w, loop="position"):
    """H_pos = 1/(s(Js+B)),  H_vel = 1/(Js+B)."""
    s = 1j * np.asarray(w)
    J = np.asarray(J, dtype=float)[..., None]
    B = np.asarray(B, dtype=float)[..., None]
    H = 1.0 / (J * s + B)
    return H / s if loop == "position" else H


def _step(plant, q, qd, tau, Ts, method, with_disturbance):
    """Advance one sample with ZOH torque (RK4 or semi-implicit Euler)."""
    if method == "euler":
        qd_new = qd + Ts * plant.accel(qd, tau, with_disturbance)
        return q + Ts * qd_new, qd_new
    a = lambda v: plant.accel(v, tau, with_disturbance)
    k1v, k1a = qd, a(qd)
    k2v, k2a = qd + 0.5 * Ts * k1a, a(qd + 0.5 * Ts * k1a)
    k3v, k3a = qd + 0.5 * Ts * k2a, a(qd + 0.5 * Ts * k2a)
    k4v, k4a = qd + Ts * k3a, a(qd + Ts * k3a)
    q_new = q + Ts / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    qd_new = qd + Ts / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)
    return q_new, qd_new


def simulate(kp, ki, kd, plant, ref, Ts=0.001, loop="position", delay=0.0,
             feedback_period=None, deriv_cutoff_hz=100.0, saturate=True,
             with_disturbance=True, method="rk4", q0=None):
    """Closed-loop simulation for an arbitrary reference.

    ref : array (T,) or (T, M) of position [rad] or velocity [rad/s] references.
    Returns dict with t (T,), y (T, N, M) controlled output, q, u (T, N, M).
    """
    kp, ki, kd = np.broadcast_arrays(*(np.atleast_2d(np.asarray(g, dtype=float)) for g in (kp, ki, kd)))
    shape = kp.shape
    ref = np.asarray(ref, dtype=float)
    if ref.ndim == 1:
        ref = ref[:, None]
    T = ref.shape[0]
    u_max = plant.tau_max if saturate else np.full_like(plant.tau_max, np.inf)
    pid = DiscretePID(kp, ki, kd, Ts, u_max, deriv_cutoff_hz)

    q = np.zeros(shape) if q0 is None else np.broadcast_to(np.asarray(q0, float), shape).copy()
    qd = np.zeros(shape)
    D = int(round(delay / Ts))
    P = max(1, int(round(feedback_period / Ts))) if feedback_period else 1

    y_hist = np.empty((T,) + shape)
    q_hist = np.empty((T,) + shape)
    u_hist = np.empty((T,) + shape)
    out = q if loop == "position" else qd
    pid.reset(out)
    for k in range(T):
        q_hist[k] = q
        y_hist[k] = q if loop == "position" else qd
        j = max(0, k - D)
        j -= j % P
        u = pid.update(ref[k], y_hist[j])
        u_hist[k] = u
        q, qd = _step(plant, q, qd, u, Ts, method, with_disturbance)
        if not np.all(np.isfinite(q)):
            y_hist[k + 1:] = np.nan
            q_hist[k + 1:] = np.nan
            u_hist[k + 1:] = np.nan
            break
    return {"t": np.arange(T) * Ts, "y": y_hist, "q": q_hist, "u": u_hist, "ref": ref}


def simulate_step(kp, ki, kd, plant, amplitude=0.3, duration=1.5, **kw):
    Ts = kw.get("Ts", 0.001)
    T = int(round(duration / Ts))
    return simulate(kp, ki, kd, plant, np.full(T, amplitude), **kw)


def zoh_resample(signal, period_steps):
    """Hold every ``period_steps``-th sample (commands arriving at the UDP rate)."""
    idx = np.arange(len(signal))
    return signal[idx - idx % period_steps]


def simulate_joint_tracking(gains, plant, q_ref, Ts=0.001, delay=0.0, ref_period=0.020,
                            deriv_cutoff_hz=100.0):
    """Joint-space position tracking of q_ref (T, 3) received over UDP at 50 Hz."""
    kp, ki, kd = (np.asarray(g, dtype=float)[None, :] for g in gains)
    q_cmd = zoh_resample(q_ref, max(1, int(round(ref_period / Ts))))
    res = simulate(kp, ki, kd, plant, q_cmd, Ts=Ts, loop="position", delay=delay,
                   deriv_cutoff_hz=deriv_cutoff_hz, q0=q_ref[0])
    return res["q"][:, 0, :], res["u"][:, 0, :]


def simulate_task_velocity_tracking(vel_gains, plant, x_ref, xd_ref, q0, Ts=0.001,
                                    Kx=20.0, L1=0.131, L2=0.131, lam_range=(0.01, 0.05),
                                    ref_period=0.020, deriv_cutoff_hz=100.0):
    """Task-space velocity control: Cartesian P loop + DLS inverse Jacobian
    producing joint-velocity references for per-joint velocity PIDs."""
    kp, ki, kd = (np.asarray(g, dtype=float) for g in vel_gains)
    pid = DiscretePID(kp, ki, kd, Ts, plant.tau_max, deriv_cutoff_hz)
    P = max(1, int(round(ref_period / Ts)))
    x_cmd, xd_cmd = zoh_resample(x_ref, P), zoh_resample(xd_ref, P)
    q, qd = np.array(q0, float), np.zeros(3)
    pid.reset(qd)
    T = len(x_ref)
    q_hist, u_hist = np.empty((T, 3)), np.empty((T, 3))
    for k in range(T):
        q_hist[k] = q
        x = kin.forward_kinematics(q, L1, L2)
        J = kin.jacobian(q, L1, L2)
        v_cmd = xd_cmd[k] + Kx * (x_cmd[k] - x)
        qd_ref = kin.dls_inverse(J, lam_range=lam_range) @ v_cmd
        u = pid.update(qd_ref, qd)
        u_hist[k] = u
        q, qd = _step(plant, q, qd, u, Ts, "rk4", True)
    return q_hist, u_hist
