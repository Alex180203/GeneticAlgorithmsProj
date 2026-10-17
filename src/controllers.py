"""Discrete PID with clamping anti-windup and a low-pass-filtered derivative.

All gains/states are numpy arrays so a whole swarm/population (shape (N, M):
N candidates x M joints) is simulated in one vectorised pass.
"""

import numpy as np


class DiscretePID:
    """u = Kp e + Ki sum(e) Ts - Kd d(y_f)/dt, saturated to +/- u_max.

    * Derivative acts on the measurement (no derivative kick on steps) and is
      filtered by a first-order low-pass with cutoff ``deriv_cutoff_hz``.
    * Anti-windup by conditional integration: the integrator freezes while the
      output is saturated *and* the error would push it further into saturation.
    """

    def __init__(self, kp, ki, kd, Ts, u_max, deriv_cutoff_hz=100.0):
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.Ts = Ts
        self.u_max = np.asarray(u_max, dtype=float)
        tf = 1.0 / (2 * np.pi * deriv_cutoff_hz) if deriv_cutoff_hz else 0.0
        self.alpha = Ts / (tf + Ts)
        shape = np.broadcast(self.kp, self.ki, self.kd, self.u_max).shape
        self.integral = np.zeros(shape)
        self.d_filt = np.zeros(shape)
        self.y_prev = None

    def reset(self, y0=0.0):
        self.integral[...] = 0.0
        self.d_filt[...] = 0.0
        self.y_prev = np.broadcast_to(np.asarray(y0, dtype=float), self.integral.shape).copy()

    def update(self, ref, y):
        if self.y_prev is None:
            self.reset(y)
        e = ref - y
        d_raw = -(y - self.y_prev) / self.Ts
        self.d_filt += self.alpha * (d_raw - self.d_filt)
        self.y_prev = np.array(y, dtype=float, copy=True)

        u_unsat = self.kp * e + self.integral + self.kd * self.d_filt
        u = np.clip(u_unsat, -self.u_max, self.u_max)
        winding = (u != u_unsat) & (np.sign(e) == np.sign(u_unsat))
        self.integral = np.where(winding, self.integral, self.integral + self.ki * e * self.Ts)
        return u


def pid_frequency_response(kp, ki, kd, w, deriv_cutoff_hz=100.0):
    """Continuous-time C(jw) of the filtered PID (for Bode / phase margin)."""
    s = 1j * np.asarray(w)
    tf = 1.0 / (2 * np.pi * deriv_cutoff_hz) if deriv_cutoff_hz else 0.0
    kp, ki, kd = (np.asarray(g, dtype=float)[..., None] for g in (kp, ki, kd))
    return kp + ki / s + kd * s / (1 + s * tf)
