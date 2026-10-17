"""Cost functions: ITAE, IAE, ISE, overshoot, settling time, steady-state error,
phase margin and delay margin, combined into a scalar fitness for PSO/GA."""

import numpy as np

from .controllers import pid_frequency_response
from .dynamics import JointPlant, plant_frequency_response, simulate_step

W_GRID = np.logspace(-1, 4, 6000)


# --------------------------------------------------------------------------- time domain
def step_metrics(res, amplitude, settle_band=0.02, ss_window=0.1):
    """Metrics of a step response; every entry has shape (N, M)."""
    t, y, u = res["t"], res["y"], res["u"]
    Ts = t[1] - t[0]
    e = amplitude - y
    ae = np.abs(e)
    tt = t[:, None, None]
    diverged = ~np.all(np.isfinite(y), axis=0) | (np.nanmax(np.abs(y), axis=0) > 5 * abs(amplitude))
    ae_f = np.nan_to_num(ae, nan=10 * abs(amplitude))

    outside = ae_f > settle_band * abs(amplitude)
    last_out = len(t) - 1 - np.argmax(outside[::-1], axis=0)
    settling = np.where(outside.any(axis=0), t[np.minimum(last_out + 1, len(t) - 1)], 0.0)
    settling = np.where(outside[-1], t[-1], settling)

    n_ss = max(1, int(round(ss_window / Ts)))
    # sustained oscillation (saturation-bounded limit cycle) counts as unstable
    tail = np.nan_to_num(y[-3 * n_ss:], nan=0.0)
    diverged |= (tail.max(axis=0) - tail.min(axis=0)) > 0.1 * abs(amplitude)
    peak = np.nanmax(np.sign(amplitude) * y, axis=0) / abs(amplitude)
    u_f = np.nan_to_num(u)
    return {
        "IAE": ae_f.sum(axis=0) * Ts / abs(amplitude),
        "ISE": (ae_f**2).sum(axis=0) * Ts / amplitude**2,
        "ITAE": (tt * ae_f).sum(axis=0) * Ts / abs(amplitude),
        "overshoot": np.maximum(0.0, np.nan_to_num(peak, nan=10.0) - 1.0),
        "settling": settling,
        "ess": ae_f[-n_ss:].mean(axis=0) / abs(amplitude),
        "torque_rms": np.sqrt((u_f**2).mean(axis=0)),
        "torque_peak": np.abs(u_f).max(axis=0),
        "diverged": diverged,
    }


# --------------------------------------------------------------------------- frequency domain
def open_loop(kp, ki, kd, J, B, w=W_GRID, loop="position", delay=0.0, deriv_cutoff_hz=100.0, Ts=0.0):
    """L(jw) = C(jw) H(jw) exp(-jw(delay + Ts/2)) with unwrapped phase [deg]."""
    C = pid_frequency_response(kp, ki, kd, w, deriv_cutoff_hz)
    H = plant_frequency_response(J, B, w, loop)
    tau = delay + Ts / 2
    mag = np.abs(C * H)
    # Re(C) > 0 so angle(C) is in (-90, 90); the plant phase is analytic -> no wrapping.
    plant_phase = -np.arctan2(np.asarray(J, float)[..., None] * w, np.asarray(B, float)[..., None])
    if loop == "position":
        plant_phase = plant_phase - np.pi / 2
    phase = np.angle(C) + plant_phase - w * tau
    return mag, np.degrees(phase)


def stability_margins(kp, ki, kd, J, B, loop="position", delay=0.0, deriv_cutoff_hz=100.0, Ts=0.0, w=W_GRID):
    """Phase margin [deg], crossover [rad/s], delay margin [s] (shape of the gains)."""
    mag, phase = open_loop(kp, ki, kd, J, B, w, loop, delay, deriv_cutoff_hz, Ts)
    above = mag >= 1.0
    has_cross = above.any(axis=-1)
    i = mag.shape[-1] - 1 - np.argmax(above[..., ::-1], axis=-1)        # last crossing
    i = np.minimum(i, mag.shape[-1] - 2)
    lm = np.log(mag)
    i0 = np.take_along_axis(lm, i[..., None], -1)[..., 0]
    i1 = np.take_along_axis(lm, (i + 1)[..., None], -1)[..., 0]
    frac = np.clip(i0 / np.where(i0 - i1 == 0, 1, i0 - i1), 0, 1)
    lw = np.log(w)
    wc = np.exp(lw[i] + frac * (lw[i + 1] - lw[i]))
    ph = np.take_along_axis(phase, i[..., None], -1)[..., 0]
    ph1 = np.take_along_axis(phase, (i + 1)[..., None], -1)[..., 0]
    pm = 180.0 + ph + frac * (ph1 - ph)
    pm = np.where(has_cross, pm, 180.0)
    wc = np.where(has_cross, wc, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dm = np.where(wc > 0, np.radians(pm) / wc, np.inf)
    return pm, wc, dm


# --------------------------------------------------------------------------- scalar cost
class PIDObjective:
    """Vectorised fitness for a population X of shape (N, 3*M): [kp, ki, kd] per joint."""

    def __init__(self, cfg, joints=(0, 1, 2), loop=None, scenario_delays=None):
        opt = cfg["optimization"]
        self.cfg = cfg
        self.joints = list(joints)
        self.loop = loop or opt["loop"]
        self.plant = JointPlant.from_config(cfg, self.joints)
        self.Ts = cfg["timing"]["Ts"]
        self.fc = cfg["timing"]["deriv_cutoff_hz"]
        self.amplitude = opt["step_amplitude"] if self.loop == "position" else 1.0
        self.duration = opt["duration"]
        self.delays = list(opt["scenario_delays"] if scenario_delays is None else scenario_delays)
        self.w = opt["weights"]
        self.pm_min = opt["pm_min_deg"]
        self.pm_penalty = opt["pm_penalty"]
        self.dm_min = opt["dm_min"]
        self.dm_penalty = opt["dm_penalty"]
        b = opt["bounds"][self.loop]
        lb, ub = np.array([b["kp"][0], b["ki"][0], b["kd"][0]]), np.array([b["kp"][1], b["ki"][1], b["kd"][1]])
        self.lb, self.ub = np.tile(lb, len(self.joints)), np.tile(ub, len(self.joints))
        self.n_evals = 0

    @property
    def dim(self):
        return 3 * len(self.joints)

    def split(self, X):
        G = np.atleast_2d(X).reshape(-1, len(self.joints), 3)
        return G[..., 0], G[..., 1], G[..., 2]

    def evaluate(self, X, delay=0.0):
        """Simulate one delay scenario; returns metrics dict + per-joint cost (N, M)."""
        kp, ki, kd = self.split(X)
        res = simulate_step(kp, ki, kd, self.plant, self.amplitude, self.duration, Ts=self.Ts,
                            loop=self.loop, delay=delay, deriv_cutoff_hz=self.fc)
        m = step_metrics(res, self.amplitude)
        w = self.w
        tau_excess = np.maximum(0.0, m["torque_rms"] / self.plant.tau_cont - 1.0)
        cost = (w["itae"] * m["ITAE"] + w["overshoot"] * m["overshoot"] + w["settling"] * m["settling"]
                + w["ess"] * m["ess"] + w["torque_rms"] * tau_excess)
        cost = np.where(m["diverged"], 1e3, cost)
        return m, cost, res

    def __call__(self, X):
        X = np.atleast_2d(X)
        self.n_evals += X.shape[0]
        kp, ki, kd = self.split(X)
        total = np.zeros(kp.shape)
        for d in self.delays:
            total += self.evaluate(X, d)[1]
        total /= len(self.delays)
        pm, _, dm = stability_margins(kp, ki, kd, self.plant.J, self.plant.B, self.loop, 0.0, self.fc, self.Ts)
        total += self.pm_penalty * np.maximum(0.0, self.pm_min - pm) / self.pm_min
        total += self.dm_penalty * np.maximum(0.0, self.dm_min - dm) / self.dm_min
        return total.sum(axis=1)
