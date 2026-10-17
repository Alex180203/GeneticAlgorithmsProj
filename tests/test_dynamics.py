import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import load_config  # noqa: E402
from src.dynamics import JointPlant, simulate_step  # noqa: E402
from src.objective import PIDObjective, stability_margins, step_metrics  # noqa: E402
from src.optimizers import ga, pso  # noqa: E402

CFG = load_config()
PLANT = JointPlant.from_config(CFG)
PD = CFG["baselines"]["thesis_pd"]
KP, KI, KD = (np.array(PD[k]) for k in ("kp", "ki", "kd"))


def test_thesis_pd_margins_match_thesis():
    """Ideal loop: w_c = 35 rad/s, PM = 70 deg  =>  delay margin 34.9 ms."""
    pm, wc, dm = stability_margins(KP, KI, KD, PLANT.J, PLANT.B, deriv_cutoff_hz=0)
    np.testing.assert_allclose(wc, 35.0, atol=0.1)
    np.testing.assert_allclose(pm, 70.0, atol=0.1)
    np.testing.assert_allclose(dm, 0.0349, atol=2e-4)


def test_step_response_converges_without_disturbance():
    res = simulate_step(KP, KI, KD, PLANT, 0.3, 1.5, with_disturbance=False)
    m = step_metrics(res, 0.3)
    assert np.all(m["ess"] < 1e-3)
    assert np.all(m["overshoot"] < 0.05)
    assert not m["diverged"].any()


def test_pd_has_documented_steady_state_error():
    """Friction/gravity bias leaves 3-8 deg steady-state error under pure PD."""
    m = step_metrics(simulate_step(KP, KI, KD, PLANT, 0.3, 1.5), 0.3)
    ess_deg = np.degrees(0.3 * m["ess"][0])
    assert np.all((ess_deg >= 3.0) & (ess_deg <= 8.0)), ess_deg


def test_integral_action_removes_steady_state_error():
    m = step_metrics(simulate_step(KP, np.array([4.0, 6.0, 2.0]), KD, PLANT, 0.3, 2.0), 0.3)
    assert np.all(np.degrees(0.3 * m["ess"]) < 0.5)


@pytest.mark.parametrize("method", ["rk4", "euler"])
def test_integrators_agree(method):
    a = simulate_step(KP, KI, KD, PLANT, 0.3, 0.5, method="rk4")["y"]
    b = simulate_step(KP, KI, KD, PLANT, 0.3, 0.5, method=method)["y"]
    np.testing.assert_allclose(a, b, atol=2e-3)


def _final_amplitude(delay):
    res = simulate_step(KP, KI, KD, PLANT, 0.3, 3.0, delay=delay, saturate=False,
                        with_disturbance=False, deriv_cutoff_hz=1000.0)
    y = res["y"][:, 0, :]
    return np.nanmax(np.abs(y[-500:] - 0.3), axis=0), np.all(np.isfinite(y))


def test_stable_below_delay_margin():
    amp, finite = _final_amplitude(0.025)
    assert finite and np.all(amp < 1e-3)


def test_unstable_above_delay_margin():
    """Beyond tau_d,max = 34.9 ms the loop oscillates with growing amplitude."""
    amp, finite = _final_amplitude(0.040)
    assert (not finite) or np.all(amp > 0.3)


def test_saturation_limits_torque():
    res = simulate_step(KP * 20, KI, KD, PLANT, 0.3, 0.3)
    assert np.all(np.abs(res["u"]) <= PLANT.tau_max + 1e-12)


@pytest.mark.parametrize("optimizer,key", [(pso, "pso"), (ga, "ga")])
def test_optimizers_beat_thesis_pd(optimizer, key):
    obj = PIDObjective(CFG, joints=[0], scenario_delays=[0.0])
    kw = dict(CFG["optimization"][key])
    kw.update({"iterations": 12, "generations": 12, "swarm_size": 16, "population": 16})
    res = optimizer(obj, obj.lb, obj.ub, seed=3, **kw)
    baseline = obj(np.array([KP[0], KI[0], KD[0]]))[0]
    assert res.best_cost < 0.5 * baseline
    assert res.best_history == sorted(res.best_history, reverse=True)      # monotone best
    assert np.all(res.best_x >= obj.lb) and np.all(res.best_x <= obj.ub)
