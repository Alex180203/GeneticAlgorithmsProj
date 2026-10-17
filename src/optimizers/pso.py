"""Global-best Particle Swarm Optimization.

v <- w v + c1 r1 (p_best - x) + c2 r2 (g_best - x)
x <- x + v

* inertia weight w decreases linearly from w_start to w_end (exploration -> exploitation)
* cognitive/social coefficients are clamped to c1 + c2 <= 4 (Clerc stability bound)
* velocity is clamped to +/- v_max_frac * (ub - lb); walls reflect the particle
"""

import numpy as np

from . import OptimizationResult


def pso(cost_fn, lb, ub, swarm_size=30, iterations=60, w_start=0.9, w_end=0.4, c1=1.6, c2=1.6,
        v_max_frac=0.2, seed=None, x0=None, callback=None, **_):
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    span = ub - lb
    if c1 + c2 > 4.0:
        c1, c2 = 4.0 * c1 / (c1 + c2), 4.0 * c2 / (c1 + c2)
    v_max = v_max_frac * span

    x = lb + rng.random((swarm_size, lb.size)) * span
    if x0 is not None:                                   # seed known designs into the swarm
        x0 = np.atleast_2d(x0)
        x[: len(x0)] = np.clip(x0, lb, ub)
    v = rng.uniform(-v_max, v_max, x.shape)
    f = cost_fn(x)
    p_best, p_cost = x.copy(), f.copy()
    g = np.argmin(p_cost)
    g_best, g_cost = p_best[g].copy(), p_cost[g]
    res = OptimizationResult(g_best, g_cost, [g_cost], [float(np.mean(f))], [x.copy()],
                             swarm_size, "PSO")

    for it in range(iterations):
        w = w_start - (w_start - w_end) * it / max(1, iterations - 1)
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
        v = np.clip(v, -v_max, v_max)
        x = x + v
        low, high = x < lb, x > ub                       # reflective boundaries
        x = np.where(low, 2 * lb - x, np.where(high, 2 * ub - x, x))
        v = np.where(low | high, -0.5 * v, v)
        x = np.clip(x, lb, ub)

        f = cost_fn(x)
        improved = f < p_cost
        p_best[improved], p_cost[improved] = x[improved], f[improved]
        g = np.argmin(p_cost)
        if p_cost[g] < g_cost:
            g_best, g_cost = p_best[g].copy(), p_cost[g]

        res.best_history.append(float(g_cost))
        res.mean_history.append(float(np.mean(f)))
        res.positions.append(x.copy())
        res.n_evals += swarm_size
        if callback:
            callback(it, g_best, g_cost)

    res.best_x, res.best_cost = g_best, float(g_cost)
    return res
