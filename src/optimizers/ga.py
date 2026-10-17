"""Real-coded Genetic Algorithm.

* tournament selection (size k)
* SBX (simulated binary, Deb & Agrawal) or BLX-alpha crossover
* Gaussian mutation (sigma relative to the gene range), bounded
* elitism: the best ``elitism`` individuals survive unchanged
"""

import numpy as np

from . import OptimizationResult


def _tournament(rng, cost, k, n):
    contenders = rng.integers(0, len(cost), size=(n, k))
    return contenders[np.arange(n), np.argmin(cost[contenders], axis=1)]


def _sbx(rng, p1, p2, eta, lb, ub):
    u = rng.random(p1.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, lb, ub), np.clip(c2, lb, ub)


def _blx(rng, p1, p2, alpha, lb, ub):
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    d = hi - lo
    sample = lambda: rng.uniform(lo - alpha * d, hi + alpha * d)
    return np.clip(sample(), lb, ub), np.clip(sample(), lb, ub)


def ga(cost_fn, lb, ub, population=30, generations=60, tournament_k=3, crossover="sbx",
       crossover_rate=0.9, sbx_eta=15.0, blx_alpha=0.5, mutation_rate=0.2, mutation_sigma=0.1,
       elitism=2, seed=None, x0=None, callback=None, **_):
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    span = ub - lb
    n, dim = population, lb.size

    pop = lb + rng.random((n, dim)) * span
    if x0 is not None:
        x0 = np.atleast_2d(x0)
        pop[: len(x0)] = np.clip(x0, lb, ub)
    cost = cost_fn(pop)
    b = np.argmin(cost)
    res = OptimizationResult(pop[b].copy(), float(cost[b]), [float(cost[b])], [float(np.mean(cost))],
                             [pop.copy()], n, "GA")

    for gen in range(generations):
        order = np.argsort(cost)
        elite = pop[order[:elitism]]
        n_children = n - elitism
        n_pairs = (n_children + 1) // 2
        parents = pop[_tournament(rng, cost, tournament_k, 2 * n_pairs)]
        p1, p2 = parents[:n_pairs], parents[n_pairs:]
        if crossover == "blx":
            c1, c2 = _blx(rng, p1, p2, blx_alpha, lb, ub)
        else:
            c1, c2 = _sbx(rng, p1, p2, sbx_eta, lb, ub)
        no_cx = rng.random(n_pairs) > crossover_rate
        c1[no_cx], c2[no_cx] = p1[no_cx], p2[no_cx]
        children = np.vstack([c1, c2])[:n_children]

        mutate = rng.random(children.shape) < mutation_rate
        children = np.where(mutate, children + rng.normal(0, mutation_sigma, children.shape) * span, children)
        children = np.clip(children, lb, ub)

        child_cost = cost_fn(children)
        pop = np.vstack([elite, children])
        cost = np.concatenate([cost[order[:elitism]], child_cost])
        b = np.argmin(cost)
        if cost[b] < res.best_cost:
            res.best_x, res.best_cost = pop[b].copy(), float(cost[b])

        res.best_history.append(res.best_cost)
        res.mean_history.append(float(np.mean(cost)))
        res.positions.append(pop.copy())
        res.n_evals += n_children
        if callback:
            callback(gen, res.best_x, res.best_cost)
    return res
