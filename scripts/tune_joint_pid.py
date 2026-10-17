"""Tune the Phantom Omni joint PID gains with PSO and/or GA.

Examples
--------
    python scripts/tune_joint_pid.py                         # PSO + GA, full 9-parameter vector
    python scripts/tune_joint_pid.py --method pso --mode per-joint
    python scripts/tune_joint_pid.py --loop velocity --method pso
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import PROJECT_ROOT, load_config  # noqa: E402
from src.objective import PIDObjective, stability_margins  # noqa: E402
from src.optimizers import OptimizationResult, ga, pso  # noqa: E402
from src import visualizer as viz  # noqa: E402

OPTIMIZERS = {"pso": pso, "ga": ga}
LABELS = {"pso": "PSO-PID", "ga": "GA-PID"}


def thesis_seed(cfg, joints):
    """The thesis PD design, used both as a reference and as one seeded individual."""
    b = cfg["baselines"]["thesis_pd"]
    return np.ravel([[b["kp"][j], b["ki"][j], b["kd"][j]] for j in joints])


def run_tuning(cfg, method="pso", mode="full", loop="position", seed=0, iterations=None, verbose=True):
    """Returns (gains dict with kp/ki/kd lists of length 3, combined OptimizationResult)."""
    opt_cfg = dict(cfg["optimization"][method])
    if iterations:
        opt_cfg["iterations" if method == "pso" else "generations"] = iterations
    groups = [[0, 1, 2]] if mode == "full" else [[0], [1], [2]]
    best = np.zeros(9)
    parts = []
    for g in groups:
        obj = PIDObjective(cfg, joints=g, loop=loop)
        x0 = thesis_seed(cfg, g) if loop == "position" else None

        def cb(it, x, c, g=g):
            if verbose and (it % 10 == 0 or it == opt_cfg.get("iterations", opt_cfg.get("generations")) - 1):
                print(f"  [{method.upper()} joints {[j + 1 for j in g]}] iter {it:3d}  best cost {c:.4f}")

        r = OPTIMIZERS[method](obj, obj.lb, obj.ub, seed=seed, x0=x0, callback=cb, **opt_cfg)
        for k, j in enumerate(g):
            best[3 * j: 3 * j + 3] = r.best_x[3 * k: 3 * k + 3]
        parts.append(r)

    # Combine the per-joint runs into one 9-D result (sum of costs, concatenated populations)
    combined = OptimizationResult(
        best_x=best, best_cost=float(sum(p.best_cost for p in parts)),
        best_history=list(np.sum([p.best_history for p in parts], axis=0)),
        mean_history=list(np.sum([p.mean_history for p in parts], axis=0)),
        positions=[np.hstack([p.positions[i] for p in parts]) if mode == "per-joint" else parts[0].positions[i]
                   for i in range(len(parts[0].positions))],
        n_evals=sum(p.n_evals for p in parts), method=method.upper())
    G = best.reshape(3, 3)
    gains = {"kp": G[:, 0].tolist(), "ki": G[:, 1].tolist(), "kd": G[:, 2].tolist(),
             "cost": combined.best_cost, "n_evals": combined.n_evals, "mode": mode, "loop": loop, "seed": seed}
    return gains, combined


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["pso", "ga", "both"], default="both")
    ap.add_argument("--mode", choices=["full", "per-joint"], default="full",
                    help="full: one 9-parameter search; per-joint: three 3-parameter searches")
    ap.add_argument("--loop", choices=["position", "velocity"], default="position")
    ap.add_argument("--iterations", type=int, default=None, help="override iterations/generations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    methods = ["pso", "ga"] if args.method == "both" else [args.method]

    all_gains, results = {}, {}
    for m in methods:
        print(f"\n=== {m.upper()} | mode={args.mode} | loop={args.loop} ===")
        t0 = time.time()
        gains, res = run_tuning(cfg, m, args.mode, args.loop, args.seed, args.iterations)
        gains["seconds"] = round(time.time() - t0, 1)
        all_gains[m], results[LABELS[m]] = gains, res

        obj = PIDObjective(cfg, loop=args.loop)
        pm, wc, dm = stability_margins(np.array(gains["kp"]), np.array(gains["ki"]), np.array(gains["kd"]),
                                       obj.plant.J, obj.plant.B, args.loop, 0.0, obj.fc, obj.Ts)
        gains.update(pm_deg=pm.tolist(), wc=wc.tolist(), delay_margin_s=dm.tolist())
        print(f"  done in {gains['seconds']} s, {gains['n_evals']} evaluations, cost {gains['cost']:.4f}")
        for j in range(3):
            print(f"  joint {j + 1}: Kp={gains['kp'][j]:.4f}  Ki={gains['ki'][j]:.4f}  Kd={gains['kd'][j]:.4f}"
                  f"   PM={pm[j]:.1f} deg  wc={wc[j]:.1f} rad/s  delay margin={dm[j] * 1e3:.1f} ms")

    path = out / f"gains_{args.loop}.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    previous.update(all_gains)
    path.write_text(json.dumps(previous, indent=2))
    print(f"\nGains saved to {path}")

    baseline = None
    if args.loop == "position":
        obj = PIDObjective(cfg)
        baseline = float(obj(thesis_seed(cfg, [0, 1, 2]))[0])
        print(f"Thesis PD cost on the same objective: {baseline:.4f}")
    viz.plot_convergence(results, out / f"convergence_{args.loop}.png", baseline)
    for label, r in results.items():
        obj = PIDObjective(cfg, loop=args.loop)
        b = cfg["baselines"]["thesis_pd"]
        viz.plot_swarm_evolution(r, obj.lb, obj.ub, out / f"population_{r.method.lower()}_{args.loop}.png",
                                 joint=0, marker=(b["kp"][0], b["kd"][0]) if args.loop == "position" else None)
    print(f"Plots saved in {out}")
    return all_gains, results


if __name__ == "__main__":
    main()
