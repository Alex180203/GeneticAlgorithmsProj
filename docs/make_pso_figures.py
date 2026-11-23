import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import load_config
from src import kinematics as kin
from src import visualizer as viz
from src.dynamics import JointPlant, simulate_joint_tracking, simulate_step, simulate_task_velocity_tracking
from src.objective import PIDObjective, stability_margins, step_metrics
from src.optimizers import pso

import matplotlib.pyplot as plt

PSO = viz.color_for("PSO-PID")
PD = viz.color_for("Thesis PD")
UNSTABLE_COST = 500.0
GAIN_NAMES = ["Kp", "Ki", "Kd"]


def thesis_vector(cfg):
    b = cfg["baselines"]["thesis_pd"]
    return np.ravel([[b["kp"][j], b["ki"][j], b["kd"][j]] for j in range(3)])


def split(x):
    G = np.asarray(x, float).reshape(3, 3)
    return G[:, 0].copy(), G[:, 1].copy(), G[:, 2].copy()


def run_pso(cfg, seed):
    obj = PIDObjective(cfg)
    costs, trail = [], []

    def cost_fn(X):
        c = obj(X)
        costs.append(c.copy())
        return c

    def callback(it, g, c):
        trail.append(np.array(g, float))

    t0 = time.time()
    res = pso(cost_fn, obj.lb, obj.ub, seed=seed, x0=thesis_vector(cfg), callback=callback,
              **cfg["optimization"]["pso"])
    seconds = time.time() - t0
    costs = np.array(costs)
    g0 = res.positions[0][int(np.argmin(costs[0]))]
    return res, np.vstack([g0, trail]), costs, obj, seconds


def diversity(positions, lb, ub):
    out = []
    for P in positions:
        Z = (P - lb) / (ub - lb)
        out.append(np.mean(np.linalg.norm(Z - Z.mean(axis=0), axis=1)) / np.sqrt(Z.shape[1]))
    return np.array(out)


def inertia(pcfg):
    T = pcfg["iterations"]
    return pcfg["w_start"] - (pcfg["w_start"] - pcfg["w_end"]) * np.arange(T) / max(1, T - 1)


def plot_swarm_dynamics(res, costs, lb, ub, pcfg, path):
    it = np.arange(len(res.positions))
    div = diversity(res.positions, lb, ub)
    unstable = 100 * np.mean(costs >= UNSTABLE_COST, axis=1)
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.4), sharex=True)
    axes[0].plot(it[1:], inertia(pcfg), color=PSO)
    axes[0].set_ylabel("w")
    axes[0].set_title("Inertia weight used to produce each iteration")
    axes[1].plot(it, div, color=PSO)
    axes[1].set_ylabel("diversity D")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Swarm diversity: mean normalised distance of the particles to their centroid")
    axes[2].plot(it, unstable, color=PSO)
    axes[2].set_ylabel("particles [%]")
    axes[2].set_ylim(bottom=0)
    axes[2].set_title("Share of particles whose gains make at least one loop unstable")
    axes[2].set_xlabel("iteration")
    viz._save(fig, path)
    return div, unstable


def plot_gbest(trail, pd_vec, path):
    it = np.arange(len(trail))
    fig, axes = plt.subplots(3, 3, figsize=(13, 7.6), sharex=True)
    for j in range(3):
        for k in range(3):
            ax = axes[k, j]
            ax.step(it, trail[:, 3 * j + k], where="post", color=PSO, label="PSO global best")
            ax.axhline(pd_vec[3 * j + k], color=PD, ls="--", lw=1.2, label="Thesis PD")
            if k == 0:
                ax.set_title(viz.JOINT_NAMES[j])
            if j == 0:
                ax.set_ylabel(GAIN_NAMES[k])
            if k == 2:
                ax.set_xlabel("iteration")
    axes[1, 0].legend(loc="center right", fontsize=8)
    fig.suptitle("Evolution of the global-best gain vector", color=viz.TEXT, fontweight="bold")
    viz._save(fig, path)


def plot_seeds(histories, baseline, path):
    H = np.array(histories)
    it = np.arange(H.shape[1])
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, h in enumerate(H):
        ax.plot(it, h, color=PSO, lw=1.0, alpha=0.35, label=f"individual seeds ({len(H)})" if i == 0 else None)
    ax.plot(it, np.median(H, axis=0), color=PSO, lw=2.4, label="median over seeds")
    ax.axhline(baseline, color=PD, lw=1.2, ls="--", label="Thesis PD cost")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("best cost (log)")
    ax.set_title(f"PSO best cost for {len(H)} random seeds")
    ax.legend(loc="lower left", fontsize=8)
    viz._save(fig, path)


def plot_swarm(res, lb, ub, path, marker, joint=0, dims=(0, 2), labels=("Kp", "Kd"), snapshots=(0, 5, 15, 30, -1)):
    snaps = [s if s >= 0 else len(res.positions) - 1 for s in snapshots]
    a, b = 3 * joint + dims[0], 3 * joint + dims[1]
    fig, axes = plt.subplots(1, len(snaps), figsize=(3.3 * len(snaps), 3.6), sharex=True, sharey=True)
    for ax, s in zip(axes, snaps):
        P = res.positions[s]
        ax.scatter(P[:, a], P[:, b], s=30, color=PSO, edgecolor=viz.SURFACE, linewidth=1.5, zorder=4,
                   label="particles")
        ax.plot(*marker, "D", ms=8, color=PD, mec=viz.SURFACE, mew=2, zorder=5, label="Thesis PD")
        ax.plot(res.best_x[a], res.best_x[b], "*", ms=15, color=viz.TEXT, mec=viz.SURFACE, mew=1.2, zorder=6,
                label="final global best")
        ax.set_xlim(lb[a], ub[a])
        ax.set_ylim(lb[b], ub[b])
        ax.set_title(f"iteration {s}")
        ax.set_xlabel(labels[0])
    axes[0].set_ylabel(labels[1])
    axes[0].legend(loc="upper right", fontsize=7.5)
    fig.suptitle(f"Swarm in the {labels[0]}-{labels[1]} plane of {viz.JOINT_NAMES[joint]}", color=viz.TEXT,
                 fontweight="bold")
    viz._save(fig, path)


def first_unstable(kp, ki, kd, plant, A, duration, Ts, fc, delays):
    first = np.full(kp.shape[0], np.nan)
    for d in delays:
        m = step_metrics(simulate_step(kp, ki, kd, plant, A, duration, Ts=Ts, delay=d, deriv_cutoff_hz=fc), A)
        bad = m["diverged"].any(axis=1) & np.isnan(first)
        first[bad] = d
    return first


def controllers(cfg, pso_x):
    b, q = cfg["baselines"]["thesis_pd"], cfg["baselines"]["quarc_pid_joint1"]
    pd = tuple(np.array(b[k], float) for k in ("kp", "ki", "kd"))
    quarc = [a.copy() for a in pd]
    for a, k in zip(quarc, ("kp", "ki", "kd")):
        a[0] = q[k]
    return {
        "Thesis PD": (*pd, np.ones(3, bool)),
        "QUARC PID": (*quarc, np.array([True, False, False])),
        "PSO-PID": (*split(pso_x), np.ones(3, bool)),
    }


def comparison_figures(cfg, pso_x, vel_gains, out):
    plant = JointPlant.from_config(cfg)
    Ts, fc, T_udp = cfg["timing"]["Ts"], cfg["timing"]["deriv_cutoff_hz"], cfg["timing"]["T_udp"]
    A, duration = cfg["optimization"]["step_amplitude"], cfg["optimization"]["duration"]
    ctrls = controllers(cfg, pso_x)
    report = {}

    for delay, fname, tag in ((0.0, "pso_step_local.png", "local 1 kHz loop"),
                              (T_udp, "pso_step_udp.png", f"{T_udp * 1e3:.0f} ms feedback delay")):
        responses = {}
        for name, (kp, ki, kd, valid) in ctrls.items():
            res = simulate_step(kp, ki, kd, plant, A, duration, Ts=Ts, delay=delay, deriv_cutoff_hz=fc)
            y = res["y"][:, 0, :].copy()
            y[:, ~valid] = np.nan
            responses[name] = (res["t"], y)
        viz.plot_step_responses(responses, A, out / fname, f"{np.degrees(A):.1f}° step response, {tag}")

    bode = {n: tuple(np.where(v, g, np.nan) for g in (kp, ki, kd)) for n, (kp, ki, kd, v) in ctrls.items()}
    viz.plot_bode(bode, plant, out / "pso_bode.png", 0.0, fc, Ts)

    delays = np.arange(0.0, 0.0501, 0.0025)
    sweep = {}
    for name in ("Thesis PD", "PSO-PID"):
        kp, ki, kd, _ = ctrls[name]
        acc = {"ITAE": [], "overshoot": [], "unstable": []}
        for d in delays:
            m = step_metrics(simulate_step(kp, ki, kd, plant, A, duration, Ts=Ts, delay=d, deriv_cutoff_hz=fc), A)
            acc["ITAE"].append(m["ITAE"][0])
            acc["overshoot"].append(m["overshoot"][0])
            acc["unstable"].append(m["diverged"][0])
        sweep[name] = {k: np.array(v) for k, v in acc.items()}
        u = np.any(sweep[name]["unstable"], axis=1)
        report[f"first_unstable_ms_{name}"] = float(delays[u][0] * 1e3) if u.any() else None
    viz.plot_delay_sweep(delays, sweep, out / "pso_delay_sweep.png", cfg["timing"]["delay_margin_ref"], T_udp)

    ws = cfg["kinematics"]["workspace"]
    L1, L2 = cfg["kinematics"]["L1"], cfg["kinematics"]["L2"]
    t_all, x_all, xd_all, q_all = kin.seam_welding_trajectory(4.0, Ts, hold=1.5, L1=L1, L2=L2, **ws)
    weld = t_all >= 0
    t, x_ref = t_all[weld], x_all[weld]
    for delay, fname in ((0.0, "pso_seam_local.png"), (T_udp, "pso_seam_udp.png")):
        tracks = {}
        for name in ("Thesis PD", "PSO-PID"):
            kp, ki, kd, _ = ctrls[name]
            q, _ = simulate_joint_tracking((kp, ki, kd), plant, q_all, Ts, delay, T_udp, fc)
            tracks[name] = kin.forward_kinematics(q[weld], L1, L2)
            err = np.linalg.norm(tracks[name] - x_ref, axis=1) * 1e3
            report[f"seam_rms_mm_{name}_{delay * 1e3:.0f}ms"] = float(np.sqrt(np.mean(err**2)))
        lbl = "local loop" if delay == 0 else f"{T_udp * 1e3:.0f} ms feedback delay"
        viz.plot_seam_welding(t, x_ref, tracks, out / fname,
                              f"Seam welding, joint-space position PID, 50 Hz UDP commands, {lbl}")

    lam = tuple(cfg["kinematics"]["dls_lambda"])
    q_tv, _ = simulate_task_velocity_tracking(tuple(np.array(vel_gains[k]) for k in ("kp", "ki", "kd")), plant,
                                              x_all, xd_all, q_all[0], Ts, L1=L1, L2=L2, lam_range=lam,
                                              ref_period=T_udp, deriv_cutoff_hz=fc)
    q_js, _ = simulate_joint_tracking(ctrls["PSO-PID"][:3], plant, q_all, Ts, 0.0, T_udp, fc)
    x_tv = kin.forward_kinematics(q_tv[weld], L1, L2)
    viz.plot_seam_welding(t, x_ref, {"PSO-PID": kin.forward_kinematics(q_js[weld], L1, L2),
                                     "Task-space velocity (PSO)": x_tv}, out / "pso_seam_task_velocity.png",
                          "Seam welding: joint-space position PID vs task-space velocity PID (both tuned by PSO)")
    err = np.linalg.norm(x_tv - x_ref, axis=1) * 1e3
    report["seam_rms_mm_task_space"] = float(np.sqrt(np.mean(err**2)))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "figures"))
    args = ap.parse_args(argv)
    cfg = load_config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pcfg = cfg["optimization"]["pso"]

    res, trail, costs, obj, seconds = run_pso(cfg, 0)
    stored = json.loads((ROOT / "results" / "gains_position.json").read_text())["pso"]
    stored_x = np.ravel(np.column_stack([stored["kp"], stored["ki"], stored["kd"]]))
    reproduced = bool(np.allclose(res.best_x, stored_x, rtol=1e-6, atol=1e-9))
    print(f"seed 0: cost {res.best_cost:.6f} in {seconds:.1f} s, matches results/gains_position.json: {reproduced}")

    pd_vec = thesis_vector(cfg)
    baseline = float(PIDObjective(cfg)(pd_vec)[0])
    viz.plot_convergence({"PSO-PID": res}, out / "pso_convergence.png", baseline)
    div, unstable = plot_swarm_dynamics(res, costs, obj.lb, obj.ub, pcfg, out / "pso_swarm_dynamics.png")
    plot_gbest(trail, pd_vec, out / "pso_gbest_evolution.png")
    b = cfg["baselines"]["thesis_pd"]
    plot_swarm(res, obj.lb, obj.ub, out / "pso_swarm_kp_kd.png", (b["kp"][0], b["kd"][0]))
    plot_swarm(res, obj.lb, obj.ub, out / "pso_swarm_kp_ki.png", (b["kp"][0], b["ki"][0]), dims=(0, 1),
               labels=("Kp", "Ki"))

    plant = JointPlant.from_config(cfg)
    Ts, fc = cfg["timing"]["Ts"], cfg["timing"]["deriv_cutoff_hz"]
    seeds = list(range(args.seeds))
    runs = [(0, res, seconds)]
    for s in seeds[1:]:
        r, _, _, _, sec = run_pso(cfg, s)
        print(f"seed {s}: cost {r.best_cost:.4f} in {sec:.1f} s")
        runs.append((s, r, sec))
    X = np.array([r.best_x for _, r, _ in runs])
    G = X.reshape(len(runs), 3, 3)
    kp, ki, kd = G[..., 0], G[..., 1], G[..., 2]
    pm, wc, dm = stability_margins(kp, ki, kd, plant.J, plant.B, "position", 0.0, fc, Ts)
    first = first_unstable(kp, ki, kd, plant, cfg["optimization"]["step_amplitude"],
                           cfg["optimization"]["duration"], Ts, fc, np.arange(0.0, 0.0601, 0.0025))
    plot_seeds([r.best_history for _, r, _ in runs], baseline, out / "pso_seeds.png")

    marks = [0, 5, 10, 20, 30, 40, 50, 60]
    study = {
        "reproduced_stored_gains": reproduced,
        "baseline_cost": baseline,
        "seed0": {
            "cost": res.best_cost, "seconds": seconds, "n_evals": res.n_evals,
            "iterations": {str(i): {"best_cost": res.best_history[i], "mean_cost": res.mean_history[i],
                                    "diversity": float(div[i]), "unstable_pct": float(unstable[i]),
                                    "gbest": trail[i].tolist()} for i in marks},
        },
        "seeds": [{"seed": s, "cost": r.best_cost, "seconds": sec, "gains": r.best_x.tolist(),
                   "best_history": [float(v) for v in r.best_history],
                   "pm_min_deg": float(pm[i].min()), "dm_min_ms": float(dm[i].min() * 1e3),
                   "first_unstable_ms": None if np.isnan(first[i]) else float(first[i] * 1e3)}
                  for i, (s, r, sec) in enumerate(runs)],
    }
    costs_all = np.array([r.best_cost for _, r, _ in runs])
    study["seed_summary"] = {"mean": float(costs_all.mean()), "std": float(costs_all.std(ddof=1)),
                             "min": float(costs_all.min()), "max": float(costs_all.max()),
                             "median": float(np.median(costs_all))}

    vel = json.loads((ROOT / "results" / "gains_velocity.json").read_text())["pso"]
    study["comparison"] = comparison_figures(cfg, res.best_x, vel, out)
    (out / "pso_study.json").write_text(json.dumps(study, indent=2))
    print(json.dumps(study["seed_summary"], indent=2))
    print(json.dumps(study["comparison"], indent=2))
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
