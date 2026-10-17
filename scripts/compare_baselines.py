"""Compare Thesis PD vs. Manual QUARC PID vs. PSO-PID vs. GA-PID.

Uses results/gains_position.json from tune_joint_pid.py (runs the tuning first
if it is missing) and writes plots plus results/summary.md.

    python scripts/compare_baselines.py
    python scripts/compare_baselines.py --retune      # force a fresh PSO/GA run
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import PROJECT_ROOT, load_config  # noqa: E402
from src import kinematics as kin  # noqa: E402
from src import visualizer as viz  # noqa: E402
from src.dynamics import JointPlant, simulate_joint_tracking, simulate_step, simulate_task_velocity_tracking  # noqa: E402
from src.objective import PIDObjective, stability_margins, step_metrics  # noqa: E402
from tune_joint_pid import run_tuning  # noqa: E402


def load_or_tune(cfg, out, loop, methods, retune, seed):
    path = out / f"gains_{loop}.json"
    gains = {} if retune or not path.exists() else json.loads(path.read_text())
    for m in methods:
        if m not in gains:
            print(f"Tuning {m.upper()} ({loop} loop)...")
            gains[m], _ = run_tuning(cfg, m, "full", loop, seed, verbose=False)
    path.write_text(json.dumps(gains, indent=2))
    return gains


def build_controllers(cfg, gains):
    """{name: (kp, ki, kd, valid_mask)}. QUARC gains exist for joint 1 only."""
    b, q = cfg["baselines"]["thesis_pd"], cfg["baselines"]["quarc_pid_joint1"]
    arr = lambda g: tuple(np.array(g[k], float) for k in ("kp", "ki", "kd"))
    quarc = [a.copy() for a in arr(b)]
    for a, k in zip(quarc, ("kp", "ki", "kd")):
        a[0] = q[k]
    all3 = np.ones(3, bool)
    return {
        "Thesis PD": (*arr(b), all3),
        "QUARC PID": (*quarc, np.array([True, False, False])),
        "PSO-PID": (*arr(gains["pso"]), all3),
        "GA-PID": (*arr(gains["ga"]), all3),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retune", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plant = JointPlant.from_config(cfg)
    Ts, fc = cfg["timing"]["Ts"], cfg["timing"]["deriv_cutoff_hz"]
    T_udp = cfg["timing"]["T_udp"]
    A = cfg["optimization"]["step_amplitude"]
    duration = cfg["optimization"]["duration"]

    gains = load_or_tune(cfg, out, "position", ["pso", "ga"], args.retune, args.seed)
    ctrls = build_controllers(cfg, gains)
    obj = PIDObjective(cfg)

    # ------------------------------------------------------------------ step responses + metrics
    rows = []
    for delay, tag in ((0.0, "local 1 kHz loop"), (T_udp, f"{T_udp * 1e3:.0f} ms UDP loop delay")):
        responses = {}
        for name, (kp, ki, kd, valid) in ctrls.items():
            res = simulate_step(kp, ki, kd, plant, A, duration, Ts=Ts, delay=delay, deriv_cutoff_hz=fc)
            y = res["y"][:, 0, :].copy()
            y[:, ~valid] = np.nan
            responses[name] = (res["t"], y)
            m = step_metrics(res, A)
            _, cost, _ = obj.evaluate(np.ravel(np.column_stack([kp, ki, kd])), delay)
            pm, wc, dm = stability_margins(kp, ki, kd, plant.J, plant.B, "position", 0.0, fc, Ts)
            for j in np.where(valid)[0]:
                rows.append({
                    "scenario": tag, "controller": name, "joint": j + 1,
                    "Kp": kp[j], "Ki": ki[j], "Kd": kd[j],
                    "overshoot_%": 100 * m["overshoot"][0, j], "settling_ms": 1e3 * m["settling"][0, j],
                    "ess_deg": np.degrees(A * m["ess"][0, j]), "ITAE": m["ITAE"][0, j],
                    "PM_deg": pm[j], "wc_rad_s": wc[j], "delay_margin_ms": 1e3 * dm[j],
                    "unstable": bool(m["diverged"][0, j]), "cost": cost[0, j],
                })
        fname = "step_local.png" if delay == 0 else "step_udp_delay.png"
        viz.plot_step_responses(responses, A, out / fname, f"{np.degrees(A):.1f}° step response, {tag}")

    # ------------------------------------------------------------------ Bode
    bode = {n: tuple(np.where(v, g, np.nan) for g in (kp, ki, kd)) for n, (kp, ki, kd, v) in ctrls.items()}
    viz.plot_bode(bode, plant, out / "bode.png", 0.0, fc, Ts)

    # ------------------------------------------------------------------ delay sweep
    delays = np.arange(0.0, 0.0501, 0.0025)
    sweep = {}
    for name, (kp, ki, kd, valid) in ctrls.items():
        if name == "QUARC PID":
            continue
        acc = {"ITAE": [], "overshoot": [], "unstable": []}
        for d in delays:
            m = step_metrics(simulate_step(kp, ki, kd, plant, A, duration, Ts=Ts, delay=d, deriv_cutoff_hz=fc), A)
            acc["ITAE"].append(m["ITAE"][0])
            acc["overshoot"].append(m["overshoot"][0])
            acc["unstable"].append(m["diverged"][0])
        sweep[name] = {k: np.array(v) for k, v in acc.items()}
    viz.plot_delay_sweep(delays, sweep, out / "delay_sweep.png", cfg["timing"]["delay_margin_ref"], T_udp)
    first_unstable = {n: (delays[np.any(s["unstable"], axis=1)][0] * 1e3 if np.any(s["unstable"]) else None)
                      for n, s in sweep.items()}

    # ------------------------------------------------------------------ seam welding (joint space)
    ws = cfg["kinematics"]["workspace"]
    L1, L2 = cfg["kinematics"]["L1"], cfg["kinematics"]["L2"]
    t_all, x_all, xd_all, q_all = kin.seam_welding_trajectory(4.0, Ts, hold=1.5, L1=L1, L2=L2, **ws)
    weld = t_all >= 0                         # score only the welding motion, not the settling dwell
    t, x_ref = t_all[weld], x_all[weld]
    seam = {}
    for delay, fname in ((0.0, "seam_welding_local.png"), (T_udp, "seam_welding_udp_delay.png")):
        tracks = {}
        for name, (kp, ki, kd, valid) in ctrls.items():
            if name == "QUARC PID":
                continue
            q, _ = simulate_joint_tracking((kp, ki, kd), plant, q_all, Ts, delay, T_udp, fc)
            tracks[name] = kin.forward_kinematics(q[weld], L1, L2)
            err = np.linalg.norm(tracks[name] - x_ref, axis=1) * 1e3
            seam[(name, delay)] = (np.sqrt(np.mean(err**2)), err.max())
        lbl = "local loop" if delay == 0 else f"{T_udp * 1e3:.0f} ms loop delay"
        viz.plot_seam_welding(t, x_ref, tracks, out / fname,
                              f"Seam welding, joint-space position PID, 50 Hz UDP commands, {lbl}")

    # ------------------------------------------------------------------ task-space velocity control
    vel_gains = load_or_tune(cfg, out, "velocity", ["pso"], False, args.seed)["pso"]
    lam = tuple(cfg["kinematics"]["dls_lambda"])
    q_tv, _ = simulate_task_velocity_tracking(
        tuple(np.array(vel_gains[k]) for k in ("kp", "ki", "kd")), plant, x_all, xd_all, q_all[0], Ts,
        L1=L1, L2=L2, lam_range=lam, ref_period=T_udp, deriv_cutoff_hz=fc)
    x_tv = kin.forward_kinematics(q_tv[weld], L1, L2)
    q_js, _ = simulate_joint_tracking(tuple(np.array(gains["pso"][k]) for k in ("kp", "ki", "kd")),
                                      plant, q_all, Ts, 0.0, T_udp, fc)
    viz.plot_seam_welding(t, x_ref, {"PSO-PID": kin.forward_kinematics(q_js[weld], L1, L2),
                                     "Task-space velocity (PSO)": x_tv}, out / "seam_task_velocity.png",
                          "Seam welding: joint-space position PID vs task-space velocity PID + DLS Jacobian")
    err_tv = np.linalg.norm(x_tv - x_ref, axis=1) * 1e3

    # ------------------------------------------------------------------ report
    write_summary(out / "summary.md", rows, seam, first_unstable, gains, vel_gains, err_tv, T_udp)
    print_table(rows)
    print("\nSeam welding RMS / max Cartesian error [mm]:")
    for (name, d), (rms, mx) in seam.items():
        print(f"  {name:10s} delay {d * 1e3:4.0f} ms : RMS {rms:6.3f}  max {mx:6.3f}")
    print(f"  Task-space velocity PID (PSO), local: RMS {np.sqrt(np.mean(err_tv**2)):.3f}  max {err_tv.max():.3f}")
    print("\nFirst delay at which any joint becomes unstable:",
          {k: (f"{v:.1f} ms" if v is not None else "> 50 ms") for k, v in first_unstable.items()})
    print(f"\nPlots and summary.md written to {out}")


def print_table(rows):
    hdr = f"{'scenario':24s} {'controller':10s} {'J':>1s} {'Kp':>7s} {'Ki':>7s} {'Kd':>7s} {'OS%':>6s} " \
          f"{'ts[ms]':>7s} {'ess°':>6s} {'ITAE':>7s} {'PM°':>6s} {'DM[ms]':>7s} {'cost':>7s}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        flag = " UNSTABLE" if r["unstable"] else ""
        print(f"{r['scenario']:24s} {r['controller']:10s} {r['joint']:1d} {r['Kp']:7.4f} {r['Ki']:7.4f} {r['Kd']:7.4f} "
              f"{r['overshoot_%']:6.1f} {r['settling_ms']:7.0f} {r['ess_deg']:6.2f} {r['ITAE']:7.4f} "
              f"{r['PM_deg']:6.1f} {r['delay_margin_ms']:7.1f} {r['cost']:7.3f}{flag}")


def write_summary(path, rows, seam, first_unstable, gains, vel_gains, err_tv, T_udp):
    L = ["# PID tuning results: Thesis PD vs QUARC PID vs PSO-PID vs GA-PID", ""]
    L += ["## Tuned gains (position loop, full 9-parameter search)", "",
          "| Method | Joint | Kp | Ki | Kd | PM [deg] | wc [rad/s] | Delay margin [ms] |", "|---|---|---|---|---|---|---|---|"]
    for m in ("pso", "ga"):
        g = gains[m]
        for j in range(3):
            pm = g.get("pm_deg", [np.nan] * 3)[j]
            wc = g.get("wc", [np.nan] * 3)[j]
            dm = g.get("delay_margin_s", [np.nan] * 3)[j] * 1e3
            L.append(f"| {m.upper()} | {j + 1} | {g['kp'][j]:.4f} | {g['ki'][j]:.4f} | {g['kd'][j]:.4f} | "
                     f"{pm:.1f} | {wc:.1f} | {dm:.1f} |")
    L.append("")
    for scen in dict.fromkeys(r["scenario"] for r in rows):
        L += [f"## Step response: {scen}", "",
              "| Controller | Joint | Overshoot [%] | Settling (2 %) [ms] | Steady-state error [deg] | ITAE | PM [deg] | Delay margin [ms] | Cost |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["scenario"] != scen:
                continue
            us = " **unstable**" if r["unstable"] else ""
            L.append(f"| {r['controller']} | {r['joint']} | {r['overshoot_%']:.1f} | {r['settling_ms']:.0f} | "
                     f"{r['ess_deg']:.2f} | {r['ITAE']:.4f} | {r['PM_deg']:.1f} | {r['delay_margin_ms']:.1f} | "
                     f"{r['cost']:.3f}{us} |")
        L.append("")
    L += ["## Seam-welding trajectory (4 s, 12 cm seam, 3 mm / 2 Hz weave, 50 Hz UDP commands)", "",
          "| Controller | Loop delay [ms] | RMS error [mm] | Max error [mm] |", "|---|---|---|---|"]
    for (name, d), (rms, mx) in seam.items():
        L.append(f"| {name} | {d * 1e3:.0f} | {rms:.3f} | {mx:.3f} |")
    L.append(f"| Task-space velocity PID (PSO) | 0 | {np.sqrt(np.mean(err_tv**2)):.3f} | {err_tv.max():.3f} |")
    L += ["", "## Delay robustness (simulated sweep 0-50 ms)", ""]
    for k, v in first_unstable.items():
        L.append(f"- **{k}**: first unstable at {f'{v:.1f} ms' if v is not None else '> 50 ms'}")
    L += ["", "Velocity-loop PSO gains (task-space controller): "
          + ", ".join(f"J{j + 1}: Kp={vel_gains['kp'][j]:.4f} Ki={vel_gains['ki'][j]:.4f} Kd={vel_gains['kd'][j]:.5f}"
                      for j in range(3)), ""]
    Path(path).write_text("\n".join(L))


if __name__ == "__main__":
    main()
