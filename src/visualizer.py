"""Plots: step responses, Bode, convergence, swarm evolution, delay sweep, 3D seam welding."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .objective import open_loop  # noqa: E402

# Reference categorical palette (fixed order - colour follows the controller, never its rank)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CONTROLLER_COLORS = {"Thesis PD": SERIES[0], "QUARC PID": SERIES[1], "PSO-PID": SERIES[2], "GA-PID": SERIES[3],
                     "Task-space velocity (PSO)": SERIES[6]}
SURFACE, TEXT, TEXT_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
JOINT_NAMES = ["Joint 1 (base)", "Joint 2 (shoulder)", "Joint 3 (elbow)"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT,
    "lines.linewidth": 2.0, "legend.frameon": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
})


def color_for(name, i=0):
    return CONTROLLER_COLORS.get(name, SERIES[i % len(SERIES)])


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_step_responses(responses, amplitude, path, title="Step responses", unit="deg"):
    """responses: {name: (t, y (T, 3) or None per joint)}; NaN columns are skipped."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    scale = np.degrees(1) if unit == "deg" else 1.0
    for j, ax in enumerate(axes):
        ax.axhline(amplitude * scale, color=MUTED, lw=1.2, ls="--", label="reference")
        ax.axhspan(0.98 * amplitude * scale, 1.02 * amplitude * scale, color=GRID, alpha=0.6, lw=0)
        for i, (name, (t, y)) in enumerate(responses.items()):
            if y is None or np.all(np.isnan(y[:, j])):
                continue
            ax.plot(t, y[:, j] * scale, color=color_for(name, i), label=name)
        ax.set_title(JOINT_NAMES[j])
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"joint angle [{unit}]")
    axes[0].legend(loc="lower right")
    fig.suptitle(title, color=TEXT, fontweight="bold")
    return _save(fig, path)


def plot_bode(controllers, plant, path, delay=0.0, deriv_cutoff_hz=100.0, Ts=0.001, loop="position"):
    """controllers: {name: (kp, ki, kd) arrays of length 3 (NaN = not available)}."""
    w = np.logspace(-0.5, 3.5, 1500)
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.5), sharex=True)
    for j in range(3):
        am, ap = axes[0, j], axes[1, j]
        for i, (name, (kp, ki, kd)) in enumerate(controllers.items()):
            if np.isnan(kp[j]):
                continue
            mag, ph = open_loop(kp[j], ki[j], kd[j], plant.J[j], plant.B[j], w, loop, delay, deriv_cutoff_hz, Ts)
            c = color_for(name, i)
            am.semilogx(w, 20 * np.log10(mag), color=c, label=name)
            ap.semilogx(w, ph, color=c)
            k = np.where(mag >= 1)[0]
            if len(k):
                wc, pm = w[k[-1]], 180 + ph[k[-1]]
                ap.plot(wc, ph[k[-1]], "o", ms=7, color=c, mec=SURFACE, mew=2,
                        label=f"{name}: PM {pm:.0f}° @ {wc:.0f} rad/s")
        ap.legend(loc="lower left", fontsize=7.5)
        am.axhline(0, color=MUTED, lw=1)
        ap.axhline(-180, color=MUTED, lw=1)
        am.set_title(JOINT_NAMES[j])
        am.set_ylabel("|L(jω)| [dB]")
        ap.set_ylabel("∠L(jω) [deg]")
        ap.set_xlabel("ω [rad/s]")
        ap.set_ylim(-270, -60)
    axes[0, 0].legend(loc="lower left")
    tag = f" with {delay * 1e3:.0f} ms loop delay" if delay else ""
    fig.suptitle(f"Open-loop Bode plots{tag}", color=TEXT, fontweight="bold")
    return _save(fig, path)


def plot_convergence(results, path, baseline_cost=None):
    """results: {name: OptimizationResult}."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, r) in enumerate(results.items()):
        c = color_for(name, i)
        it = np.arange(len(r.best_history))
        ax.plot(it, r.best_history, color=c, label=f"{name} best")
        ax.plot(it, r.mean_history, color=c, lw=1.2, ls=":", alpha=0.8, label=f"{name} population mean")
        ax.annotate(f"{r.best_history[-1]:.2f}", (it[-1], r.best_history[-1]), textcoords="offset points",
                    xytext=(4, 4), fontsize=9, color=TEXT_2)
    if baseline_cost is not None:
        ax.axhline(baseline_cost, color=color_for("Thesis PD"), lw=1.2, ls="--", label="Thesis PD cost")
    ax.set_yscale("log")
    ax.set_xlabel("iteration / generation")
    ax.set_ylabel("cost (log)")
    ax.set_title("Optimizer convergence")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, path)


def plot_swarm_evolution(result, lb, ub, path, joint=0, snapshots=(0, 5, 20, -1), dims=(0, 2),
                         labels=("Kp", "Kd"), marker=None):
    """Scatter of the population in a 2-D gain plane at several iterations."""
    snaps = [s if s >= 0 else len(result.positions) - 1 for s in snapshots]
    fig, axes = plt.subplots(1, len(snaps), figsize=(4 * len(snaps), 3.8), sharex=True, sharey=True)
    a, b = joint * 3 + dims[0], joint * 3 + dims[1]
    for ax, s in zip(axes, snaps):
        P = result.positions[s]
        ax.scatter(P[:, a], P[:, b], s=28, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, alpha=0.85, zorder=4, label="particles")
        if marker is not None:
            ax.plot(marker[0], marker[1], "D", ms=8, color=SERIES[1], mec=SURFACE, mew=2, label="Thesis PD")
        ax.plot(result.best_x[a], result.best_x[b], "*", ms=14, color=SERIES[2], mec=SURFACE, mew=1.5,
                label="final best")
        ax.set_xlim(lb[a], ub[a])
        ax.set_ylim(lb[b], ub[b])
        ax.set_title(f"iteration {s}")
        ax.set_xlabel(labels[0])
    axes[0].set_ylabel(labels[1])
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{result.method} population in the {labels[0]}-{labels[1]} plane, {JOINT_NAMES[joint]}",
                 color=TEXT, fontweight="bold")
    return _save(fig, path)


def plot_delay_sweep(delays, sweep, path, dm_ref=0.0349, T_udp=0.020):
    """sweep: {name: {"overshoot": (D, 3), "ITAE": (D, 3), "unstable": (D, 3)}} averaged over joints."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    d_ms = np.asarray(delays) * 1e3
    for i, (name, m) in enumerate(sweep.items()):
        c = color_for(name, i)
        for ax, key in zip(axes, ("ITAE", "overshoot")):
            v = np.nanmean(m[key], axis=1) * (100 if key == "overshoot" else 1)
            v = np.where(np.any(m["unstable"], axis=1), np.nan, v)
            ax.plot(d_ms, v, color=c, label=name, marker="o", ms=4)
    for ax in axes:
        ax.axvline(dm_ref * 1e3, color=MUTED, ls="--", lw=1.2)
        ax.text(dm_ref * 1e3, 0.03, " thesis delay\n margin 34.9 ms", transform=ax.get_xaxis_transform(),
                va="bottom", fontsize=8, color=TEXT_2)
        ax.axvline(T_udp * 1e3, color=MUTED, ls=":", lw=1.2)
        ax.text(T_udp * 1e3, 0.80, " 1 UDP period", transform=ax.get_xaxis_transform(), va="top",
                fontsize=8, color=TEXT_2)
        ax.set_xlabel("in-loop feedback delay [ms]")
    axes[0].set_ylabel("ITAE (mean of 3 joints)")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("overshoot [%] (mean of 3 joints)")
    axes[0].set_title("Tracking error vs delay (line ends where a joint loses stability)")
    axes[1].set_title("Overshoot vs delay")
    axes[0].legend(loc="center left", fontsize=8)
    return _save(fig, path)


def plot_seam_welding(t, x_ref, tracks, path, title="Seam-welding trajectory tracking"):
    """tracks: {name: x_actual (T, 3)} in metres."""
    fig = plt.figure(figsize=(15, 5.2))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.set_facecolor(SURFACE)
    ax3.plot(*(x_ref * 1e3).T, color=MUTED, lw=1.4, ls="--", label="seam reference")
    for i, (name, x) in enumerate(tracks.items()):
        ax3.plot(*(x * 1e3).T, color=color_for(name, i), lw=1.6, label=name)
    ax3.set_xlabel("x [mm]")
    ax3.set_ylabel("y [mm]")
    ax3.set_zlabel("z [mm]")
    # equal metric scaling so millimetre errors are not visually exaggerated
    lo, hi = x_ref.min(axis=0) * 1e3, x_ref.max(axis=0) * 1e3
    half = max(hi - lo) / 2 * 1.1
    for setter, c in zip((ax3.set_xlim, ax3.set_ylim, ax3.set_zlim), (lo + hi) / 2):
        setter(c - half, c + half)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_title("Tool path (weave pattern)")
    ax3.legend(loc="upper left", fontsize=8)

    ax = fig.add_subplot(1, 2, 2)
    for i, (name, x) in enumerate(tracks.items()):
        err = np.linalg.norm(x - x_ref, axis=1) * 1e3
        rms = np.sqrt(np.mean(err**2))
        ax.plot(t, err, color=color_for(name, i), lw=1.4, label=f"{name}  (RMS {rms:.2f} mm)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Cartesian tracking error [mm]")
    ax.set_title("Tracking error")
    ax.legend(loc="center right", fontsize=8)
    fig.suptitle(title, color=TEXT, fontweight="bold")
    return _save(fig, path)
