# CLAUDE.md - Evolutionary & Swarm PID Tuning for Phantom Omni (Geomagic Touch)

## 1. Project Overview & Context
Build a modular Python simulation and optimization project that implements **Particle Swarm Optimization (PSO)** and a **Genetic Algorithm (GA)** to automatically tune the 3-DoF Joint-Space Position and Task-Space Velocity PID controllers of the **Geomagic Touch (Phantom Omni)** haptic robot.

This project directly implements the **"Adaptive Control and Automatic Retuning"** milestone from Section 7.9 of the bachelor's thesis *"Omni Bundle - Haptic Device's Industrial Approach"* (Găvan, 2026).

---

## 2. Mathematical & Physical Robot Specifications

### 2.1 Kinematics (Denavit-Hartenberg Parameters)
Active Joints: 3 ($q_1$: base rotation, $q_2$: shoulder flexion, $q_3$: elbow flexion). Link lengths: $L_1 = L_2 = 0.131\text{ m}$.

| Joint $i$ | $\theta_i$ | $d_i$ (m) | $a_i$ (m) | $\alpha_i$ (rad) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | $q_1$ | 0 | 0 | $-\pi/2$ |
| **2** | $q_2$ | 0 | $L_1 = 0.131$ | 0 |
| **3** | $q_3 - \pi/2$ | 0 | $L_2 = 0.131$ | 0 |

* **Forward Kinematics (FK):**
  * $x = (L_1 \cos(q_2) + L_2 \cos(q_2 + q_3)) \cos(q_1)$
  * $y = (L_1 \cos(q_2) + L_2 \cos(q_2 + q_3)) \sin(q_1)$
  * $z = -L_1 \sin(q_2) - L_2 \sin(q_2 + q_3)$
* **Inverse Kinematics (IK):** Elbow-up configuration (+ sign in $q_3 = \text{atan2}(+\sqrt{1-D^2}, D)$).
* **Workspace Safety Shell:** $0.05\text{ m} \le r = \sqrt{x^2 + y^2 + z^2} \le 0.25\text{ m}$. Max step jump: $0.15\text{ m}$.
* **Damped Least-Squares (DLS) Inverse Jacobian:** $J^\dagger = J^\top (J J^\top + \lambda^2 I)^{-1}$ with $\lambda \in [0.01, 0.05]$.

### 2.2 Decoupled Per-Joint Dynamical Model
Each actuated joint $i \in \{1, 2, 3\}$ is governed by the differential equation:
$$J_i \ddot{q}_i + B_i \dot{q}_i + d_i(t) = \tau_i$$
where $d_i(t)$ represents unmodelled Coulomb/static friction and gravity bias (causing $3^\circ\text{--}8^\circ$ steady-state error under pure PD control).

| Joint $i$ | Inertia $J_i$ ($\text{kg}\cdot\text{m}^2$) | Viscous Damping $B_i$ ($\text{N}\cdot\text{m}\cdot\text{s/rad}$) |
| :--- | :--- | :--- |
| **1 (Base)** | `0.0031` | `0.0089` |
| **2 (Shoulder)** | `0.0022` | `0.0170` |
| **3 (Elbow)** | `0.0009` | `0.0058` |

* **Position Plant (Laplace):** $H_{\text{pos},i}(s) = \frac{1}{s(J_i s + B_i)}$
* **Velocity Plant (Laplace):** $H_{\text{vel},i}(s) = \frac{1}{J_i s + B_i}$
* **Execution & Network Rates:**
  * QUARC Real-Time Control Loop Sample Time: $T_s = 0.001\text{ s}$ ($1\text{ kHz}$).
  * UDP Network Command/Feedback Rate: $T_{\text{udp}} = 0.020\text{ s}$ ($50\text{ Hz}$).
  * Maximum Theoretical Delay Margin: $\tau_{d,\max} = 34.9\text{ ms}$ (derived from $\omega_c = 35\text{ rad/s}$, Phase Margin $\gamma = 70^\circ$).
  * Actuator Torque Saturation: Peak torque bound based on $3.3\text{ N}$ peak / $0.88\text{ N}$ continuous tip force.

### 2.3 Baseline Controller Gains (Benchmarks from Thesis)
Use these exact values as reference baselines in plots and metrics:
* **Frequency-Domain PD Gains (Simulation Baseline):**
  * Joint 1: $K_{p1} = 1.592$, $K_{d1} = 0.0989$
  * Joint 2: $K_{p2} = 1.481$, $K_{d2} = 0.0665$
  * Joint 3: $K_{p3} = 0.5678$, $K_{d3} = 0.0276$
* **Manual QUARC Hardware PID Gains (Joint 1 Reference):**
  * Joint 1: $K_{p1} = 2.7262$, $K_{i1} = 1.2000$, $K_{d1} = 0.0808$

---

## 3. Target Repository Structure
Generate the following clean, modular Python architecture:
```text
omni_pid_optimizer/
├── pyproject.toml
├── README.md
├── config/
│   └── robot_params.yaml       # J_i, B_i, L_1, L_2, Ts, UDP delay, search bounds
├── src/
│   ├── __init__.py
│   ├── kinematics.py           # DH matrices, FK, closed-form IK, Jacobian, DLS inverse
│   ├── dynamics.py             # Discrete 1 kHz Euler/RK4 plant with UDP delay & friction/gravity bias
│   ├── controllers.py          # Discrete PID with anti-windup and derivative low-pass filter
│   ├── objective.py            # Cost functions: ITAE, IAE, ISE, Overshoot, Settling Time, Phase Margin
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── pso.py              # Particle Swarm Optimization (inertia weight, cognitive/social clamp)
│   │   └── ga.py               # Genetic Algorithm (tournament selection, SBX/BLX-alpha crossover, mutation)
│   └── visualizer.py           # Step responses, Bode plots, convergence curves, 3D seam-welding trajectory
├── scripts/
│   ├── tune_joint_pid.py       # CLI to run PSO/GA tuning per joint or full 9-parameter vector
│   └── compare_baselines.py    # Compare Thesis PD vs. Manual QUARC PID vs. PSO-PID vs. GA-PID
└── tests/
    ├── test_kinematics.py      # Verify FK/IK reciprocity and workspace guards
    └── test_dynamics.py        # Verify step response and delay instability at >34.9 ms