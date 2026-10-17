"""Kinematics of the 3-DoF Phantom Omni: DH chain, FK, closed-form IK, Jacobian, DLS.

Joint convention
----------------
The closed-form FK below is written in *kinematic* joint angles (q1, q2, q3).
The DH table lists theta_3 = q3_enc - pi/2, where q3_enc is the raw elbow encoder
angle; i.e. q3 = q3_enc - pi/2. ``dh_forward`` takes encoder angles and
``encoder_to_kinematic`` / ``kinematic_to_encoder`` convert between the two.
"""

import numpy as np

L1_DEFAULT = 0.131
L2_DEFAULT = 0.131


class WorkspaceError(ValueError):
    """Raised when a Cartesian target violates the workspace safety shell."""


def dh_matrix(theta, d, a, alpha):
    """Standard Denavit-Hartenberg homogeneous transform."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def encoder_to_kinematic(q_enc):
    q = np.array(q_enc, dtype=float)
    q[..., 2] -= np.pi / 2
    return q


def kinematic_to_encoder(q):
    q_enc = np.array(q, dtype=float)
    q_enc[..., 2] += np.pi / 2
    return q_enc


def dh_forward(q_enc, L1=L1_DEFAULT, L2=L2_DEFAULT):
    """Tool-frame transform from the DH table (takes encoder angles)."""
    q1, q2, q3 = q_enc
    T = dh_matrix(q1, 0.0, 0.0, -np.pi / 2)
    T = T @ dh_matrix(q2, 0.0, L1, 0.0)
    T = T @ dh_matrix(q3 - np.pi / 2, 0.0, L2, 0.0)
    return T


def forward_kinematics(q, L1=L1_DEFAULT, L2=L2_DEFAULT):
    """Closed-form FK. ``q`` has shape (..., 3); returns (..., 3) positions [m]."""
    q = np.asarray(q, dtype=float)
    q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2]
    rho = L1 * np.cos(q2) + L2 * np.cos(q2 + q3)
    x = rho * np.cos(q1)
    y = rho * np.sin(q1)
    z = -L1 * np.sin(q2) - L2 * np.sin(q2 + q3)
    return np.stack([x, y, z], axis=-1)


def check_workspace(p, p_prev=None, r_min=0.05, r_max=0.25, max_step=0.15):
    """Validate a target against the safety shell and the max-jump limit."""
    p = np.asarray(p, dtype=float)
    r = np.linalg.norm(p)
    if not (r_min <= r <= r_max):
        raise WorkspaceError(f"target radius {r:.4f} m outside [{r_min}, {r_max}] m")
    if p_prev is not None:
        jump = np.linalg.norm(p - np.asarray(p_prev, dtype=float))
        if jump > max_step:
            raise WorkspaceError(f"step jump {jump:.4f} m exceeds {max_step} m")


def inverse_kinematics(p, L1=L1_DEFAULT, L2=L2_DEFAULT, p_prev=None, check=True, **shell):
    """Closed-form elbow-up IK: q3 = atan2(+sqrt(1 - D^2), D)."""
    p = np.asarray(p, dtype=float)
    if check:
        check_workspace(p, p_prev, **shell)
    x, y, z = p
    q1 = np.arctan2(y, x)
    rho = np.hypot(x, y)
    h = -z                                   # FK: z = -(L1 s2 + L2 s23)
    D = (rho**2 + h**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(D) > 1.0:
        raise WorkspaceError(f"target unreachable (D = {D:.4f})")
    q3 = np.arctan2(+np.sqrt(1.0 - D**2), D)
    q2 = np.arctan2(h, rho) - np.arctan2(L2 * np.sin(q3), L1 + L2 * np.cos(q3))
    return np.array([q1, q2, q3])


def jacobian(q, L1=L1_DEFAULT, L2=L2_DEFAULT):
    """Analytic 3x3 linear-velocity Jacobian of ``forward_kinematics``."""
    q1, q2, q3 = q
    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s23, c23 = np.sin(q2 + q3), np.cos(q2 + q3)
    rho = L1 * c2 + L2 * c23
    drho_dq2 = -L1 * s2 - L2 * s23
    drho_dq3 = -L2 * s23
    return np.array([
        [-rho * s1, drho_dq2 * c1, drho_dq3 * c1],
        [rho * c1, drho_dq2 * s1, drho_dq3 * s1],
        [0.0, -L1 * c2 - L2 * c23, -L2 * c23],
    ])


def dls_lambda(J, lam_range=(0.01, 0.05), sigma_threshold=0.05):
    """Adaptive damping: grows from lam_min to lam_max as sigma_min -> 0."""
    lam_min, lam_max = lam_range
    sigma_min = np.linalg.svd(J, compute_uv=False)[-1]
    if sigma_min >= sigma_threshold:
        return lam_min
    return lam_min + (lam_max - lam_min) * (1.0 - sigma_min / sigma_threshold) ** 2


def dls_inverse(J, lam=None, lam_range=(0.01, 0.05)):
    """Damped least-squares inverse J^T (J J^T + lambda^2 I)^-1."""
    if lam is None:
        lam = dls_lambda(J, lam_range)
    lam = float(np.clip(lam, *lam_range))
    return J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(J.shape[0]))


def seam_welding_trajectory(duration=4.0, Ts=0.001, center=(0.15, 0.0, -0.04), seam_length=0.12,
                            weave_amp=0.003, weave_hz=2.0, hold=0.0, L1=L1_DEFAULT, L2=L2_DEFAULT, **shell):
    """Straight weld seam along y with a sinusoidal weave across it (along x).

    Quintic time scaling gives zero velocity/acceleration at both ends. Every
    sample is checked against the workspace shell and converted with IK.
    ``hold`` seconds of dwell at the seam start are prepended (the arm settles
    on its start pose before the torch moves). Returns t, x_ref, xd_ref, q_ref
    (each (T,3) except t); t is negative during the dwell.
    """
    t = (np.arange(int(round((duration + hold) / Ts))) * Ts) - hold
    s = np.clip(t / duration, 0.0, 1.0)
    s5 = 10 * s**3 - 15 * s**4 + 6 * s**5
    cx, cy, cz = center
    weave = weave_amp * np.sin(2 * np.pi * weave_hz * np.maximum(t, 0)) * np.sin(np.pi * s)   # fades in/out
    x_ref = np.column_stack([cx + weave, cy - seam_length / 2 + seam_length * s5, np.full_like(t, cz)])
    xd_ref = np.gradient(x_ref, Ts, axis=0)
    q_ref = np.empty_like(x_ref)
    prev = None
    for k, p in enumerate(x_ref):
        q_ref[k] = inverse_kinematics(p, L1, L2, p_prev=prev, **shell)
        prev = p
    return t, x_ref, xd_ref, q_ref
