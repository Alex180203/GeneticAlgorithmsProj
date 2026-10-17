import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import kinematics as kin  # noqa: E402

L = 0.131


def random_reachable_points(n, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        q = np.array([rng.uniform(-1.2, 1.2), rng.uniform(-0.6, 0.9), rng.uniform(0.3, 2.4)])
        p = kin.forward_kinematics(q)
        if 0.06 <= np.linalg.norm(p) <= 0.24:
            pts.append(p)
    return np.array(pts)


def test_fk_ik_reciprocity():
    for p in random_reachable_points(200):
        q = kin.inverse_kinematics(p)
        np.testing.assert_allclose(kin.forward_kinematics(q), p, atol=1e-10)


def test_ik_is_elbow_up():
    q = kin.inverse_kinematics([0.15, 0.02, -0.05])
    assert q[2] > 0      # + sign in q3 = atan2(+sqrt(1 - D^2), D)


def test_fk_known_poses():
    np.testing.assert_allclose(kin.forward_kinematics([0, 0, 0]), [2 * L, 0, 0], atol=1e-12)
    np.testing.assert_allclose(kin.forward_kinematics([np.pi / 2, 0, np.pi / 2]), [0, L, -L], atol=1e-12)


def test_dh_chain_matches_closed_form():
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.uniform(-1.5, 1.5, 3)
        T = kin.dh_forward(kin.kinematic_to_encoder(q))
        np.testing.assert_allclose(T[:3, 3], kin.forward_kinematics(q), atol=1e-12)


@pytest.mark.parametrize("p", [[0.02, 0.0, 0.0], [0.0, 0.03, -0.02], [0.26, 0.0, 0.0], [0.2, 0.16, 0.0]])
def test_workspace_shell_rejects(p):
    with pytest.raises(kin.WorkspaceError):
        kin.inverse_kinematics(p)


def test_workspace_max_step_jump():
    kin.inverse_kinematics([0.15, 0.0, 0.0], p_prev=[0.15, 0.10, 0.0])          # 0.10 m: ok
    with pytest.raises(kin.WorkspaceError):
        kin.inverse_kinematics([0.15, 0.08, 0.0], p_prev=[0.15, -0.08, 0.0])   # 0.16 m: too far


def test_jacobian_matches_finite_difference():
    q = np.array([0.3, 0.2, 1.1])
    J = kin.jacobian(q)
    eps = 1e-7
    J_fd = np.column_stack([(kin.forward_kinematics(q + eps * e) - kin.forward_kinematics(q - eps * e)) / (2 * eps)
                            for e in np.eye(3)])
    np.testing.assert_allclose(J, J_fd, atol=1e-7)


def test_dls_inverse_close_to_inverse_away_from_singularity():
    J = kin.jacobian([0.3, 0.2, 1.1])
    assert kin.dls_lambda(J) == pytest.approx(0.01)
    np.testing.assert_allclose(kin.dls_inverse(J) @ J, np.eye(3), atol=0.05)


def test_dls_bounded_at_singularity():
    J = kin.jacobian([0.0, 0.0, 0.0])            # arm fully stretched: rank deficient
    lam = kin.dls_lambda(J)
    assert 0.01 <= lam <= 0.05 and lam > 0.04
    assert np.all(np.isfinite(kin.dls_inverse(J)))
    assert np.linalg.norm(kin.dls_inverse(J)) < 1.0 / 0.01


def test_seam_trajectory_inside_workspace():
    t, x, xd, q = kin.seam_welding_trajectory(duration=1.0, hold=0.2)
    r = np.linalg.norm(x, axis=1)
    assert r.min() >= 0.05 and r.max() <= 0.25
    np.testing.assert_allclose(kin.forward_kinematics(q), x, atol=1e-10)
