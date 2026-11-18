"""Robomimic Square (NutAssemblySquare) with multi-human demonstrations: shared pieces, mirroring can_common.py.

Observation (23-D, the robomimic low-dim set): end-effector position (3) and quaternion (4), gripper finger
positions (2), nut position (3) and quaternion (4), nut-to-end-effector position (3) and quaternion (4).
Action (7-D, OSC_POSE deltas as in the dataset). Success (robosuite): the square nut on the square peg.

    MUJOCO_GL=disable ROBO_SQUARE=<path to square_mh_low_dim_v141.hdf5> <robosuite-venv>/python ...
"""
import json
import os

import h5py
import numpy as np

import lift_common as LC

DATA = os.environ.get("ROBO_SQUARE", "square_mh_low_dim_v141.hdf5")
OPERATORS = LC.OPERATORS
HORIZON = 500
PEG = np.array([0.23, 0.10])        # square peg (robosuite peg1), fixed
HANDLE = 0.054                      # handle site, along the nut's local x axis from its centre
# grasp points: (distance from the nut's centre, direction in the nut's frame, gripper yaw relative to that
# direction) -- the handle, whose bar is radial, and the three walls of the ring, which the fingers close across;
# as the operators do (handle at 0.06--0.07, walls at 0.044--0.05)
GRIPS = [(0.060, 0.0, 0.0), (0.049, np.pi / 2, np.pi / 2), (0.049, -np.pi / 2, np.pi / 2), (0.044, np.pi, np.pi / 2)]
CARRY_Z = 1.00                      # end-effector height while carrying (clears the peg)
HOLD_W = 0.045                      # finger width below which a near handle is held (open 0.08, on the handle 0.03)
REST_Z = 0.83                       # nut height once on the peg
DROP_Z = 0.975                      # nut height from which it is dropped onto the peg (peg top 0.95)


def load():
    f = h5py.File(DATA, "r")
    d = f["data"]
    env_kwargs = json.loads(d.attrs["env_args"])["env_kwargs"]
    out = {}
    for op in OPERATORS:
        demos = []
        for k in [x.decode() for x in f["mask"][op][:]]:
            e = d[k]
            o = e["obs"]
            obs = np.concatenate([o["robot0_eef_pos"][:], o["robot0_eef_quat"][:], o["robot0_gripper_qpos"][:],
                                  o["object"][:]], axis=1)
            demos.append(dict(name=k, obs=obs.astype(np.float32), actions=e["actions"][:].astype(np.float32),
                              states=e["states"][:]))
        out[op] = demos
    return out, env_kwargs


def make_env(env_kwargs):
    import robosuite
    kw = dict(env_kwargs)
    kw.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
    env = robosuite.make("NutAssemblySquare", **kw)
    env.hard_reset = False
    return env


def obs_vec(o):
    return np.concatenate([o["robot0_eef_pos"], o["robot0_eef_quat"], o["robot0_gripper_qpos"],
                           o["object-state"]]).astype(np.float32)


def _yaw(q):
    x, y, z, w = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a, period):
    return (a + period / 2) % period - period / 2


def _q_down(psi):
    """Gripper pointing down with yaw psi (xyzw): Rz(psi) * Rx(pi)."""
    return np.array([np.cos(psi / 2), np.sin(psi / 2), 0.0, 0.0])


def eef_yaw(q):
    """Yaw of a down-pointing gripper quaternion (inverse of _q_down)."""
    return 2 * np.arctan2(q[1], q[0])


def on_peg(n):
    return bool(np.linalg.norm(n[:2] - PEG) < 0.03 and n[2] < REST_Z + 0.02)


def labeler(x):
    """Scripted pick-and-place law on the 23-D observation (Markovian): align above the nut's handle with the
    gripper yawed along it and open, descend, close, lift to carrying height while turning the nut to the peg's
    orientation, move the nut over the peg, lower it and release; move up once the nut is on the peg."""
    X = np.atleast_2d(x)
    out = np.zeros((len(X), 7), dtype=np.float32)
    for i, s in enumerate(X):
        p, q, g, n, nq = s[0:3], s[3:7], s[7:9], s[9:12], s[12:16]
        th = _yaw(nq)
        width = g[0] - g[1]
        psi = eef_yaw(q)
        dz = p[2] - n[2]
        pts = [n[:2] + r * np.array([np.cos(th + d), np.sin(th + d)]) for r, d, _ in GRIPS]
        dmin = min(np.linalg.norm(p[:2] - h) for h in pts)
        holding = dmin < 0.04 and abs(dz) < 0.05 and 0.02 < width < HOLD_W     # the nut shifts as the fingers close
        if on_peg(n) and not holding:
            tgt, grip, yaw = np.array([p[0], p[1], CARRY_Z]), -1.0, psi            # done
        elif holding:
            # turn the nut to the nearest multiple of 90 degrees (the peg's orientation)
            dth = _wrap(th, np.pi / 2)
            yaw = psi - dth
            off = p[:2] - n[:2]
            err = np.linalg.norm(n[:2] - PEG)
            over = err < 0.004 and abs(dth) < np.radians(3)
            if on_peg(n) or (err < 0.004 and n[2] < DROP_Z):
                tgt, grip = p.copy(), -1.0                                          # just above the peg: drop it
            elif over:
                tgt, grip = np.array([p[0], p[1], DROP_Z - 0.01 + (p[2] - n[2])]), 1.0   # lower to just above the peg
            elif p[2] < CARRY_Z - 0.04 and err > 0.03:
                tgt, grip = np.array([p[0], p[1], CARRY_Z]), 1.0                   # lift first
            else:
                tgt, grip = np.array([PEG[0] + off[0], PEG[1] + off[1], CARRY_Z]), 1.0
                if err < 0.03:                                                      # fine alignment: higher gain
                    tgt[:2] = p[:2] + (tgt[:2] - p[:2]) * 3.0
        else:
            # grasp point: the one on the robot's side of the nut (the arm is near full extension over the peg),
            # with a gripper yaw near zero as a tie-breaker
            cost = [(1 + np.cos(th + d)) / 2 + 0.3 * abs(_wrap(th + d + y, np.pi)) / (np.pi / 2) for _, d, y in GRIPS]
            k = int(np.argmin(cost))
            h = pts[k]
            yaw = _wrap(th + GRIPS[k][1] + GRIPS[k][2], np.pi)
            dxy = np.linalg.norm(p[:2] - h)
            near = dxy < 0.015 and abs(dz) < 0.012                 # the operators close level with the nut
            yaw_ok = abs(_wrap(yaw - psi, 2 * np.pi)) < np.radians(8)
            if width < 0.015:
                tgt, grip = np.array([p[0], p[1], n[2] + 0.06]), -1.0              # closed on nothing: open, retry
            elif near:
                tgt, grip = p.copy(), 1.0                                          # close in place
            elif dxy < 0.012 and yaw_ok:
                tgt, grip = np.array([h[0], h[1], n[2] - 0.002]), -1.0             # descend onto the nut
            else:
                tgt, grip = np.array([h[0], h[1], max(n[2] + 0.10, p[2]) if dxy > 0.05 else n[2] + 0.10]), -1.0
        dp = np.clip((tgt - p) / 0.05 * 1.5, -1, 1)
        dr = np.clip(LC._quat_err_xyzw(q, _q_down(yaw)) / 0.5 * 1.5, -1, 1)
        out[i, :3], out[i, 3:6], out[i, 6] = dp, dr, grip
    return out[0] if np.ndim(x) == 1 else out


def success(env):
    return bool(env._check_success())


def rollout(env, pol, horizon=HORIZON, state=None):
    """Closed loop from a saved simulator state for `horizon` steps or until success holds for 10 steps."""
    env.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    env.robots[0].controller.update(force=True)
    env.robots[0].controller.reset_goal()
    x = obs_vec(env._get_observations(force_update=True))
    if hasattr(pol, "reset"):
        pol.reset()
    X, U, held = [], [], 0
    for _ in range(horizon):
        a = np.clip(np.asarray(pol(x), dtype=np.float64).ravel(), -1, 1)
        X.append(x); U.append(a)
        o, _, _, _ = env.step(a)
        x = obs_vec(o)
        held = held + 1 if success(env) else 0
        if held >= 10:
            break
    X.append(x)
    return np.array(X), np.array(U), held >= 10
