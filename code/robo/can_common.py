"""Robomimic Can (PickPlaceCan) with multi-human demonstrations: shared pieces, mirroring lift_common.py.

Observation (23-D, the robomimic low-dim set): end-effector position (3) and quaternion (4), gripper finger
positions (2), can position (3) and quaternion (4), can-to-end-effector position (3) and quaternion (4).
Action (7-D, OSC_POSE deltas as in the dataset). Success (robosuite): the can inside the target bin and the gripper
away from it.

    MUJOCO_GL=disable ROBO_CAN=<path to can_mh_low_dim_v141.hdf5> <robosuite-venv>/python can_study.py ...
"""
import json
import os

import h5py
import numpy as np

import lift_common as LC

DATA = os.environ.get("ROBO_CAN", "can_mh_low_dim_v141.hdf5")
OPERATORS = LC.OPERATORS
HORIZON = 400
TARGET = np.array([0.1975, 0.4025])          # centre of the can's bin (robosuite target_bin_placements[3])
BIN_LO, BIN_HI = np.array([0.1, 0.28]), np.array([0.295, 0.525])
CARRY_Z = 1.05                               # end-effector height while carrying
DROP_Z = 1.0                                 # release height over the bin (the operators drop from about 1 m)
HOLD_W = 0.065                               # finger width below which a near can is held (open 0.08, can 0.05)


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
    env = robosuite.make("PickPlaceCan", **kw)
    env.hard_reset = False      # reset without rebuilding the model (same physics, far faster)
    return env


def obs_vec(o):
    return np.concatenate([o["robot0_eef_pos"], o["robot0_eef_quat"], o["robot0_gripper_qpos"],
                           o["object-state"]]).astype(np.float32)


def in_bin(c):
    return bool(np.all(c[:2] > BIN_LO) and np.all(c[:2] < BIN_HI) and 0.8 < c[2] < 0.9)


def labeler(x):
    """Scripted pick-and-place law on the 23-D observation (Markovian): align above the can with the gripper
    open, descend, close, lift to carrying height, move over the bin, release, and move up and away once the can
    is in the bin; keep the gripper pointing down. Accepts (23,) or (n, 23)."""
    X = np.atleast_2d(x)
    out = np.zeros((len(X), 7), dtype=np.float32)
    for i, s in enumerate(X):
        p, q, g, c = s[0:3], s[3:7], s[7:9], s[9:12]
        width = g[0] - g[1]
        dxy = np.linalg.norm(p[:2] - c[:2])
        dz = p[2] - c[2]
        near = dxy < 0.02 and abs(dz) < 0.03
        holding = near and width < HOLD_W
        if in_bin(c) and not holding:
            tgt, grip = np.array([p[0], p[1], CARRY_Z + 0.05]), -1.0        # done: open and move away
        elif holding:
            over_bin = bool(np.all(c[:2] > BIN_LO + 0.05) and np.all(c[:2] < BIN_HI - 0.05))
            if over_bin and p[2] > DROP_Z - 0.05:
                tgt, grip = p.copy(), -1.0                                   # over the bin: release
            elif p[2] < CARRY_Z - 0.05 and not over_bin:
                tgt, grip = np.array([p[0], p[1], CARRY_Z]), 1.0             # lift first
            else:
                tgt, grip = np.array([TARGET[0], TARGET[1], CARRY_Z]), 1.0   # then travel over the bin
        elif near:
            tgt, grip = p.copy(), 1.0                                        # close in place
        elif dxy < 0.02:
            tgt, grip = np.array([c[0], c[1], c[2]]), -1.0                  # descend onto the can
        else:
            hover = c[2] + 0.12
            tgt = np.array([c[0], c[1], max(hover, p[2]) if dxy > 0.05 else hover])
            grip = -1.0
        dp = np.clip((tgt - p) / 0.05 * 1.5, -1, 1)
        dr = np.clip(LC._quat_err_xyzw(q, LC.Q_DOWN) / 0.5 * 1.5, -1, 1)
        out[i, :3], out[i, 3:6], out[i, 6] = dp, dr, grip
    return out[0] if np.ndim(x) == 1 else out


def success(env):
    return bool(env._check_success())


def perturb(env, rng, joint_sd=0.0):
    """Off-nominal start: perturb the arm joints by N(0, joint_sd); the can keeps its sampled placement."""
    sim = env.sim
    robot = env.robots[0]
    if joint_sd > 0:
        qpos = sim.data.qpos.copy()
        idx = robot._ref_joint_pos_indexes
        qpos[idx] = qpos[idx] + rng.normal(0, joint_sd, len(idx))
        sim.data.qpos[:] = qpos
    sim.forward()
    robot.controller.update(force=True)
    robot.controller.reset_goal()
    return obs_vec(env._get_observations(force_update=True))


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
