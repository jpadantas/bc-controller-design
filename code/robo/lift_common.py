"""Robomimic Lift with multi-human demonstrations: shared pieces for the human-demonstration study.

Run with the separate robosuite environment (robosuite 1.4.1, mujoco 3.2.7, ASCII paths):
    MUJOCO_GL=disable <robosuite-venv>/python lift_study.py ...   (ROBO_DATA=<path to lift_mh_low_dim_v141.hdf5>)

Observation (19-D, the robomimic low-dim set): end-effector position (3) and quaternion (4), gripper finger
positions (2), cube position (3), cube quaternion (4), gripper-to-cube vector (3). Action (7-D, OSC_POSE
deltas as in the dataset): position (3), rotation (3), gripper (1; -1 open, +1 close).
"""
import json
import os

import h5py
import numpy as np

DATA = os.environ.get("ROBO_DATA", "lift_mh_low_dim_v141.hdf5")
OPERATORS = ["better_operator_1", "better_operator_2", "okay_operator_1", "okay_operator_2",
             "worse_operator_1", "worse_operator_2"]
HORIZON = 250
TABLE_Z = 0.80          # table top height in Lift (cube rests at about 0.83)
LIFT_Z = 0.04           # success: cube raised this far above its resting height


def load():
    """Per operator: list of demos, each a dict with obs (T,19), actions (T,7), states (T,S) and the
    environment kwargs."""
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
    return robosuite.make("Lift", **kw)


def obs_vec(o):
    return np.concatenate([o["robot0_eef_pos"], o["robot0_eef_quat"], o["robot0_gripper_qpos"],
                           o["object-state"]]).astype(np.float32)


def _quat_err_xyzw(q, qt):
    """Rotation vector taking orientation q to qt (both xyzw)."""
    def conj(a):
        return np.array([-a[0], -a[1], -a[2], a[3]])

    def mul(a, b):
        x1, y1, z1, w1 = a
        x2, y2, z2, w2 = b
        return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
    e = mul(qt, conj(q))
    if e[3] < 0:
        e = -e
    v, w = e[:3], np.clip(e[3], -1, 1)
    ang = 2 * np.arccos(w)
    n = np.linalg.norm(v)
    return np.zeros(3) if n < 1e-9 else v / n * ang


Q_DOWN = np.array([1.0, 0.0, 0.0, 0.0])   # gripper pointing down (xyzw), as at reset


def labeler(x):
    """Scripted pick-and-lift law on the 19-D observation (Markovian): align above the cube with the gripper
    open, descend, close, and lift once the fingers have closed on the cube; keep the gripper pointing down.
    Accepts (19,) or (n, 19)."""
    X = np.atleast_2d(x)
    out = np.zeros((len(X), 7), dtype=np.float32)
    for i, s in enumerate(X):
        p, q, g, c = s[0:3], s[3:7], s[7:9], s[9:12]
        width = g[0] - g[1]
        dxy = np.linalg.norm(p[:2] - c[:2])
        dz = p[2] - c[2]
        near = dxy < 0.02 and dz < 0.02
        holding = near and width < 0.05
        if holding:
            tgt, grip = np.array([c[0], c[1], c[2] + 0.25]), 1.0
            tgt[:2] = p[:2]
        elif near:
            tgt, grip = p.copy(), 1.0                                  # close in place
        elif dxy < 0.02:
            tgt, grip = np.array([c[0], c[1], c[2] - 0.005]), -1.0     # descend
        else:
            tgt, grip = np.array([c[0], c[1], max(c[2] + 0.08, p[2] if dxy > 0.05 else c[2] + 0.08)]), -1.0
        dp = np.clip((tgt - p) / 0.05 * 1.5, -1, 1)
        dr = np.clip(_quat_err_xyzw(q, Q_DOWN) / 0.5 * 1.5, -1, 1)
        out[i, :3], out[i, 3:6], out[i, 6] = dp, dr, grip
    return out[0] if np.ndim(x) == 1 else out


def success(x):
    return x[11] > TABLE_Z + 0.03 + LIFT_Z


def perturb(env, rng, joint_sd=0.0, cube_xy=None):
    """Off-nominal start: perturb the arm joints by N(0, joint_sd) and place the cube at cube_xy (offset from
    the table center); the demonstrations start with the arm at its reset pose and the cube within 3 cm."""
    sim = env.sim
    robot = env.robots[0]
    if joint_sd > 0:
        qpos = sim.data.qpos.copy()
        idx = robot._ref_joint_pos_indexes
        qpos[idx] = qpos[idx] + rng.normal(0, joint_sd, len(idx))
        sim.data.qpos[:] = qpos
    if cube_xy is not None:
        jid = sim.model.joint_name2id(env.cube.joints[0])
        adr = sim.model.jnt_qposadr[jid]
        sim.data.qpos[adr:adr + 2] = cube_xy
    sim.forward()
    robot.controller.update(force=True)
    robot.controller.reset_goal()
    return obs_vec(env._get_observations(force_update=True))


def rollout(env, pol, x0=None, horizon=HORIZON, state=None):
    """Closed loop for `horizon` steps (or until the cube is lifted and held for 10 steps). Returns the
    observation trajectory (T,19), actions (T,7) and success."""
    if state is not None:
        env.reset()
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        env.robots[0].controller.update(force=True)
        env.robots[0].controller.reset_goal()
        x = obs_vec(env._get_observations(force_update=True))
    else:
        x = x0
    if hasattr(pol, "reset"):   # stateful policies (action chunking) start each rollout afresh
        pol.reset()
    X, U, held = [], [], 0
    for _ in range(horizon):
        a = np.clip(np.asarray(pol(x), dtype=np.float64).ravel(), -1, 1)
        X.append(x); U.append(a)
        o, _, _, _ = env.step(a)
        x = obs_vec(o)
        held = held + 1 if success(x) else 0
        if held >= 10:
            break
    X.append(x)
    return np.array(X), np.array(U), held >= 10
