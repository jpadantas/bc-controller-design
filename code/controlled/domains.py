"""Controlled domains for the v3 study (demonstrator with its own style, engineered labeler).

Every domain exposes, on batches of FULL states x (n, nx):
    step(x, u)            one control step of the true dynamics (u in [-1, 1]^m)
    obs(x)                policy input
    error(x)              distance to the task goal (the support coordinate)
    demonstrator(o)       Markovian "human" policy on observations: its own style
                          (preferred offset / gains / structure), defined only
                          near nominal in spirit -- its data come from nominal starts
    labeler(o)            engineered controller (different style, large region of attraction)
    nominal_starts(n,rng) starts inside the demonstrations' support
    box_states(n,rng,r)   full-state samples with error in [r, r_max] and every other
                          coordinate over its physical range
    probe(radius, n, rng) probe starts at a given error radius (full-state perturbations)
Demonstrations are rollouts of the demonstrator from nominal starts with small
process noise; there is no lag or hidden drift (the demonstrator is Markovian).
"""
import numpy as np

G = 9.81


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Quadrotor 3D: hold a station (formation slot) -- 12-D state, 13-D observation
# ---------------------------------------------------------------------------
class Quad3D:
    name = "quad3d"
    nx, nobs, nu = 12, 13, 4
    dt, horizon = 0.02, 300              # 6 s probes
    demo_T, demo_eps = 300, 30
    alpha_max = np.array([30.0, 30.0, 8.0])   # rad/s^2 per unit torque command
    drag = 0.15
    r_nominal = 1.5                      # support radius of the demonstrations (m)
    r_max = 12.0
    radii = [0.0, 2.0, 4.0, 7.0, 10.0]
    success = 1.5                        # back inside the nominal set (the demonstrator's slot is 0.75 m away)
    # demonstrator style: holds a slot offset and a yaw angle, LQR-like smooth gains
    demo_offset = np.array([-0.6, 0.4, 0.2])
    demo_yaw = 0.35

    # state: p(3) v(3) phi theta psi (3) omega(3)
    @staticmethod
    def obs(x):
        return np.column_stack([x[:, 0:6], x[:, 6:8], np.sin(x[:, 8]), np.cos(x[:, 8]), x[:, 9:12]])

    @staticmethod
    def error(x):
        return np.linalg.norm(x[:, 0:3], axis=1)

    @staticmethod
    def obs_error(o):
        return np.linalg.norm(o[:, 0:3], axis=1)

    def step(self, x, u):
        u = np.clip(u, -1.0, 1.0)
        p, v = x[:, 0:3], x[:, 3:6]
        ph, th, ps = x[:, 6], x[:, 7], x[:, 8]
        w = x[:, 9:12]
        T = G * (1.0 + u[:, 0])
        bz = np.column_stack([np.cos(ps) * np.sin(th) * np.cos(ph) + np.sin(ps) * np.sin(ph),
                              np.sin(ps) * np.sin(th) * np.cos(ph) - np.cos(ps) * np.sin(ph),
                              np.cos(th) * np.cos(ph)])
        acc = bz * T[:, None] - np.array([0.0, 0.0, G]) - self.drag * v
        wdot = self.alpha_max * u[:, 1:4] - 2.0 * w * np.array([0.3, 0.3, 0.5])  # rotational damping
        pq, qq, rq = w[:, 0], w[:, 1], w[:, 2]
        cth = np.clip(np.cos(th), 0.05, None) * np.sign(np.cos(th) + 1e-9)
        phd = pq + (qq * np.sin(ph) + rq * np.cos(ph)) * np.tan(th)
        thd = qq * np.cos(ph) - rq * np.sin(ph)
        psd = (qq * np.sin(ph) + rq * np.cos(ph)) / cth
        dt = self.dt
        vn = v + acc * dt
        out = np.empty_like(x)
        out[:, 0:3] = p + vn * dt
        out[:, 3:6] = vn
        out[:, 6] = wrap(ph + phd * dt)
        out[:, 7] = np.clip(th + thd * dt, -1.45, 1.45)
        out[:, 8] = wrap(ps + psd * dt)
        out[:, 9:12] = w + wdot * dt
        return out

    @staticmethod
    def _unpack(o):
        return o[:, 0:3], o[:, 3:6], o[:, 6], o[:, 7], np.arctan2(o[:, 8], o[:, 9]), o[:, 10:13]

    def _cascade(self, o, offset, yaw_ref, kp, kd, amax, tilt_max, k_att, k_yaw, k_rate):
        p, v, ph, th, ps, w = self._unpack(o)
        e = p - offset
        a = np.clip(-kp * e - kd * v, -amax, amax)
        az = a[:, 2] + G
        axb = np.cos(ps) * a[:, 0] + np.sin(ps) * a[:, 1]
        ayb = -np.sin(ps) * a[:, 0] + np.cos(ps) * a[:, 1]
        th_d = np.clip(np.arctan2(axb, az), -tilt_max, tilt_max)
        ph_d = np.clip(np.arctan2(-ayb, np.sqrt(axb ** 2 + az ** 2)), -tilt_max, tilt_max)
        tilt = np.clip(np.cos(ph) * np.cos(th), 0.3, 1.0)
        u0 = az / (G * tilt) - 1.0
        w_d = np.column_stack([k_att * (ph_d - ph), k_att * (th_d - th), k_yaw * wrap(yaw_ref - ps)])
        tau = k_rate * (w_d - w) / self.alpha_max
        return np.clip(np.column_stack([u0, tau]), -1.0, 1.0)

    def labeler(self, o):
        """Engineered station-keeping law: aggressive cascaded PD to the nominal slot, yaw 0."""
        return self._cascade(o, np.zeros(3), 0.0, kp=1.2, kd=1.6, amax=6.0, tilt_max=0.7,
                             k_att=6.0, k_yaw=2.0, k_rate=8.0)

    def demonstrator(self, o):
        """'Pilot' style: own slot offset and yaw, softer and more damped, smaller tilts."""
        return self._cascade(o, self.demo_offset, self.demo_yaw, kp=0.6, kd=1.4, amax=2.5,
                             tilt_max=0.3, k_att=4.0, k_yaw=1.0, k_rate=6.0)

    def nominal_starts(self, n, rng):
        x = np.zeros((n, 12))
        x[:, 0:3] = self.demo_offset + rng.normal(0, 0.35, (n, 3))
        x[:, 3:6] = rng.normal(0, 0.3, (n, 3))
        x[:, 6:8] = rng.normal(0, np.radians(4), (n, 2))
        x[:, 8] = self.demo_yaw + rng.normal(0, 0.1, n)
        x[:, 9:12] = rng.normal(0, 0.2, (n, 3))
        return x

    def process_noise(self, x, rng):
        x = x.copy()
        x[:, 3:6] += rng.normal(0, 0.03, (len(x), 3))
        x[:, 9:12] += rng.normal(0, 0.05, (len(x), 3))
        return x

    def _random_full(self, n, rng, r_lo, r_hi):
        x = np.zeros((n, 12))
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        x[:, 0:3] = d * rng.uniform(r_lo, r_hi, (n, 1))
        x[:, 3:6] = rng.uniform(-4.0, 4.0, (n, 3))
        x[:, 6:8] = rng.uniform(-np.radians(70), np.radians(70), (n, 2))
        x[:, 8] = rng.uniform(-np.pi, np.pi, n)
        x[:, 9:12] = rng.uniform(-2.0, 2.0, (n, 3))
        return x

    def box_states(self, n, rng, r_in):
        return self._random_full(n, rng, r_in, self.r_max)

    def box_partial(self, n, rng, r_in, demo_x):
        """Ablation: far positions, every other coordinate copied from logged frames."""
        x = demo_x[rng.integers(0, len(demo_x), n)].copy()
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        x[:, 0:3] = d * rng.uniform(r_in, self.r_max, (n, 1))
        return x

    def probe(self, radius, n, rng):
        x = self._random_full(n, rng, radius, radius)
        x[:, 3:6] = rng.uniform(-2.0, 2.0, (n, 3))
        x[:, 6:8] = rng.uniform(-np.radians(40), np.radians(40), (n, 2))
        x[:, 9:12] = rng.uniform(-1.0, 1.0, (n, 3))
        if radius == 0.0:
            return self.nominal_starts(n, rng)
        return x

    def diverged(self, x):
        return (np.abs(x[:, 6]) > np.radians(100)) | (np.abs(x[:, 7]) > np.radians(85)) | (self.error(x) > 40.0)


# ---------------------------------------------------------------------------
# Pendulum (Gymnasium Pendulum-v1 dynamics): smooth swing-up labeler, leaning demonstrator
# ---------------------------------------------------------------------------
class Pendulum:
    name = "pendulum"
    nx, nobs, nu = 2, 3, 1
    dt, horizon = 0.05, 150
    demo_T, demo_eps = 200, 30
    MAX_SPEED, MAX_TORQUE, G_, L = 8.0, 2.0, 10.0, 1.0
    r_nominal = np.radians(20)
    r_max = np.pi
    radii = [0.0, np.radians(30), np.radians(60), np.radians(100), np.radians(140), np.radians(175)]
    success = np.radians(10)
    demo_lean = np.radians(5.0)

    @staticmethod
    def obs(x):
        return np.column_stack([np.cos(x[:, 0]), np.sin(x[:, 0]), x[:, 1]])

    @staticmethod
    def error(x):
        return np.abs(wrap(x[:, 0]))

    @staticmethod
    def obs_error(o):
        return np.abs(np.arctan2(o[:, 1], o[:, 0]))

    def step(self, x, u):
        u = np.clip(u[:, 0], -1, 1) * self.MAX_TORQUE
        th, thd = x[:, 0], x[:, 1]
        nthd = np.clip(thd + (3 * self.G_ / (2 * self.L) * np.sin(th) + 3.0 * u) * self.dt, -self.MAX_SPEED,
                       self.MAX_SPEED)
        return np.column_stack([th + nthd * self.dt, nthd])

    def labeler(self, o):
        """Energy pumping far from upright, PD near it, blended SMOOTHLY (Lipschitz)."""
        th = np.arctan2(o[:, 1], o[:, 0])
        thd = o[:, 2]
        e = 0.5 * thd ** 2 + 15.0 * np.cos(th)
        swing = 0.6 * thd * (15.0 - e) / self.MAX_TORQUE
        pd = (-10.0 * th - 3.0 * thd) / self.MAX_TORQUE
        w = 1.0 / (1.0 + np.exp((np.abs(th) - 0.7) / 0.08))
        return np.clip(w * pd + (1 - w) * swing, -1, 1)[:, None]

    def demonstrator(self, o):
        """'Human' balancing: leans 5 deg, soft and lightly damped."""
        th = np.arctan2(o[:, 1], o[:, 0])
        thd = o[:, 2]
        hold = -5.0 * np.sin(self.demo_lean)  # gravity compensation at the lean angle (in torque units /3)
        u = (hold - 7.0 * (th - self.demo_lean) - 1.2 * thd) / self.MAX_TORQUE
        return np.clip(u, -1, 1)[:, None]

    def nominal_starts(self, n, rng):
        return np.column_stack([self.demo_lean + rng.normal(0, np.radians(5), n), rng.normal(0, 0.3, n)])

    def process_noise(self, x, rng):
        x = x.copy()
        x[:, 1] += rng.normal(0, 0.05, len(x))
        return x

    def box_states(self, n, rng, r_in):
        th = rng.uniform(r_in, self.r_max, n) * rng.choice([-1.0, 1.0], n)
        return np.column_stack([th, rng.uniform(-self.MAX_SPEED, self.MAX_SPEED, n)])

    def box_partial(self, n, rng, r_in, demo_x):
        x = demo_x[rng.integers(0, len(demo_x), n)].copy()
        x[:, 0] = rng.uniform(r_in, self.r_max, n) * rng.choice([-1.0, 1.0], n)
        return x

    def probe(self, radius, n, rng):
        if radius == 0.0:
            return self.nominal_starts(n, rng)
        return np.column_stack([radius * rng.choice([-1.0, 1.0], n), rng.uniform(-2.0, 2.0, n)])

    def diverged(self, x):
        return np.zeros(len(x), dtype=bool)

    def envelope(self, o, r_in):
        e, thd = self.obs_error(o), np.abs(o[:, 2])
        return (e > r_in) | (thd > 3.0), (e < 0.7 * r_in) & (thd < 1.0)

    @staticmethod
    def fid_distance(traj, ref):
        return np.abs(wrap(traj[:, :, 0] - ref[:, :, 0]))


def _quad_envelope(self, o, r_in):
    e = self.obs_error(o)
    tilt = np.maximum(np.abs(o[:, 6]), np.abs(o[:, 7]))
    spd = np.linalg.norm(o[:, 3:6], axis=1)
    return ((e > r_in) | (tilt > np.radians(35)) | (spd > 2.5),
            (e < 0.7 * r_in) & (tilt < np.radians(15)) & (spd < 1.0))


Quad3D.envelope = _quad_envelope
Quad3D.fid_distance = staticmethod(lambda traj, ref: np.linalg.norm(traj[:, :, 0:3] - ref[:, :, 0:3], axis=2))


# ---------------------------------------------------------------------------
# MuJoCo Reacher-v5 (unmodified Gymnasium model): task-space labeler, joint-space operator
# ---------------------------------------------------------------------------
class Reacher:
    """State x = [q0, q1, qd0, qd1, target x, target y]; observation = Reacher-v5 layout
    [cos q0, cos q1, sin q0, sin q1, target, qd, tip - target]; error = |tip - target| (m)."""
    name = "reacher"
    nx, nobs, nu = 6, 10, 2
    dt, horizon = 0.02, 200               # 4 s probes
    demo_T, demo_eps = 150, 30
    L1, L2 = 0.1, 0.11
    r_nominal = 0.03
    r_max = 0.4
    radii = [0.0, 0.05, 0.09, 0.14, 0.2]
    success = 0.03
    demo_short = 0.012                     # the operator stops 1.2 cm short of the target

    def __init__(self):
        import gymnasium as gym
        import mujoco
        self.mj = mujoco
        env = gym.make("Reacher-v5").unwrapped
        self.model, self.frame_skip = env.model, env.frame_skip
        self._pool = []

    def tip(self, q):
        return np.column_stack([self.L1 * np.cos(q[:, 0]) + self.L2 * np.cos(q[:, 0] + q[:, 1]),
                                self.L1 * np.sin(q[:, 0]) + self.L2 * np.sin(q[:, 0] + q[:, 1])])

    def obs(self, x):
        q, qd, tg = x[:, 0:2], x[:, 2:4], x[:, 4:6]
        return np.column_stack([np.cos(q[:, 0]), np.cos(q[:, 1]), np.sin(q[:, 0]), np.sin(q[:, 1]), tg, qd,
                                self.tip(q) - tg])

    def error(self, x):
        return np.linalg.norm(self.tip(x[:, 0:2]) - x[:, 4:6], axis=1)

    @staticmethod
    def obs_error(o):
        return np.linalg.norm(o[:, 8:10], axis=1)

    @staticmethod
    def _unpack(o):
        q = np.column_stack([np.arctan2(o[:, 2], o[:, 0]), np.arctan2(o[:, 3], o[:, 1])])
        return q, o[:, 6:8], o[:, 4:6], o[:, 8:10]

    def step(self, x, u):
        while len(self._pool) < len(x):
            self._pool.append(self.mj.MjData(self.model))
        out = np.empty_like(x)
        u = np.clip(u, -1, 1)
        for i in range(len(x)):
            d = self._pool[i]
            d.qpos[:] = [x[i, 0], x[i, 1], x[i, 4], x[i, 5]]
            d.qvel[:] = [x[i, 2], x[i, 3], 0.0, 0.0]
            d.ctrl[:] = u[i]
            self.mj.mj_step(self.model, d, nstep=self.frame_skip)
            out[i] = [d.qpos[0], d.qpos[1], d.qvel[0], d.qvel[1], x[i, 4], x[i, 5]]
        out[:, 1] = np.clip(out[:, 1], -3.1, 3.1)
        return out

    def _jac(self, q):
        s0, c0 = np.sin(q[:, 0]), np.cos(q[:, 0])
        s01, c01 = np.sin(q.sum(1)), np.cos(q.sum(1))
        J = np.zeros((len(q), 2, 2))
        J[:, 0, 0] = -self.L1 * s0 - self.L2 * s01
        J[:, 0, 1] = -self.L2 * s01
        J[:, 1, 0] = self.L1 * c0 + self.L2 * c01
        J[:, 1, 1] = self.L2 * c01
        return J

    def labeler(self, o):
        """Task-space PD through the Jacobian: straight, fast tip paths to the target."""
        q, qd, tg, d = self._unpack(o)
        J = self._jac(q)
        F = -25.0 * d - 1.5 * np.einsum("nij,nj->ni", J, qd)
        return np.clip(4.0 * (np.einsum("nji,nj->ni", J, F) - 0.05 * qd), -1, 1)

    def _ik_branch(self, aim, q):
        """Joint offset to the inverse-kinematics solution of `aim` on the current elbow branch."""
        L1, L2 = self.L1, self.L2
        r2 = np.sum(aim ** 2, axis=1)
        c1 = np.clip((r2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1)
        sgn = np.where(q[:, 1] >= 0, 1.0, -1.0)
        q1 = sgn * np.arccos(c1)
        q0 = np.arctan2(aim[:, 1], aim[:, 0]) - np.arctan2(L2 * np.sin(q1), L1 + L2 * np.cos(q1))
        return wrap(np.column_stack([q0, q1]) - q)

    def _aim(self, tg):
        r = np.linalg.norm(tg, axis=1, keepdims=True)
        return tg * (1.0 - self.demo_short / np.maximum(r, 1e-6))

    def demonstrator(self, o):
        """Operator style: joint-space PD toward the nearest inverse-kinematics solution of a point
        1.2 cm short of the target, on the current elbow branch (curved tip paths, different
        equilibrium, softer)."""
        q, qd, tg, _ = self._unpack(o)
        dq = self._ik_branch(self._aim(tg), q)
        return np.clip(2.0 * dq - 0.25 * qd, -1, 1)

    @staticmethod
    def _targets(n, rng):
        rad = rng.uniform(0.06, 0.19, n)
        ang = rng.uniform(-np.pi, np.pi, n)
        return np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])

    def nominal_starts(self, n, rng):
        tg = self._targets(n, rng)
        q0 = np.column_stack([rng.uniform(-np.pi, np.pi, n), rng.choice([-1.0, 1.0], n) * 1.0])
        q = q0 + self._ik_branch(self._aim(tg), q0) + rng.normal(0, 0.06, (n, 2))
        return np.column_stack([q, rng.normal(0, 0.3, (n, 2)), tg])

    def process_noise(self, x, rng):
        x = x.copy()
        x[:, 2:4] += rng.normal(0, 0.05, (len(x), 2))
        return x

    def _random(self, n, rng, lo, hi, qd_max):
        out = np.empty((0, 6))
        while len(out) < n:
            m = 4 * (n - len(out)) + 16
            tg = self._targets(m, rng)
            x = np.column_stack([rng.uniform(-np.pi, np.pi, m), rng.uniform(-2.9, 2.9, m),
                                 rng.uniform(-qd_max, qd_max, (m, 2)), tg])
            e = self.error(x)
            out = np.concatenate([out, x[(e >= lo) & (e <= hi)]])
        return out[:n]

    def box_states(self, n, rng, r_in):
        return self._random(n, rng, r_in, self.r_max, 10.0)

    def box_partial(self, n, rng, r_in, demo_x):
        base = demo_x[rng.integers(0, len(demo_x), n)].copy()
        out = np.empty_like(base)
        todo = np.arange(n)
        while len(todo):
            cand = base[todo].copy()
            cand[:, 0] = rng.uniform(-np.pi, np.pi, len(todo))
            cand[:, 1] = rng.uniform(-2.9, 2.9, len(todo))
            ok = self.error(cand) >= r_in
            out[todo[ok]] = cand[ok]
            todo = todo[~ok]
        return out

    def probe(self, radius, n, rng):
        if radius == 0.0:
            return self.nominal_starts(n, rng)
        return self._random(n, rng, radius - 0.005, radius + 0.005, 2.0)

    def diverged(self, x):
        return np.zeros(len(x), dtype=bool)

    def envelope(self, o, r_in):
        e, sp = self.obs_error(o), np.abs(o[:, 6:8]).max(1)
        return (e > r_in) | (sp > 4.0), (e < 0.7 * r_in) & (sp < 1.5)

    def fid_distance(self, traj, ref):
        T, n, _ = traj.shape
        a = self.tip(traj.reshape(-1, 6)[:, 0:2]).reshape(T, n, 2)
        b = self.tip(ref.reshape(-1, 6)[:, 0:2]).reshape(T, n, 2)
        return np.linalg.norm(a - b, axis=2)


DOMAINS = {"quad3d": Quad3D, "pendulum": Pendulum, "reacher": Reacher}


def mismatched(base, gain):
    """The same domain and demonstrator with a high-gain, saturating labeler tanh(gain * labeler(o)):
    it still recovers, but its actions near nominal differ from the demonstrator's in a way that is
    harder to fit than the demonstrator's own (the regime of the flight study). The domain keeps its
    name, so the demonstrations are identical to the base domain's."""
    class Mismatched(base):
        def labeler(self, o):
            return np.tanh(gain * base.labeler(self, o))
    Mismatched.__name__ = f"{base.__name__}HG{gain}"
    return Mismatched


DOMAINS.update({"quad3d_hg4": mismatched(Quad3D, 4), "quad3d_hg16": mismatched(Quad3D, 16),
                "reacher_hg4": mismatched(Reacher, 4), "reacher_hg16": mismatched(Reacher, 16)})


def three_level(base, a):
    """A bang-bang labeler: each action is -1, 0 or +1, the sign of the base labeler's action where its
    magnitude exceeds a (a quantized, high-gain version of the same law)."""
    class ThreeLevel(base):
        def labeler(self, o):
            k = base.labeler(self, o)
            return np.where(np.abs(k) > a, np.sign(k), 0.0)
    ThreeLevel.__name__ = f"{base.__name__}Q{a}"
    return ThreeLevel


# held-out settings for the prospective test of the three-outcome rule
DOMAINS.update({"pendulum_hg4": mismatched(Pendulum, 4), "pendulum_q10": three_level(Pendulum, 0.1),
                "quad3d_q05": three_level(Quad3D, 0.05), "reacher_hg2": mismatched(Reacher, 2)})
