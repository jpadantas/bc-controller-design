"""Closed-loop evaluation of wingman policies in FlightGear 2020.3.19 with the
ASA F-16CJ Block 52 (the pilots' own simulator and aircraft).

One rollout = launch FlightGear in the air at a pilot's first (cleaned) state,
optionally displaced by (lateral, longitudinal) feet in the leader's frame,
then fly `policy` at 10 Hz of simulated time for `horizon` seconds while the
leader follows that pilot's logged leader track. Logs slot error over time.

    python fg_eval.py --policy labeler --pilot 1 --horizon 120
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.argv_saved = sys.argv
sys.argv = sys.argv[:1]
import fg_bridge as B  # noqa: E402
import flight_unified as F  # noqa: E402
sys.argv = sys.argv_saved

OUT = os.path.join(F.ROOT, "outputs", "flightgear")


class LeaderTrack:
    """Logged leader track of one pilot's session, linearly interpolated in time
    (relative to the first cleaned frame); speed from successive positions."""

    def __init__(self, g):
        self.t = g.time_s.values - g.time_s.values[0]
        self.lat, self.lon = g.leader_lat.values, g.leader_lon.values
        self.alt, self.hdg = g.leader_alt_ft.values, np.unwrap(np.radians(g.leader_heading.values))
        dn = np.gradient(self.lat) * 364567.2
        de = np.gradient(self.lon) * 364567.2 * np.cos(np.radians(self.lat))
        self.kts = np.hypot(dn, de) / np.maximum(np.gradient(self.t), 1e-3) / 1.68781

    def __call__(self, t):
        f = lambda v: float(np.interp(t, self.t, v))  # noqa: E731
        return f(self.lat), f(self.lon), f(self.alt), float(np.degrees(f(self.hdg)) % 360.0), f(self.kts)


def start_state(g, off_lat=0.0, off_lon=0.0):
    r = g.iloc[0]
    psi = np.radians(r.leader_heading)
    dE = off_lat * np.cos(psi) + off_lon * np.sin(psi)
    dN = -off_lat * np.sin(psi) + off_lon * np.cos(psi)
    lat = r.latitude_deg + np.degrees(dN / B.R_FEET)
    lon = r.longitude_deg + np.degrees(dE / (B.R_FEET * np.cos(np.radians(r.latitude_deg))))
    return lat, lon, r.altitude_ft, r.heading_deg, r.airspeed_kt, r.roll_deg, r.pitch_deg


def rollout(act, g, off=(0.0, 0.0), horizon=120.0, dt=0.1, needs_leader=False, tag="", perturb=None):
    """act(x9) -> 4 (or 6, with trims) controls, or act(x9, leader) if needs_leader. `perturb`
    optionally adds dict(roll=deg, pitch=deg, kias=kt) to the start state. Returns a log DataFrame."""
    io = B.FGUdp()
    st0 = list(start_state(g, *off))
    if perturb:
        st0[4] += perturb.get("kias", 0.0)
        st0[5] += perturb.get("roll", 0.0)
        st0[6] += perturb.get("pitch", 0.0)
    proc = B.launch(*st0)
    try:
        fg = B.FGProps(timeout=300)
        fg.wait_ready()
        fg.close()
        lead = LeaderTrack(g)
        st = io.latest(timeout=60)
        t0, next_t, rows = st["t"], st["t"], []
        u = np.array([0.0, 0.0, 0.0, 0.75])
        while True:
            st = io.latest()
            t = st["t"] - t0
            if t >= horizon:
                break
            if st["t"] + 1e-6 < next_t:
                time.sleep(0.002)
                continue
            next_t += dt
            L = lead(t)
            x = B.policy_inputs(st, L[:4])
            u = np.asarray(act(x, L) if needs_leader else act(x), dtype=float).ravel()
            if len(u) < 6:  # policies without trim outputs fly with trims at zero
                u = np.concatenate([u[:4], np.zeros(6 - len(u[:4]))])
            u[:3] = np.clip(u[:3], -1, 1)
            u[3] = np.clip(u[3], 0, 1)
            u[4:6] = np.clip(u[4:6], -1, 1)
            io.send(u[:4], u[4:6])
            e = float(np.linalg.norm(x[:3] - B.SLOT))
            rows.append(dict(t=t, slot_err=e, xb=x[0], yb=x[1], dz=x[2], roll=x[3], pitch=x[4], kias=x[6],
                             alt=st["alt"], ail=u[0], ele=u[1], rud=u[2], thr=u[3], ail_trim=u[4], ele_trim=u[5],
                             hdg_rel=x[5], vs=x[7], nz=x[8]))
            if st["alt"] < 300:  # ground impact (no scenery: sea level); stop the rollout
                break
    finally:
        io.close()
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            proc.kill()
    df = pd.DataFrame(rows)
    df["tag"] = tag
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=1)
    ap.add_argument("--horizon", type=float, default=120.0)
    ap.add_argument("--off-lat", type=float, default=0.0)
    ap.add_argument("--off-lon", type=float, default=0.0)
    a = ap.parse_args()
    df = F.prepare(pd.read_csv(F.DATA))
    g = df[df.pilot_id == a.pilot].reset_index(drop=True)
    ctrl = B.FormationController()
    log = rollout(ctrl, g, (a.off_lat, a.off_lon), a.horizon, tag="labeler")
    os.makedirs(OUT, exist_ok=True)
    log.to_csv(os.path.join(OUT, f"labeler_pilot{a.pilot}.csv"), index=False)
    for t in (5, 10, 30, 60, 90, 119):
        r = log.iloc[(log.t - t).abs().argmin()]
        print(f"t={r.t:5.1f}s slot_err={r.slot_err:8.0f} ft  xb={r.xb:7.0f} yb={r.yb:7.0f} dz={r.dz:6.0f} "
              f"roll={r.roll:6.1f} kias={r.kias:5.0f}  ail={r.ail:5.2f} ele={r.ele:5.2f} thr={r.thr:4.2f}", flush=True)
    print("human slot error at the same times:", [round(float(np.interp(t, g.time_s - g.time_s.iloc[0], g.err)))
                                                    for t in (5, 10, 30, 60, 90, 119)])


if __name__ == "__main__":
    main()
