"""Open-loop replay check of the FlightGear pipeline.

For each pilot: launch the F-16 at the pilot's first cleaned frame (as in the
experiment), replay the pilot's logged stick, rudder and throttle at 10 Hz
for 30 s, either with the logged aileron/elevator trims or with trims held at
zero (as every policy flies), and compare the simulated roll, pitch, heading,
altitude and airspeed with the log.

    python fg_replay.py
Writes outputs/flightgear/replay/replay_p{k}_{trim|notrim}.csv and replay_summary.csv.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.argv = sys.argv[:1]
import fg_bridge as B  # noqa: E402
import fg_eval as E  # noqa: E402
import flight_unified as F  # noqa: E402

OUT = os.path.join(F.ROOT, "outputs", "flightgear", "replay")
HORIZON = 30.0


def replay(g, with_trim):
    io = B.FGUdp()
    proc = B.launch(*E.start_state(g))
    rows = []
    try:
        fg = B.FGProps(timeout=300)
        fg.wait_ready()
        fg.close()
        st = io.latest(timeout=60)
        t0, next_t = st["t"], st["t"]
        tl = g.time_s.values - g.time_s.values[0]
        while True:
            st = io.latest()
            t = st["t"] - t0
            if t >= HORIZON:
                break
            if st["t"] + 1e-6 < next_t:
                time.sleep(0.002)
                continue
            next_t += 0.1
            i = int(np.searchsorted(tl, t, side="right") - 1)
            r = g.iloc[max(i, 0)]
            trims = (r.aileron_trim, r.elevator_trim) if with_trim else (0.0, 0.0)
            io.send([r.aileron, r.elevator, r.rudder, r.throttle], trims)
            rows.append(dict(t=t, roll=st["roll"], pitch=st["pitch"], hdg=st["heading"], alt=st["alt"],
                             kias=st["kias"], log_roll=r.roll_deg, log_pitch=r.pitch_deg, log_hdg=r.heading_deg,
                             log_alt=r.altitude_ft, log_kias=r.airspeed_kt))
            if st["alt"] < 300:
                break
    finally:
        io.close()
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            proc.kill()
    return pd.DataFrame(rows)


def summarize(d):
    out = {}
    for w in (5, 10, 30):
        s = d[d.t <= w]
        dh = ((s.hdg - s.log_hdg + 180) % 360) - 180
        out.update({f"roll_rmse_{w}s": float(np.sqrt(np.mean((s.roll - s.log_roll) ** 2))),
                    f"pitch_rmse_{w}s": float(np.sqrt(np.mean((s.pitch - s.log_pitch) ** 2))),
                    f"hdg_rmse_{w}s": float(np.sqrt(np.mean(dh ** 2))),
                    f"alt_rmse_{w}s": float(np.sqrt(np.mean((s.alt - s.log_alt) ** 2)))})
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    df = F.prepare(pd.read_csv(F.DATA))
    summ = []
    for k in [1, 2, 3, 4, 5]:
        g = df[df.pilot_id == k].reset_index(drop=True)
        for with_trim in (True, False):
            tag = "trim" if with_trim else "notrim"
            path = os.path.join(OUT, f"replay_p{k}_{tag}.csv")
            if os.path.exists(path):
                d = pd.read_csv(path)
            else:
                d = replay(g, with_trim)
                d.to_csv(path, index=False)
            s = dict(pilot=k, trims=tag, t_end=float(d.t.iloc[-1]), **summarize(d))
            summ.append(s)
            print({k_: (round(v, 1) if isinstance(v, float) else v) for k_, v in s.items()}, flush=True)
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "replay_summary.csv"), index=False)


if __name__ == "__main__":
    main()
