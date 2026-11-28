"""Human reference distances on robomimic Lift: for each demonstration of a held-out operator (every other one,
as in lift_study.py), the end-effector path distance (time-aligned RMS, as in the study) to the demonstration of
ANOTHER operator whose initial cube position is closest, and to the closest other demonstration of the SAME
operator. Writes outputs/robo/between_operators.csv (with ROBO_TASK=can or square: between_operators_<task>.csv).

    ROBO_DATA=<path> python lift_between.py
    ROBO_TASK=can ROBO_CAN=<path> python lift_between.py
    ROBO_TASK=square ROBO_SQUARE=<path> python lift_between.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("ROBO_TASK") == "can":
    import can_common as L  # noqa: E402
elif os.environ.get("ROBO_TASK") == "square":
    import square_common as L  # noqa: E402
else:
    import lift_common as L  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rms(P, R):
    T = max(len(P), len(R))
    P = np.concatenate([P, np.repeat(P[-1:], T - len(P), 0)])
    R = np.concatenate([R, np.repeat(R[-1:], T - len(R), 0)])
    return float(np.sqrt(np.mean(np.sum((P - R) ** 2, 1))))


def main():
    data, _ = L.load()
    rows = []
    for h in L.OPERATORS:
        for d in data[h][::2]:
            c0 = d["obs"][0, 9:11]
            others = [e for o in L.OPERATORS if o != h for e in data[o]]
            j = int(np.argmin([np.linalg.norm(e["obs"][0, 9:11] - c0) for e in others]))
            same = [e for e in data[h] if e is not d]
            k = int(np.argmin([np.linalg.norm(e["obs"][0, 9:11] - c0) for e in same]))
            rows.append(dict(held_out=h, between=rms(others[j]["obs"][:, 0:3], d["obs"][:, 0:3]),
                             within=rms(same[k]["obs"][:, 0:3], d["obs"][:, 0:3])))
    df = pd.DataFrame(rows)
    task = os.environ.get("ROBO_TASK", "lift")
    out = os.path.join(ROOT, "outputs", "robo", "between_operators.csv" if task == "lift"
                       else f"between_operators_{task}.csv")
    df.to_csv(out, index=False)
    print(df.groupby("held_out")[["between", "within"]].mean().mul(100).round(1))
    print("mean (cm):", df[["between", "within"]].mean().mul(100).round(1).to_dict())


if __name__ == "__main__":
    main()
