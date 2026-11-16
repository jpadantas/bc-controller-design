"""Parallel-instance and determinism check: run the same 20 s trimmed replay of
pilot 2 in N FlightGear instances at once (FG_INSTANCE=0..N-1) and compare.

    python fg_parallel_test.py 3 OUTDIR
"""
import os
import subprocess
import sys

if len(sys.argv) > 1 and sys.argv[1] == "--child":
    out = sys.argv[2]
    sys.argv = sys.argv[:1]
    import pandas as pd
    import fg_replay as R
    import flight_unified as F
    R.HORIZON = 20.0
    df = F.prepare(pd.read_csv(F.DATA))
    g = df[df.pilot_id == 2].reset_index(drop=True)
    R.replay(g, True).to_csv(out, index=False)
else:
    n, outdir = int(sys.argv[1]), sys.argv[2]
    procs = []
    for i in range(n):
        env = dict(os.environ, FG_INSTANCE=str(i))
        procs.append(subprocess.Popen([sys.executable, __file__, "--child", os.path.join(outdir, f"inst{i}.csv")], env=env))
    for p in procs:
        p.wait()
    import pandas as pd
    ds = [pd.read_csv(os.path.join(outdir, f"inst{i}.csv")) for i in range(n)]
    for i, d in enumerate(ds):
        print(i, len(d), "final roll %.3f alt %.1f" % (d.roll.iloc[-1], d.alt.iloc[-1]),
              "max|roll-log| %.2f" % (d.roll - d.log_roll).abs().max())
