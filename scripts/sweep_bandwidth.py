"""Sweep the inter-DC bandwidth and print time to a target loss for every variant, as CSV.

    python scripts/sweep_bandwidth.py --target 3.158 --gbps 1 2 5 10 25 50 100 400 800

Two data centres of four workers, module [free] (transform not charged) and [high] (99 GFLOP/s, charged).
The Jenkinsfile's nightly stage runs it; nothing in results.md comes from it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from training_transit_sim.hardware import Module, Topology  # noqa: E402
from training_transit_sim.model import TRANSIT, VARIANTS, Config  # noqa: E402
from training_transit_sim.study import evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=3.158)
    ap.add_argument("--gbps", type=float, nargs="+", default=[1, 2, 5, 10, 25, 50, 100, 400, 800])
    a = ap.parse_args()
    mods = {"free": Module(gflops=99e9, transform_charged=False), "high": Module(gflops=99e9)}
    print("wan_gbps,variant,module,steps,step_ms,time_s,hot_spot")
    for g in a.gbps:
        topo = Topology("2x4", workers=8, dcs=2).with_inter(g * 1e9 / 8)
        for v in VARIANTS:
            for mk in (mods if v in TRANSIT else ["high"]):
                o = evaluate(Config(v, topo, module=mods[mk]), a.target)
                print(f"{g:g},{v},{mk if v in TRANSIT else ''},{o.steps or ''},{o.step_time * 1e3:.4f},"
                      f"{'' if o.steps is None else f'{o.time:.4f}'},{o.result.hot_spot}")


if __name__ == "__main__":
    main()
