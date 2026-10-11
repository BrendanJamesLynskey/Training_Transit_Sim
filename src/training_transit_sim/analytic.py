"""Closed forms for one step, written from the collective formulas, independent of the schedules.

At zero background load the discrete-event engine must reproduce these to rounding (``tests/test_analytic.py``):
step = compute + (latency + bytes / bandwidth, summed over the sync's rounds) / sync_every.

n = workers per DC, D = DCs, Li / bi = intra-DC latency and bandwidth, Lw / bw = inter-DC, c = a pluggable's
port cap on each flow it sends.

* ring all-reduce of B:  2 (n - 1) (Li + B / (n bi)) + 2 (D - 1) (Lw + B / (D bw))
* ring all-gather of P:  (n - 1) (Li + P / min(bi, c)) + (D - 1) (Lw + max(n P / bw, P / c))
  + [D > 1] (n - 1) (Li + (D - 1) P / min(bi, c))
* module stream of b bytes (pipelined stages, the slowest sets the rate):
  Lm + max(b / port, (b / 2) planes / optics bits/s, b x charged ops per byte / FLOP/s, FPGA ops / FPGA rate)
"""

from __future__ import annotations

import math

from .hardware import Kernel, Module, band_kernel, sum_kernel
from .model import StepPlan


def module_stream(m: Module, b: float, k: Kernel) -> float:
    values = b / 2
    charged = 0.0
    for op, v in k.ops_per_byte.items():
        if op in ("add", "mul", "mac", "fp_mac") or (op == "fft" and m.transform_charged):
            charged += v
    return m.latency + max(b / m.port_rate,
                           values * m.planes / m.optics_bits_per_s if k.optical else 0.0,
                           b * charged / m.gflops,
                           (values * k.fpga_ops_per_value + b * k.ops_per_byte.get("modular", 0.0)) / m.fpga_ops_per_s)


def ring_allreduce(n, D, B, intra, inter) -> float:
    return 2 * (n - 1) * (intra.latency + B / n / intra.bandwidth) + 2 * (D - 1) * (inter.latency + B / D / inter.bandwidth)


def ring_allgather(n, D, P, intra, inter, cap=None) -> float:
    c = cap or math.inf
    t = (n - 1) * (intra.latency + P / min(intra.bandwidth, c))
    t += (D - 1) * (inter.latency + max(n * P / inter.bandwidth, P / c))
    if D > 1:
        t += (n - 1) * (intra.latency + (D - 1) * P / min(intra.bandwidth, c))
    return t


def sync_time(p: StepPlan) -> float:
    cfg, t, m = p.cfg, p.cfg.topo, p.cfg.module
    n, D, Li, bi = t.per_dc, t.dcs, t.intra.latency, t.intra.bandwidth
    v = cfg.variant
    if v in ("adamw", "band-allreduce"):
        return ring_allreduce(n, D, p.payload, t.intra, t.inter)
    if v in ("topk", "band"):
        return ring_allgather(n, D, p.payload, t.intra, t.inter)
    if v == "band-pluggable":
        cap = m.port_rate if m.caps_port else None
        return module_stream(m, p.dense, band_kernel(planes=m.planes)) + ring_allgather(n, D, p.payload, t.intra, t.inter, cap)
    if v == "band-switch":
        B = p.payload
        return ((Li + p.dense / bi)                                          # dense delta up
                + module_stream(m, p.dense, band_kernel(planes=m.planes))    # band at the port
                + max(module_stream(m, B, sum_kernel(n)), Li + B / bi)       # sum, and own band back
                + 2 * (D - 1) * (t.inter.latency + B / D / t.inter.bandwidth)  # all-reduce between switches
                + (Li + B / bi))                                             # global sum down
    raise ValueError(v)


def step_time(p: StepPlan) -> float:
    return p.compute_s + sync_time(p) / p.cfg.sync_every


def wire_per_worker(p: StepPlan) -> float:
    """Bytes a worker sends per sync, from the ring formulas (switch: up the dense delta)."""
    t, v = p.cfg.topo, p.cfg.variant
    N = t.workers
    if v in ("adamw", "band-allreduce"):
        n, D = t.per_dc, t.dcs
        return 2 * (n - 1) / n * p.payload + 2 * (D - 1) / D * p.payload / n
    if v in ("topk", "band", "band-pluggable"):
        return (N - 1) * p.payload
    return p.dense
