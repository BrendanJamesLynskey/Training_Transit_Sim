"""The training variants: what each worker computes and sends per step, and how many steps it needs.

Six variants, four measured and two hypothetical:

========================  =========================================================  ================================
variant                   what it is                                                  steps to a loss from
========================  =========================================================  ================================
``adamw``                 A: AdamW, dense BF16 gradients, ring all-reduce              A's runs (3 seeds)
``topk``                  B: DeMo top-k (2-D 64x64 chunks) on the GPU, all-gather      B's runs (3 seeds)
``band``                  C: DeMo fixed band (1-D 256) on the GPU, all-gather          C's runs (3 seeds)
``band-pluggable``        E: the band computed in transit by each worker's pluggable    E's runs at the module's ENOB and
                          module, all-gather (feedback of the band sent, which the      planes (one seed each); or C's,
                          all-gather returns)                                           with ``exact_steps=True``
``band-allreduce``        hypothetical: C's band, summed by a ring all-reduce           C's runs (assumed: not run)
``band-switch``           hypothetical: the band computed and summed in the network     as band-pluggable (assumed: not
                          by switch-placed modules, each worker's own band looped back  run)
========================  =========================================================  ================================

Every worker sends the same band positions, and DeMo averages before taking the sign, so summing the bands gives
the same update as gathering them; that is why the two hypothetical variants reuse measured step counts. They
were not run. That the repaired optics band reaches a loss in fewer steps than the exact band is a hypothesis
(one seed per E run), so every in-transit result is also given with the exact band's steps.

Bytes per step scale with the parameter count by each variant's measured payload per parameter. Steps to a loss
were measured only at 15.8M parameters, 8 workers and 3,000 steps: any other size is an extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from . import inputs
from .collectives import Schedule, allgather, allreduce, pluggable_band, switch_band_allreduce
from .hardware import Module, Topology, band_kernel, sum_kernel

GPU_FLOPS_PER_VALUE = {
    # forward transform on the worker's GPU, FFTW's 2.5 n log2 n convention (as the hub's cit_budget.py)
    "topk": 2 * 2.5 * 6 + 2,     # 2-D DCT of 64x64 chunks (2 x 2.5 log2 64) + top-k, ~2 comparisons (illustrative)
    "band": 2.5 * 8,             # 1-D DCT of 256-point chunks
    "band-allreduce": 2.5 * 8,
}

VARIANTS = {
    "adamw": dict(label=inputs.LABEL_ADAMW, name="A  AdamW, dense ring all-reduce"),
    "topk": dict(label=inputs.LABEL_TOPK, name="B  DeMo top-k, all-gather"),
    "band": dict(label=inputs.LABEL_BAND, name="C  fixed band on the GPU, all-gather"),
    "band-pluggable": dict(label=None, name="E  band in transit (pluggable), all-gather"),
    "band-allreduce": dict(label=inputs.LABEL_BAND, name="C' fixed band on the GPU, ring all-reduce (hypothetical)"),
    "band-switch": dict(label=None, name="E' band in transit, summed at the switch (hypothetical)"),
}
TRANSIT = ("band-pluggable", "band-switch")
HYPOTHETICAL = ("band-allreduce", "band-switch")


@dataclass(frozen=True)
class Config:
    variant: str
    topo: Topology
    module: Module = field(default_factory=Module)
    params: float | None = None          # None: the experiment's 15,833,472
    tokens_per_worker: int | None = None  # None: the experiment's 8 x 256 = 2,048
    sync_every: int = 1                  # H > 1: DiLoCo-style, an extrapolation (the experiment synced every step)
    exact_steps: bool = False            # in-transit variants: use the exact band's (C's) steps, not E's

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.variant == "band-switch" and self.module.placement != "switch":
            object.__setattr__(self, "module", replace(self.module, placement="switch"))
        if self.variant == "band-pluggable" and self.module.placement != "pluggable":
            object.__setattr__(self, "module", replace(self.module, placement="pluggable"))


@dataclass
class StepPlan:
    cfg: Config
    params: float
    compute_s: float          # per worker per step
    flops: float              # per worker per step (training + any compression on the GPU)
    schedule: Schedule        # one sync
    payload: float            # bytes a worker originates per sync (after compression)
    dense: float              # BF16 gradient / delta bytes per worker
    n_modules: int

    @property
    def link_of(self):
        t = self.cfg.topo
        return lambda key: t.inter if key[0] == "wan" else t.intra


def steps_label(cfg: Config) -> str:
    v = VARIANTS[cfg.variant]["label"]
    if v is not None:
        return v
    if cfg.exact_steps:
        return inputs.LABEL_BAND
    return inputs.label_optics(cfg.module.enob, cfg.module.planes)


def steps_to_target(cfg: Config, target: float, exp=None) -> float | None:
    """Steps to reach ``target`` (a ladder loss), or None when not every seed reached it in 3,000 steps."""
    exp = exp or inputs.load()
    label = steps_label(cfg)
    if label not in exp.ladder:
        raise KeyError(f"no measured run for {label!r}: ENOB {inputs.ENOBS} at 8 planes, planes {inputs.PLANES} at ENOB 8")
    return exp.steps_to(label, target)


def plan(cfg: Config, exp=None) -> StepPlan:
    exp = exp or inputs.load()
    topo, mod = cfg.topo, cfg.module
    P = cfg.params if cfg.params is not None else exp.setup.params
    tokens = cfg.tokens_per_worker if cfg.tokens_per_worker is not None else exp.setup.tokens_per_worker
    dense = exp.bytes_per_param(inputs.LABEL_ADAMW) * P            # 2 bytes per parameter (BF16)
    flops = 6.0 * P * tokens
    acc = topo.accelerator
    rate = acc.peak_flops * topo.mfu
    gpu_extra = GPU_FLOPS_PER_VALUE.get(cfg.variant, 0.0) * P
    v = cfg.variant
    n_modules = 0
    if v == "adamw":
        payload = dense
        sched = allreduce(topo, dense)
    elif v in ("topk", "band"):
        payload = exp.bytes_per_param(VARIANTS[v]["label"]) * P
        sched = allgather(topo, payload)
    elif v == "band-allreduce":
        payload = exp.bytes_per_param(inputs.LABEL_BAND) * P
        sched = allreduce(topo, payload)
    elif v == "band-pluggable":
        payload = exp.bytes_per_param(steps_label(cfg)) * P
        cap = mod.port_rate if mod.caps_port else None
        sched = Schedule()
        sched.add("band computed in each worker's pluggable", pluggable_band(topo, dense, band_kernel(planes=mod.planes)))
        g = allgather(topo, payload, cap=cap)
        sched.phases.extend(g.phases)
        n_modules = topo.workers
    elif v == "band-switch":
        payload = exp.bytes_per_param(inputs.LABEL_BAND) * P
        sched = switch_band_allreduce(topo, dense, payload, band_kernel(planes=mod.planes), sum_kernel(topo.per_dc))
        n_modules = topo.workers           # one per worker-facing switch port
    else:  # pragma: no cover
        raise ValueError(v)
    return StepPlan(cfg=cfg, params=P, compute_s=(flops + gpu_extra) / rate, flops=flops + gpu_extra,
                    schedule=sched, payload=payload, dense=dense, n_modules=n_modules)
