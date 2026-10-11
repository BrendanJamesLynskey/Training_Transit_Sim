"""The SimPy discrete-event engine: runs a step plan and measures time, utilisation, energy and hot spots.

Each link direction and each module is a ``simpy.Resource`` of capacity 1. A step is the device's compute, then
(every ``sync_every`` steps) the sync schedule round by round: the ops of a round start together, ops on the same
resource share it (their bytes add, the latency is paid once), and the round ends when the last op ends. The
round's time is charged to the slowest op's hot spot: compute, the intra-DC link, the inter-DC link, the module's
port, optics, compute budget, or FPGA stage.

Optional background traffic on the inter-DC links (``background``: the fraction of each link's capacity other
jobs use) arrives as a Poisson stream of ``bg_bytes`` transfers. It queues with the sync traffic first come first
served, which is where the discrete-event model and the closed forms (``analytic.py``) part company; at zero load
they agree exactly.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

import simpy

from .collectives import Process, Transfer
from .model import StepPlan

CATEGORIES = ("compute", "intra-DC link", "inter-DC link", "module port", "module optics", "module compute", "FPGA")


def link_category(key: tuple) -> str:
    return "inter-DC link" if key[0] == "wan" else "intra-DC link"


def group_time(plan: StepPlan, key: tuple, ops: list) -> tuple[float, str]:
    """Seconds a resource needs for its ops in one round, and the hot spot that sets it."""
    if key[0] == "mod":
        best, cat, total = 0.0, "module compute", 0.0
        for op in ops:
            t, stage = plan.cfg.module.stream_time(op.nbytes, op.kernel)
            total += t
            if t > best:
                best, cat = t, stage
        return total, cat
    lk = plan.link_of(key)
    total = sum(op.nbytes for op in ops)
    t_link = total / lk.bandwidth
    t_cap = max((op.nbytes / op.cap for op in ops if op.cap), default=0.0)
    if t_cap > t_link:
        return lk.latency + t_cap, "module port"
    return lk.latency + t_link, link_category(key)


@dataclass
class Result:
    plan: StepPlan
    steps: int                                  # steps simulated
    sim_time: float
    attribution: dict = field(default_factory=dict)      # category -> seconds over the run
    busy: dict = field(default_factory=dict)             # resource -> seconds of sync traffic
    bg_busy: dict = field(default_factory=dict)          # resource -> seconds of background traffic
    bytes_on: dict = field(default_factory=dict)         # resource -> bytes of sync traffic
    module_bytes: float = 0.0                            # bytes processed by modules over the run

    @property
    def step_time(self) -> float:
        return self.sim_time / self.steps

    def hot_spots(self) -> dict:
        """Share of the step time each category is responsible for (sums to 1)."""
        return {c: self.attribution.get(c, 0.0) / self.sim_time for c in CATEGORIES}

    @property
    def hot_spot(self) -> str:
        h = self.hot_spots()
        return max(h, key=h.get)

    def utilisation(self, kind: str, background: bool = False) -> float:
        """Busiest resource of a kind ("up", "down", "wan", "mod"): busy time / simulated time."""
        keys = [k for k in self.busy if k[0] == kind]
        if not keys:
            return 0.0
        return max((self.busy[k] + (self.bg_busy.get(k, 0.0) if background else 0.0)) / self.sim_time for k in keys)

    def mean_utilisation(self, kind: str) -> float:
        keys = [k for k in self.busy if k[0] == kind]
        return sum(self.busy[k] for k in keys) / len(keys) / self.sim_time if keys else 0.0

    def energy(self) -> dict:
        """Joules per step: compute (device static + dynamic), link (pJ/bit over every byte), module (static and
        laser for every module the whole time + pJ/bit over the bytes it processes)."""
        p, topo = self.plan, self.plan.cfg.topo
        acc, mod = topo.accelerator, p.cfg.module
        T = self.step_time
        compute = topo.workers * (acc.idle_w * T + acc.pj_per_flop * 1e-12 * p.flops)
        link = sum(b * 8 * p.link_of(k).pj_per_bit * 1e-12 for k, b in self.bytes_on.items()) / self.steps
        module = p.n_modules * mod.static_power * T + self.module_bytes * 8 * mod.pj_per_bit * 1e-12 / self.steps
        return {"compute": compute, "link": link, "module": module}


def simulate(plan: StepPlan, syncs: int = 1, background: float = 0.0, bg_bytes: float = 1e6,
             seed: int = 0) -> Result:
    H = plan.cfg.sync_every
    env = simpy.Environment()
    res: dict = {}
    busy: dict = defaultdict(float)
    bg_busy: dict = defaultdict(float)
    bytes_on: dict = defaultdict(float)
    attribution: dict = defaultdict(float)
    module_bytes = [0.0]
    rounds = plan.schedule.rounds

    def resource(key):
        if key not in res:
            res[key] = simpy.Resource(env, capacity=1)
        return res[key]

    def key_of(op):
        return op.module if isinstance(op, Process) else op.link

    def run_group(key, ops, finish):
        t, cat = group_time(plan, key, ops)
        with resource(key).request() as req:
            yield req
            yield env.timeout(t)
        busy[key] += t
        for op in ops:
            if isinstance(op, Transfer):
                bytes_on[key] += op.nbytes
            else:
                module_bytes[0] += op.nbytes
        finish.append((env.now, cat))

    def main():
        for _ in range(syncs):
            for _ in range(H):
                yield env.timeout(plan.compute_s)
                attribution["compute"] += plan.compute_s
            for rnd in rounds:
                start = env.now
                groups = defaultdict(list)
                for op in rnd:
                    groups[key_of(op)].append(op)
                finish: list = []
                yield simpy.AllOf(env, [env.process(run_group(k, g, finish)) for k, g in groups.items()])
                attribution[max(finish)[1]] += env.now - start

    if background > 0:
        rng = random.Random(seed)
        wan = sorted({op.link for r in rounds for op in r if isinstance(op, Transfer) and op.link[0] == "wan"})

        def bg_job(key, t):
            with resource(key).request() as req:
                yield req
                yield env.timeout(t)
            bg_busy[key] += t

        def bg_source(key):
            bw = plan.cfg.topo.inter.bandwidth
            t = bg_bytes / bw
            rate = background * bw / bg_bytes          # arrivals per second
            while True:
                yield env.timeout(rng.expovariate(rate))
                env.process(bg_job(key, t))

        for k in wan:
            env.process(bg_source(k))

    proc = env.process(main())
    env.run(until=proc)
    return Result(plan=plan, steps=syncs * H, sim_time=env.now, attribution=dict(attribution), busy=dict(busy),
                  bg_busy=dict(bg_busy), bytes_on=dict(bytes_on), module_bytes=module_bytes[0])
