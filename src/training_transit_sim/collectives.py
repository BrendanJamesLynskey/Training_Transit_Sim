"""Collective schedules as data, and a checker that they move the right data.

A schedule is a list of phases; a phase is a list of rounds; a round is a list of ops that run at the same time
(the next round starts when every op of this one has finished). Two kinds of op:

* ``Transfer``: ``nbytes`` from ``src`` to ``dst`` over one link resource (``("up", w)``: worker w's port to its DC
  fabric; ``("down", w)``: the DC's switch to worker w; ``("wan", d)``: DC d to DC d+1 on the ring of DCs). Flows
  in the same round on the same resource share it.
* ``Process``: ``nbytes`` through one in-transit module (``("mod", ...)``) with a ``Kernel``.

Each op also names the data it carries or makes (``keys``), so ``verify`` can replay the schedule on sets of
contributors and check the result: every worker ends with every block (all-gather) or with every chunk summed over
every worker exactly once (all-reduce; a reduce of overlapping contributors raises, so double counting is caught).
The timing engine (``engine.py``) never looks at the keys.

Wire bytes (what each worker sends), as the experiment counts them:

* ring all-reduce of B on N workers: 2 (N - 1) / N x B per worker;
* ring all-gather of P per worker: (N - 1) x P per worker; it grows with N, an all-reduce does not;
* hierarchical (D > 1 data centres, W/D workers each): the same totals, split into intra-DC and inter-DC traffic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .hardware import Kernel, Topology


@dataclass(frozen=True)
class Transfer:
    src: object
    dst: object
    link: tuple
    nbytes: float
    keys: tuple = ()
    mode: str = "copy"          # "copy": receiver takes the sender's value; "reduce": receiver adds it
    cap: float | None = None    # per-flow rate limit, bytes/s (a pluggable module's port)


@dataclass(frozen=True)
class Process:
    node: object
    module: tuple
    nbytes: float
    kernel: Kernel
    consumes: tuple = ()
    produces: tuple = ()        # empty: timing only (part of a job another op accounts for)


@dataclass
class Schedule:
    phases: list = field(default_factory=list)   # [(name, [round, ...])], round = [op, ...]

    def add(self, name: str, rounds: list) -> None:
        self.phases.append((name, [r for r in rounds if r]))

    @property
    def rounds(self) -> list:
        return [r for _, rs in self.phases for r in rs]

    def transfers(self):
        return [op for r in self.rounds for op in r if isinstance(op, Transfer)]

    def processes(self):
        return [op for r in self.rounds for op in r if isinstance(op, Process)]

    def sent_by(self, node) -> float:
        return sum(t.nbytes for t in self.transfers() if t.src == node)

    def received_by(self, node) -> float:
        return sum(t.nbytes for t in self.transfers() if t.dst == node)


def _ring_next(topo: Topology, w: int) -> int:
    """The next worker on w's intra-DC ring."""
    d, j = divmod(w, topo.per_dc)
    return d * topo.per_dc + (j + 1) % topo.per_dc


def _peer(topo: Topology, w: int, step: int = 1) -> int:
    """Worker w's counterpart in the DC ``step`` along the ring of DCs."""
    d, j = divmod(w, topo.per_dc)
    return ((d + step) % topo.dcs) * topo.per_dc + j


# ── hierarchical ring all-reduce ─────────────────────────────────────────────────────────────────────
def allreduce(topo: Topology, nbytes: float, cap: float | None = None) -> Schedule:
    """Reduce-scatter in each DC, all-reduce of each shard across DCs, all-gather in each DC.

    The buffer is cut into per_dc x dcs pieces, keyed ("c", c, s): c = the intra-DC chunk, s = its piece
    across DCs. With one DC it is the flat ring (2 (N-1) rounds of B/N); with one worker per DC it is the flat
    ring over the DCs."""
    n, D, W = topo.per_dc, topo.dcs, topo.workers
    piece = nbytes / (n * D)
    s = Schedule()

    def chunk(c):
        return tuple(("c", c, x) for x in range(D))

    rs = [[Transfer(w, _ring_next(topo, w), ("up", w), piece * D, chunk((w % n - r) % n), "reduce", cap)
           for w in range(W)] for r in range(n - 1)]
    s.add("intra-DC reduce-scatter", rs)
    # after it, local worker j holds chunk (j + 1) % n summed over its DC
    inter = []
    for r in range(D - 1):          # reduce-scatter across DCs, piece by piece
        inter.append([Transfer(w, _peer(topo, w), ("wan", topo.dc_of(w)), piece,
                               (("c", (w % n + 1) % n, (topo.dc_of(w) - r) % D),), "reduce", cap)
                      for w in range(W)])
    for r in range(D - 1):          # all-gather across DCs
        inter.append([Transfer(w, _peer(topo, w), ("wan", topo.dc_of(w)), piece,
                               (("c", (w % n + 1) % n, (topo.dc_of(w) + 1 - r) % D),), "copy", cap)
                      for w in range(W)])
    s.add("inter-DC all-reduce", inter)
    ag = [[Transfer(w, _ring_next(topo, w), ("up", w), piece * D, chunk((w % n + 1 - r) % n), "copy", cap)
           for w in range(W)] for r in range(n - 1)]
    s.add("intra-DC all-gather", ag)
    return s


def allreduce_keys(topo: Topology) -> list:
    return [("c", c, x) for c in range(topo.per_dc) for x in range(topo.dcs)]


# ── hierarchical ring all-gather ─────────────────────────────────────────────────────────────────────
def allgather(topo: Topology, payload: float, cap: float | None = None) -> Schedule:
    """Each worker's block ("blk", w) of ``payload`` bytes reaches every worker: all-gather in each DC, then
    across DCs between counterparts, then the remote blocks around each DC. (N - 1) x payload per worker."""
    n, D, W = topo.per_dc, topo.dcs, topo.workers
    s = Schedule()

    def local(w, idx):          # the worker with local index idx in w's DC
        return topo.dc_of(w) * n + idx

    s.add("intra-DC all-gather", [[Transfer(w, _ring_next(topo, w), ("up", w), payload,
                                            (("blk", local(w, (w % n - r) % n)),), "copy", cap)
                                   for w in range(W)] for r in range(n - 1)])
    s.add("inter-DC all-gather", [[Transfer(w, _peer(topo, w), ("wan", topo.dc_of(w)), payload,
                                            (("blk", _peer(topo, w, -r)),), "copy", cap)
                                   for w in range(W)] for r in range(D - 1)])
    rounds = []
    for r in range(n - 1 if D > 1 else 0):
        rnd = []
        for w in range(W):
            j = (w % n - r) % n                      # forward what came from local worker j's counterparts
            keys = tuple(("blk", _peer(topo, local(w, j), k)) for k in range(1, D))
            rnd.append(Transfer(w, _ring_next(topo, w), ("up", w), payload * (D - 1), keys, "copy", cap))
        rounds.append(rnd)
    s.add("intra-DC all-gather of remote blocks", rounds)
    return s


# ── the hypothetical fixed-band all-reduce in the network ────────────────────────────────────────────
def switch_band_allreduce(topo: Topology, dense: float, band: float, band_k: Kernel, sum_k: Kernel) -> Schedule:
    """The fixed band computed and summed at each DC's edge switch (one module per worker-facing port).

    1. every worker sends its dense delta to its switch;
    2. each port's module computes that worker's band (``band_k``);
    3. the switch's modules sum the local bands (``sum_k``, split evenly over the modules) and, in the same
       round, **return each worker its own optical band** (the loopback the error-feedback fix needs: an
       all-reduce returns only the sum, so without it the sender cannot feed back what was sent);
    4. a ring all-reduce of the band sum between the DCs' switches;
    5. each switch sends the global sum to its workers."""
    n, D, W = topo.per_dc, topo.dcs, topo.workers
    sw = [("sw", d) for d in range(D)]
    s = Schedule()
    s.add("dense delta to the switch", [[Transfer(w, sw[topo.dc_of(w)], ("up", w), dense, (("dense", w),))
                                         for w in range(W)]])
    s.add("band computed at the switch", [[Process(sw[topo.dc_of(w)], ("mod", topo.dc_of(w), w % n), dense, band_k,
                                                   (("dense", w),), (("band", w),)) for w in range(W)]])
    sums = tuple(("s", x) for x in range(D))
    rnd = []
    for d in range(D):
        local = tuple(("band", d * n + j) for j in range(n))
        for j in range(n):
            # the sum is one job split over the n modules; the first op carries its data, the rest are timing only
            rnd.append(Process(sw[d], ("mod", d, j), band, sum_k, local if j == 0 else (), sums if j == 0 else ()))
            rnd.append(Transfer(sw[d], d * n + j, ("down", d * n + j), band, (("band", d * n + j),)))
    s.add("sum at the switch, and each worker's own band back", [rnd])
    inter = []
    for r in range(D - 1):
        inter.append([Transfer(sw[d], sw[(d + 1) % D], ("wan", d), band / D, (("s", (d - r) % D),), "reduce")
                      for d in range(D)])
    for r in range(D - 1):
        inter.append([Transfer(sw[d], sw[(d + 1) % D], ("wan", d), band / D, (("s", (d + 1 - r) % D),), "copy")
                      for d in range(D)])
    s.add("inter-DC all-reduce of the band sum", inter)
    s.add("global sum to the workers", [[Transfer(sw[topo.dc_of(w)], w, ("down", w), band, sums) for w in range(W)]])
    return s


def pluggable_band(topo: Topology, dense: float, band_k: Kernel, module_key=lambda w: ("mod", w)) -> list:
    """The band computed by each worker's pluggable module as its delta leaves: one round."""
    return [[Process(w, module_key(w), dense, band_k, (("dense", w),), (("blk", w),)) for w in range(topo.workers)]]


# ── semantic check ───────────────────────────────────────────────────────────────────────────────────
class ScheduleError(AssertionError):
    pass


def replay(schedule: Schedule, init: dict) -> dict:
    """Replay a schedule on contributor sets. ``init``: node -> {key: frozenset(contributors)}. Every op in a
    round reads the state as it was when the round began (the ops run at the same time)."""
    state = {node: dict(v) for node, v in init.items()}
    for rnd in schedule.rounds:
        snap = {node: dict(v) for node, v in state.items()}
        for op in rnd:
            if isinstance(op, Transfer):
                dst = state.setdefault(op.dst, {})
                for k in op.keys:
                    if k not in snap.get(op.src, {}):
                        raise ScheduleError(f"{op.src} sends {k} it does not hold")
                    v = snap[op.src][k]
                    if op.mode == "reduce":
                        if k in dst and dst[k] & v:
                            raise ScheduleError(f"{op.dst} would count {sorted(dst[k] & v)} twice in {k}")
                        dst[k] = dst.get(k, frozenset()) | v
                    else:
                        dst[k] = v
            elif op.produces:
                node = state.setdefault(op.node, {})
                missing = [k for k in op.consumes if k not in snap.get(op.node, {})]
                if missing:
                    raise ScheduleError(f"{op.node} processes {missing} it does not hold")
                v = frozenset().union(*(snap[op.node][k] for k in op.consumes))
                for k in op.produces:
                    node[k] = v
    return state
