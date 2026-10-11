"""Time and energy to a target loss, and break-even points found by bisection."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from . import inputs
from .engine import Result, simulate
from .model import Config, plan, steps_to_target


@dataclass
class Outcome:
    cfg: Config
    target: float
    steps: float | None        # None: not every seed reached the target in 3,000 steps
    result: Result

    @property
    def step_time(self) -> float:
        return self.result.step_time

    @property
    def time(self) -> float:
        """Seconds to the target (inf when the measured runs never reached it)."""
        return math.inf if self.steps is None else self.steps * self.step_time

    @property
    def energy_per_step(self) -> dict:
        return self.result.energy()

    @property
    def energy(self) -> float:
        """Joules to the target, all workers, links and modules."""
        return math.inf if self.steps is None else self.steps * sum(self.energy_per_step.values())


def evaluate(cfg: Config, target: float, syncs: int = 1, background: float = 0.0, seed: int = 0) -> Outcome:
    p = plan(cfg)
    return Outcome(cfg, target, steps_to_target(cfg, target), simulate(p, syncs=syncs, background=background, seed=seed))


# ── bisection ────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class BreakEven:
    x: float | None            # the crossing, or None when the sign never changes in [lo, hi]
    lo: float
    hi: float
    f_lo: float
    f_hi: float
    iterations: int

    @property
    def found(self) -> bool:
        return self.x is not None


def bisect(f, lo: float, hi: float, log: bool = True, rel_tol: float = 1e-4, max_iter: int = 200) -> BreakEven:
    """The x in [lo, hi] where f changes sign, to a relative tolerance (log space by default). f may return
    +/-inf (a variant that never reaches the target). With no sign change, x is None."""
    a, b = lo, hi
    fa, fb = f(a), f(b)
    if math.isnan(fa) or math.isnan(fb):
        return BreakEven(None, lo, hi, fa, fb, 0)
    if fa == 0:
        return BreakEven(a, lo, hi, fa, fb, 0)
    if fb == 0:
        return BreakEven(b, lo, hi, fa, fb, 0)
    if (fa > 0) == (fb > 0):
        return BreakEven(None, lo, hi, fa, fb, 0)
    it = 0
    while it < max_iter and (b - a) > rel_tol * max(abs(a), abs(b)):
        m = math.sqrt(a * b) if log and a > 0 else (a + b) / 2
        fm = f(m)
        it += 1
        if fm == 0:
            a = b = m
            break
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
        else:
            b, fb = m, fm
    return BreakEven(math.sqrt(a * b) if log and a > 0 else (a + b) / 2, lo, hi, f(lo), f(hi), it)


def gap(cfg_x: Config, cfg_y: Config, target: float, knob) -> callable:
    """f(v) = time to target of X minus Y after ``knob(cfg, v)`` is applied to both (negative: X is faster)."""
    def f(v):
        tx = evaluate(knob(cfg_x, v), target).time
        ty = evaluate(knob(cfg_y, v), target).time
        if math.isinf(tx) and math.isinf(ty):
            return math.nan             # neither reaches the target: no comparison
        return tx - ty
    return f


def knob_inter_bw(cfg: Config, bw: float) -> Config:
    return replace(cfg, topo=cfg.topo.with_inter(bw))


def knob_gflops(cfg: Config, g: float) -> Config:
    return replace(cfg, module=replace(cfg.module, gflops=g))


def measured_enob(x: float) -> int:
    """The highest measured ENOB at or below x: ENOB maps to convergence only through the runs at 6, 8, 10, 12,
    so between them the steps are those of the run below (a step function, no interpolation)."""
    below = [e for e in inputs.ENOBS if e <= x + 1e-9]
    return below[-1] if below else inputs.ENOBS[0]


def knob_enob(cfg: Config, x: float) -> Config:
    return replace(cfg, module=replace(cfg.module, enob=measured_enob(x)))
