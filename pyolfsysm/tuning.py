"""Re-tuning a fixed-parameter model to hold its response rates, and running one parameter
set on a population of simulators (e.g. one per re-drawn PN->KC wiring).

The simulators are functions of a parameter dict returning (responses, spike counts), KCs
x odors, as `mb_model.make_fixed_param_kc_simulator` builds them: `bisect_factor` finds the
factor on a parameter (or on several, tied) at which the number of responses hits a
target, `hold_rates` re-tunes the somatic threshold with APL off and then the APL scales
with APL on to hold both response counts, and `simulate_population` runs every member of
a population with APL on and off.

Moved from the KC-remapping experiment queue (model_experiment_queue's E69_kc_firing_ceiling
`bisect_factor`, E77_input_screen `hold_rates`, E72_wiring_population
`simulate_population` / `mean_distances`), 2026-11-12; the queue keeps its own wrappers
with its defaults (its fitted model's parameters and targets, the APL-weight handling of
its breadth-weighted model).
"""
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


Simulator = Callable[[Dict[str, float]], Tuple[pd.DataFrame, pd.DataFrame]]

# the APL-off condition of the fixed-parameter simulators (APL->KC and APL->bouton weights
# at 0), and the scales tied to APL->KC when the APL scale is re-tuned (KC->APL and
# bouton->APL)
APL_OFF: Dict[str, float] = {'wAPLKC_scale': 0.0, 'wAPLPN_scale': 0.0}
TIED_APL_SCALES: Tuple[str, ...] = ('wAPLKC_scale', 'wKCAPL_scale', 'wPNAPL_scale')
BISECT_ITERS = 20


def n_responses(responses: pd.DataFrame) -> int:
    """Returns the number of responses of a binarized (KCs x odors) response matrix."""
    return int(responses.to_numpy(dtype=float).sum())


def bisect_factor(run: Callable[[float], Tuple[pd.DataFrame, pd.DataFrame]],
    target_n: int, size: int, bracket: Tuple[float, float], step0: float, label: str,
    max_iters: int = BISECT_ITERS) -> Tuple[float, tuple, List[dict], bool]:
    """Returns the factor at which `run(factor)`'s number of responses is `target_n`.

    The number of responses must fall with the factor (e.g. a threshold, or APL scales).
    Searched on a log scale: from 1, steps of `step0` (doubling) in the direction of the
    target until it is bracketed (within `bracket`), then regula falsi on the number of
    responses with the Illinois modification (each new point kept within the middle 90 %
    of the bracket), until the count is hit or the bracket is narrower than 1e-9 (log
    units) or `max_iters` evaluations. Every evaluation is printed with `label`.

    Args:
        run: factor -> (responses, spike counts).
        target_n: the number of responses (`n_responses`) to hit.
        size: the number of KC x odor entries (for the printed and recorded rates).
        bracket: the factors searched.
        step0: the first step's factor.

    Returns (factor, outputs, history, at_bound): the evaluated point nearest the target
    if it cannot be hit exactly; at_bound when `bracket` does not contain it; history one
    dict per evaluation (stage, factor, n_responses, rate, target_n).
    """
    hist, best = [], None

    def ev(x: float) -> int:
        nonlocal best
        f = float(np.exp(x))
        out = run(f)
        n = n_responses(out[0])
        hist.append(dict(stage=label, factor=f, n_responses=n, rate=n / size,
            target_n=target_n
        ))
        print(f'  {label}: factor {f:.6g} -> {n} responses (rate {n / size:.5f}; target '
            f'{target_n})', flush=True
        )
        if best is None or abs(n - target_n) < abs(best[1] - target_n):
            best = (f, n, out)
        return n - target_n

    lo_end, hi_end = np.log(bracket[0]), np.log(bracket[1])
    y0 = ev(0.0)
    if y0 != 0:
        # too many responses (y > 0): a larger factor
        direction = 1.0 if y0 > 0 else -1.0
        x_prev, y_prev, step = 0.0, y0, np.log(step0)
        while True:
            x = float(np.clip(x_prev + direction * step, lo_end, hi_end))
            y = ev(x)
            if y == 0 or np.sign(y) != np.sign(y0):
                break
            if x in (lo_end, hi_end):
                return best[0], best[2], hist, True
            x_prev, y_prev, step = x, y, 2 * step
        if y != 0:
            # a: y > 0 (smaller factor), b: y < 0
            (xa, ya), (xb, yb) = sorted([(x_prev, y_prev), (x, y)])
            last = 0
            for _ in range(max_iters):
                if xb - xa < 1e-9:
                    break
                xm = xa - ya * (xb - xa) / (yb - ya)
                xm = float(np.clip(xm, xa + 0.05 * (xb - xa), xb - 0.05 * (xb - xa)))
                ym = ev(xm)
                if ym == 0:
                    break
                if ym > 0:
                    xa, ya = xm, ym
                    if last == 1:
                        yb /= 2
                    last = 1
                else:
                    xb, yb = xm, ym
                    if last == -1:
                        ya /= 2
                    last = -1
    f, n, out = best
    return f, out, hist, False


def hold_rates(sim: Simulator, params: Dict[str, float], *, target_off: int,
    target_on: int, size: int, thr_key: str = 'fixed_thr',
    off_params: Optional[Dict[str, float]] = None,
    tied_on: Sequence[str] = TIED_APL_SCALES, thr_bracket: Tuple[float, float] = (0.1, 10.0),
    thr_step: float = 1.1, apl_bracket: Tuple[float, float] = (0.1, 10.0),
    apl_step: float = 1.25, label: str = '',
    after_off: Optional[Callable[[pd.DataFrame, pd.DataFrame], None]] = None,
    max_iters: int = BISECT_ITERS) -> dict:
    """Returns a fixed-parameter simulator re-tuned to hold both response counts.

    First the somatic threshold (`thr_key`) is bisected with APL off (`off_params`,
    default `APL_OFF`) to `target_off` responses, then the APL scales `tied_on` (one
    factor on all) to `target_on` responses at that threshold (`bisect_factor`); the other
    parameters stay at `params`'.

    Args:
        sim: params -> (responses, spike counts).
        params: the absolute parameters the factors multiply.
        target_off, target_on: response counts (two or more spikes) to hold.
        size: KCs x odors (for the printed rates).
        thr_bracket, thr_step, apl_bracket, apl_step: the bisections' brackets and first
            steps.
        label: printed with each bisection step.
        after_off: called with the threshold-held APL-off (responses, spike counts) before
            the APL scale is bisected (e.g. to change the simulator's APL weights in place
            from that run).
        max_iters: `bisect_factor`'s.

    Returns dict with 'params' (absolute, APL on), 'params_off' (the APL-off run's),
    'thr_factor', 'apl_factor', 'n_on', 'n_off' (responses), 'target_on', 'target_off',
    'at_bound_thr', 'at_bound_apl', 'n_sims', 'history' (both bisections), 'on' / 'off'
    ((responses, spike counts)).
    """
    p = dict(params)
    off_q = dict(APL_OFF if off_params is None else off_params)
    g, (r_off, s_off), h_off, b_off = bisect_factor(
        lambda f: sim(dict(p, **{thr_key: p[thr_key] * f}, **off_q)), target_off, size,
        thr_bracket, thr_step, f'{label} threshold', max_iters
    )
    thr = p[thr_key] * g
    if after_off is not None:
        after_off(r_off, s_off)
    x, (r_on, s_on), h_on, b_on = bisect_factor(
        lambda f: sim(dict(p, **{thr_key: thr}, **{k: p[k] * f for k in tied_on})),
        target_on, size, apl_bracket, apl_step, f'{label} APL scale', max_iters
    )
    on_params = dict(p, **{thr_key: thr}, **{k: p[k] * x for k in tied_on})
    return dict(params=on_params, params_off=dict(p, **{thr_key: thr}, **off_q),
        thr_factor=float(g), apl_factor=float(x), n_on=_n_resp(r_on), n_off=_n_resp(r_off),
        target_on=target_on, target_off=target_off, at_bound_thr=bool(b_off),
        at_bound_apl=bool(b_on), n_sims=len(h_off) + len(h_on), history=h_off + h_on,
        on=(r_on, s_on), off=(r_off, s_off)
    )


def _n_resp(responses: pd.DataFrame) -> int:
    return int((responses.to_numpy(dtype=float) > 0.5).sum())


def simulate_population(sims: Mapping[str, Simulator], params: Mapping[str, Dict[str, float]],
    off_params: Optional[Dict[str, float]] = None) -> List[Dict[str, tuple]]:
    """Returns every member of a population of simulators run with APL on and off.

    Args:
        sims: {member: simulator} (e.g. one per re-drawn PN->KC wiring).
        params: {member: its absolute parameters} (e.g. one shared set, with tied scales
            set per member).
        off_params: the APL-off condition, applied over the member's parameters (default
            `APL_OFF`).

    Returns one {'on': (responses, spike counts), 'off': ...} per member, in `sims`' order
    (APL on run first).
    """
    off_q = dict(APL_OFF if off_params is None else off_params)
    outs = []
    for lab, sim in sims.items():
        p = params[lab]
        outs.append({'on': sim(p), 'off': sim(dict(p, **off_q))})
    return outs


def mean_distances(ds: Sequence[Dict[str, pd.Series]], conditions: Sequence[str] = ('on', 'off')
    ) -> Dict[str, pd.Series]:
    """Returns, per condition, the mean over a population's members of their pair distances
    (one {condition: pair Series} per member)."""
    return {c: pd.concat([d[c] for d in ds], axis=1).mean(axis=1) for c in conditions}
