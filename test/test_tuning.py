"""Tests of pyolfsysm.tuning on a deterministic stand-in simulator (the bisection on the
number of responses, the two-stage re-tuning that holds both response rates, a population
run), not on the model; the queue's E77 test_simulated_identity checks `hold_rates` on the
model itself (C2 at its own targets returns factors 1 and its saved outputs)."""
import numpy as np
import pandas as pd
import pytest

from pyolfsysm import tuning as T
from .frozen_outputs import TOY_PARAMS, toy_simulator

SIZE = 200 * 16


def test_bisect_factor_hits_or_brackets_the_target():
    sim = toy_simulator(seed=7)
    p = dict(TOY_PARAMS)
    run = lambda f: sim(dict(p, fixed_thr=p['fixed_thr'] * f))
    n1 = T.n_responses(run(1.0)[0])
    for target in (n1, 500, 1200):
        f, out, hist, at_bound = T.bisect_factor(run, target, SIZE, (0.1, 10.0), 1.1, 'test')
        assert not at_bound
        n = T.n_responses(out[0])
        assert abs(n - target) <= min(abs(h['n_responses'] - target) for h in hist)
        assert hist[0]['factor'] == 1.0 and hist[0]['target_n'] == target
        assert all(h['rate'] == h['n_responses'] / SIZE for h in hist)
        if target == n1:
            assert f == 1.0 and len(hist) == 1
    # a target outside the bracket: the nearest evaluated point, at_bound
    f, out, hist, at_bound = T.bisect_factor(run, 0, SIZE, (0.5, 2.0), 1.1, 'test')
    assert at_bound and f == pytest.approx(2.0)


def test_hold_rates_holds_both_counts():
    sim = toy_simulator()
    seen = []
    res = T.hold_rates(sim, dict(TOY_PARAMS), target_off=420, target_on=160, size=SIZE,
        label='test', after_off=lambda r, s: seen.append(T.n_responses(r)))
    assert res['n_off'] == 420 and res['n_on'] == 160
    assert seen == [420]
    assert res['params_off']['wAPLKC_scale'] == 0.0 and res['params_off']['wAPLPN_scale'] == 0.0
    assert res['params']['fixed_thr'] == res['params_off']['fixed_thr'] == \
        TOY_PARAMS['fixed_thr'] * res['thr_factor']
    for k in T.TIED_APL_SCALES:
        assert res['params'][k] == TOY_PARAMS[k] * res['apl_factor']
    for k in ('wAPLPN_scale', 'wKCKC_scale'):
        assert res['params'][k] == TOY_PARAMS[k]
    assert res['n_sims'] == len(res['history'])
    # the held outputs are the simulator's at the returned parameters
    for c in ('on', 'off'):
        p = res['params'] if c == 'on' else res['params_off']
        assert res[c][1].equals(sim(p)[1])


def test_simulate_population_and_mean_distances():
    sims = {lab: toy_simulator(seed=s) for lab, s in (('a', 0), ('b', 1))}
    params = {'a': dict(TOY_PARAMS), 'b': dict(TOY_PARAMS, fixed_thr=20.0)}
    outs = T.simulate_population(sims, params)
    assert len(outs) == 2
    for o, lab in zip(outs, sims):
        assert o['on'][1].equals(sims[lab](params[lab])[1])
        assert o['off'][1].equals(sims[lab](dict(params[lab], **T.APL_OFF))[1])
    idx = pd.Index(['x', 'y'])
    ds = [{'on': pd.Series([1.0, 2.0], index=idx), 'off': pd.Series([3.0, 4.0], index=idx)},
        {'on': pd.Series([3.0, 2.0], index=idx), 'off': pd.Series([5.0, np.nan], index=idx)}]
    m = T.mean_distances(ds)
    assert m['on'].tolist() == [2.0, 2.0] and m['off'].tolist() == [4.0, 4.0]
