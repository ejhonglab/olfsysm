#!/usr/bin/env python3
"""Tests for the `fit_mb_model` APL time constant kwarg (`APL_DYNAMICS_PARAMS`: `apl_taum`),
and `optimize_mb_model_params`' `condition_spike_times` and `extra_params`.
"""

from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import olfsysm as osm

from hong2p.types import ParamDict

from pyolfsysm.mb_model import (fit_mb_model, megamat_orn_deltas,
    make_fixed_param_kc_simulator, fixed_params_to_fit_mb_model_kws, format_model_params,
    onestep_lr_cache_key, assert_fit_outputs_equal, optimize_mb_model_params,
    max_apl_unit_coupling, APL_DYNAMICS_PARAMS, apl_dynamics_param_dict, _get_fixed_params,
    FitMBModelOutputs
)

from .conftest import test_data_dir
from .test_model_timing import TIMING_MODEL_KWS, _model_inputs, _odor_columns, _quiet


pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

# olfsysm's default
APL_DEFAULT = dict(apl_taum=0.05)


@pytest.fixture(scope='module')
def orn_deltas() -> pd.DataFrame:
    return megamat_orn_deltas()


def _id(kws: ParamDict) -> str:
    return format_model_params({k: v for k, v in kws.items()})


@pytest.fixture(scope='module', params=TIMING_MODEL_KWS, ids=_id)
def taum_model(request, orn_deltas
    ) -> Tuple[pd.DataFrame, ParamDict, FitMBModelOutputs]:
    """Returns `orn_deltas`, model kwargs and tuned outputs w/o `apl_taum` passed (do not
    modify). The models of test_model_timing: the one-row-per-KC hemibrain model, and a
    reduced claw + bouton model w/ connectome APL weights, KC-KC inhibition, a refractory
    period and adaptation.
    """
    orn_deltas, kws = _model_inputs(orn_deltas, request.param)
    return orn_deltas, kws, fit_mb_model(orn_deltas, **kws)


def _pop_params(ret: FitMBModelOutputs, expected: ParamDict) -> FitMBModelOutputs:
    responses, spike_counts, wPNKC, params = ret
    params = dict(params)
    for k, v in expected.items():
        x = params.pop(k)
        assert type(x) is float and x == v
    return responses, spike_counts, wPNKC, params


def _apl_off(initial: ParamDict) -> ParamDict:
    """APL output off (APL->KC and APL->bouton scales 0), everything else as `initial`."""
    return {k: 0.0 for k in ('wAPLKC_scale', 'wAPLPN_scale') if k in initial}


def test_apl_taum_default_unchanged(taum_model):
    """Tests that passing olfsysm's default APL time constant explicitly changes no tuned
    output (and records it), and that it is not recorded when not passed.
    """
    orn_deltas, kws, ret = taum_model
    assert not any(k in ret[3] for k in APL_DYNAMICS_PARAMS)
    assert osm.ModelParams().kc.apl_taum == APL_DEFAULT['apl_taum']

    ret2 = fit_mb_model(orn_deltas, **APL_DEFAULT, **kws)
    assert_fit_outputs_equal(ret, _pop_params(ret2, APL_DEFAULT))


@pytest.mark.parametrize('case, kws', [('default', dict()),
    ('connectome-APL_True', dict(use_connectome_APL_weights=True))], ids=['default',
    'connectome-APL_True']
)
def test_apl_taum_default_repro_committed(case, kws):
    """Tests models w/ olfsysm's default APL time constant passed explicitly still reproduce
    committed reference outputs (the latest `test_data/reference_outputs_for_repro` run with
    the case, as `test_fitandplot_repro` uses), exactly.
    """
    root = test_data_dir / 'reference_outputs_for_repro'
    ref_dir = sorted(d for d in root.iterdir() if (d / case).is_dir())[-1] / case
    committed = pd.read_parquet(ref_dir / 'spike_counts.parquet')
    orn_deltas = pd.read_parquet(ref_dir / 'orn_deltas.parquet')

    responses, spike_counts, _, params = fit_mb_model(orn_deltas, **APL_DEFAULT, **kws,
        **_quiet
    )
    assert params['apl_taum'] == APL_DEFAULT['apl_taum']
    assert spike_counts.equals(committed)
    assert responses.equals(pd.read_parquet(ref_dir / 'responses.parquet'))


def test_apl_taum_fixed_params(taum_model):
    """Tests the APL time constant w/ the tuned parameters fixed: olfsysm gets it, the
    simulator matches `fit_mb_model` w/ it, a slower APL (0.25 s) gives more APL-on spikes
    and a faster one (0.01 s) fewer, and APL-off outputs (APL->KC / bouton scales 0) are the
    same as w/ the default.
    """
    orn_deltas, kws, ret = taum_model
    params = _get_fixed_params(ret[3])
    # (odors w/ the largest spike counts)
    sub = _odor_columns(orn_deltas, ret[1].sum().sort_values(ascending=False).index[:4])
    fixed_kws = {**kws, **fixed_params_to_fit_mb_model_kws(params)}

    sim0, initial = make_fixed_param_kc_simulator(sub, initial_params=params, **kws)
    off = _apl_off(initial)
    _, s0 = sim0(dict())
    r0_off, s0_off = sim0(off)
    assert s0.values.sum() < s0_off.values.sum()

    for taum, more in ((0.25, True), (0.01, False)):
        r_fit, s_fit, _, p = fit_mb_model(sub, apl_taum=taum, return_olfsysm_vars=True,
            **fixed_kws
        )
        assert p['mp'].kc.apl_taum == taum and p['apl_taum'] == taum
        # (the feedforward APL, not simulated, keeps its own)
        assert p['mp'].kc.ignore_ffapl

        sim, initial2 = make_fixed_param_kc_simulator(sub, initial_params=params,
            apl_taum=taum, **kws
        )
        assert initial2 == initial
        r1, s1 = sim(dict())
        assert r1.equals(r_fit) and s1.equals(s_fit)
        assert (s1.values.sum() > s0.values.sum()) == more
        assert (s1.values.sum() < s0.values.sum()) == (not more)

        r1_off, s1_off = sim(off)
        assert s1_off.equals(s0_off) and r1_off.equals(r0_off)
        print(f'apl_taum {taum}: {s1.values.sum():.0f} APL-on spikes ({s0.values.sum():.0f} '
            f'w/ the default 0.05 s), {s1_off.values.sum():.0f} APL-off'
        )


def test_apl_taum_tuning(orn_deltas):
    """Tests that the tuning follows the APL time constant: the hemibrain model tuned w/ a
    slower APL reaches the same response rate as w/ the default, w/ a different APL weight.
    """
    kws = dict(weight_divisor=20, **_quiet)
    r0, _, _, p0 = fit_mb_model(orn_deltas, **kws)
    r1, _, _, p1 = fit_mb_model(orn_deltas, apl_taum=0.25, **kws)
    rate0 = float(r0.values.mean())
    rate1 = float(r1.values.mean())
    assert abs(rate1 - rate0) < 0.1 * rate0
    assert _get_fixed_params(p1)['wAPLKC_scale'] != _get_fixed_params(p0)['wAPLKC_scale']
    assert p1['apl_taum'] == 0.25 and 'apl_taum' not in p0


def test_apl_taum_invalid(orn_deltas):
    """Tests invalid APL time constants raise ValueError.
    """
    for bad in (0.0, -0.05, 1e-4, 4.9e-4, np.nan, np.inf):
        with pytest.raises(ValueError, match='apl_taum'):
            apl_dynamics_param_dict(apl_taum=bad)
        with pytest.raises(ValueError, match='apl_taum'):
            fit_mb_model(orn_deltas, weight_divisor=20, apl_taum=bad, **_quiet)

    assert apl_dynamics_param_dict() == dict()
    assert apl_dynamics_param_dict(apl_taum=1) == dict(apl_taum=1.0)
    # (one time step is the smallest olfsysm integrates stably)
    dt = osm.ModelParams().time_dt
    assert apl_dynamics_param_dict(apl_taum=dt) == dict(apl_taum=dt)


def test_apl_taum_model_id():
    """Tests model IDs / LR cache keys only change when `apl_taum` is passed.
    """
    kws = dict(weight_divisor=20)
    none_kws = {**kws, **{k: None for k in APL_DYNAMICS_PARAMS}}
    assert format_model_params(none_kws) == format_model_params(kws)
    assert onestep_lr_cache_key(none_kws) == onestep_lr_cache_key(kws)

    variants = (dict(), APL_DEFAULT, dict(apl_taum=0.25), dict(apl_taum=0.01),
        dict(apl_taum=0.25, time_stim_end=2.0, time_end=2.25)
    )
    ids = {format_model_params({**kws, **x}) for x in variants}
    assert len(ids) == len(variants)
    lr_keys = {onestep_lr_cache_key({**kws, **x}) for x in variants}
    assert len(lr_keys) == len(variants)


def test_max_apl_unit_coupling_apl_taum():
    """Tests the coupled APL units' limit follows the APL time constant.
    """
    m0 = max_apl_unit_coupling()
    assert max_apl_unit_coupling(apl_taum=APL_DEFAULT['apl_taum']) == m0
    dt = osm.ModelParams().time_dt
    m1 = max_apl_unit_coupling(apl_taum=0.25)
    assert np.isclose(m1 / m0, (0.25 / dt - 1) / (APL_DEFAULT['apl_taum'] / dt - 1))


def test_optimize_condition_spike_times(taum_model):
    """Tests `optimize_mb_model_params`' `condition_spike_times`: those conditions' outputs
    reach the loss w/ their spike times (a condition sharing their simulation w/o them),
    the history and the returned outputs are as without it, and the best evaluation's spike
    times are returned and add up to its spike counts.
    """
    orn_deltas, kws, ret = taum_model
    params = _get_fixed_params(ret[3])
    sub = _odor_columns(orn_deltas, ret[1].sum().sort_values(ascending=False).index[:3])
    seen = []

    def loss(outputs):
        seen.append({c: len(o) for c, o in outputs.items()})
        r_on, s_on = outputs['on'][:2]
        r_off, s_off = outputs['off'][:2]
        assert outputs['on2'][1].equals(s_on)
        return {'loss': float(abs(r_on.values.mean() - 0.05) + abs(r_off.values.mean() - 0.1)),
            'n_on': float(s_on.values.sum()), 'n_off': float(s_off.values.sum())
        }

    off = _apl_off(params)
    # ('on2' shares the 'on' simulation)
    overrides = {'on': {}, 'off': off, 'on2': {}}
    common = dict(loss_fn=loss, condition_param_overrides=overrides,
        params_to_optimize=['fixed_thr'], initial_params=params, maxfev=3, verify=False,
        verbose=False, **kws
    )
    res0 = optimize_mb_model_params(sub, **common)
    assert all(s == {'on': 2, 'off': 2, 'on2': 2} for s in seen)
    seen.clear()

    res1 = optimize_mb_model_params(sub, condition_spike_times=['on', 'off'], **common)
    assert len(seen) == len(res0['history']) and all(
        s == {'on': 3, 'off': 3, 'on2': 2} for s in seen
    )
    pd.testing.assert_frame_equal(res0['history'].drop(columns='time_s'),
        res1['history'].drop(columns='time_s')
    )
    assert 'condition_spike_times' not in res0
    assert set(res1['condition_spike_times']) == {'on', 'off'}
    for c in overrides:
        o0, o1 = res0['condition_outputs'][c], res1['condition_outputs'][c]
        assert len(o1) == 2 and o1[0].equals(o0[0]) and o1[1].equals(o0[1])

    for c, st in res1['condition_spike_times'].items():
        counts = np.zeros(res1['condition_outputs'][c][1].shape)
        np.add.at(counts, (st.kc.to_numpy(), st.odor.to_numpy()), 1)
        assert np.array_equal(counts, res1['condition_outputs'][c][1].to_numpy(dtype=float))

    with pytest.raises(ValueError, match='condition_spike_times'):
        optimize_mb_model_params(sub, **{**common, 'condition_spike_times': ['nope']})
    with pytest.raises(ValueError, match='condition_spike_times'):
        optimize_mb_model_params(sub, loss_fn=lambda r, s: 0.0, initial_params=params,
            params_to_optimize=['fixed_thr'], maxfev=1, condition_spike_times=['on'], **kws
        )


def test_optimize_extra_params(taum_model):
    """Tests `optimize_mb_model_params`' `extra_params`: a parameter outside the simulator (here a
    factor on its unscaled APL->KC weights, set in place through the closure of `simulate`) is
    set before every evaluation and optimized with the simulator's; at its initial value (a
    no-op) the first evaluation is the one without it; the history, 'params' and
    'extra_params' carry its values, the conditions' parameters and 'fit_mb_model_kws' do not;
    and the refusals.
    """
    orn_deltas, kws, ret = taum_model
    params = _get_fixed_params(ret[3])
    sub = _odor_columns(orn_deltas, ret[1].sum().sort_values(ascending=False).index[:3])

    def loss(outputs):
        r_on, s_on = outputs['on']
        r_off, _ = outputs['off']
        return {'loss': float(abs(r_on.values.mean() - 0.05) + abs(r_off.values.mean() - 0.1)),
            'n_on': float(s_on.values.sum())
        }

    common = dict(loss_fn=loss, condition_param_overrides={'on': {}, 'off': _apl_off(params)},
        params_to_optimize=['fixed_thr'], initial_params=params, maxfev=4, verify=False,
        verbose=False, **kws
    )
    res0 = optimize_mb_model_params(sub, **common)
    calls = []
    base = {}

    def set_factor(simulate, d):
        cells = dict(zip(simulate.__code__.co_freevars, simulate.__closure__))
        w = cells['wAPLKC_unscaled'].cell_contents
        base.setdefault('w', w.copy())
        w[...] = base['w'] * (1.0 + d)
        calls.append(d)

    extra = {'factor': dict(initial=0.0, bounds=(0.0, 4.0), set=set_factor)}
    res1 = optimize_mb_model_params(sub, extra_params=extra, **common)
    h0, h1 = res0['history'], res1['history']
    assert 'factor' not in h0 and 'extra_params' not in res0
    # at its initial value the extra parameter changes nothing: the first evaluation is the same
    assert calls[0] == 0.0 and h1['factor'].iloc[0] == 0.0
    assert h1['loss'].iloc[0] == h0['loss'].iloc[0] and h1['n_on'].iloc[0] == h0['n_on'].iloc[0]
    assert h1['fixed_thr'].iloc[0] == h0['fixed_thr'].iloc[0]
    # set before every evaluation, with the value the history records
    assert len(calls) == len(h1) and list(h1['factor']) == calls
    # the simplex's step on it (the threshold at its initial value) inhibits more: fewer APL-on spikes
    step = h1[(h1.factor > 0) & np.isclose(h1.fixed_thr, params['fixed_thr'], rtol=1e-12, atol=0)]
    assert len(step) and (step.n_on < h0['n_on'].iloc[0]).all()
    assert res1['extra_params'] == {'factor': res1['params']['factor']}
    assert res1['initial_extra_params'] == {'factor': 0.0}
    assert 'factor' in res1['params_to_optimize']
    assert all('factor' not in p for p in res1['condition_params'].values())
    assert 'factor' not in res1['fit_mb_model_kws']
    assert 0.0 <= res1['params']['factor'] <= 4.0

    bad = [dict(extra_params={'fixed_thr': dict(extra['factor'])}),
        dict(extra_params={'factor': dict(initial=0.0, bounds=(0.0, 4.0))}),
        dict(extra_params={'factor': dict(initial=5.0, bounds=(0.0, 4.0), set=set_factor)}),
        dict(extra_params={'factor': dict(initial=0.0, bounds=(0.0, 4.0), set=None)}),
        dict(extra_params=extra, verify=True),
    ]
    for b in bad:
        with pytest.raises(ValueError):
            optimize_mb_model_params(sub, **{**common, **b})
