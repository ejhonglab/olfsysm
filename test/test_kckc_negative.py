"""Tests of negative KC-KC weights (KC-KC excitation; `fit_mb_model`'s
`allow_negative_kc_kc`, olfsysm's `mp.kc.allow_negative_kc_kc`; Yang's OK of 2026-11-23,
for the queue's E132).

(a) With the flag off (the default) the outputs of KC-KC runs are those of the build
before the flag (olfsysm fa3543d), frozen in test_data/kckc_negative/flag_off.npz, and a
negative scale or matrix entry is refused (in pyolfsysm and in olfsysm's own check); (b)
with the flag on and non-negative weights the outputs are those of the flag off; (c) with
the flag on a negative weight runs through the same term, as excitation: it raises the
drive of the KC it is onto and nothing else, and a negative scale on a positive matrix is
a positive scale on the negated one.

Each KC branch of olfsysm's `sim_KC_layer` is covered: the one-row-per-KC hemibrain model
(weight_divisor=20) and the hemibrain bouton model (claws and boutons, as
test_mbc1_cells.py), at fixed parameters on 4 megamat odors, to keep them fast.

To re-freeze (only with a build whose KC-KC outputs are known good, e.g. olfsysm
fa3543d, with which it was frozen): `python test/test_kckc_negative.py --freeze`. The
freezing uses only what fa3543d's pyolfsysm has.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

import olfsysm as osm
from pyolfsysm.mb_model import (fit_mb_model, fixed_params_to_fit_mb_model_kws,
    format_model_params, make_fixed_param_kc_simulator, megamat_orn_deltas,
    onestep_lr_cache_key, KC_ID
)
from pyolfsysm.tuning import APL_OFF

pytestmark = pytest.mark.filterwarnings('ignore')

# (not from conftest, so that the file can also be run as a script, to freeze)
FROZEN = Path(__file__).resolve().parent / 'test_data' / 'kckc_negative' / 'flag_off.npz'
FLAG = 'allow_negative_kc_kc'

COMMON = dict(make_plots=False, silent=True)
# the one-row-per-KC hemibrain model (olfsysm's branch without claws)
PERKC_KWS = dict(weight_divisor=20, **COMMON)
PERKC_FIXED = dict(fixed_thr=333.0, wAPLKC=2.6)
# the hemibrain bouton model (claws and boutons; test_mbc1_cells.py's)
BOUTON_KWS = dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
    multibouton_claws='first', use_connectome_APL_weights=True, pn_claw_to_apl=True,
    n_spikes_for_response=2, **COMMON
)
BOUTON_FIXED = dict(fixed_thr=275.0, wAPLKC=0.13)
# KC-KC scales of the frozen runs (enough to lower the response rates)
PERKC_SCALE = 50.0
BOUTON_SCALE = 20.0
# the dose of the excitation tests
NEG = -2.0


@pytest.fixture(scope='module')
def orn():
    return megamat_orn_deltas().iloc[:, :4]


def bouton_simulator(orn, scale: float, **kws):
    """The bouton model's fixed-parameter simulator at KC-KC scale `scale`."""
    initial = dict(fixed_thr=BOUTON_FIXED['fixed_thr'],
        wAPLKC_scale=BOUTON_FIXED['wAPLKC'], wKCKC_scale=scale
    )
    return make_fixed_param_kc_simulator(orn, initial_params=initial, **BOUTON_KWS,
        **kws
    )


def frozen_runs(orn, **kws):
    """{name: (responses, spike counts, params or None)} of the frozen KC-KC runs, with
    `kws` (e.g. the flag) passed to each.

    perkc_on / perkc_off: the one-row-per-KC model with KC-KC inhibition, APL on / APL's
    output at 0; bouton_on: the bouton model; bouton_off: its simulator with APL's output
    weights at 0 (`APL_OFF`).
    """
    out = dict()
    for name, apl in (('perkc_on', dict()), ('perkc_off', dict(wAPLKC=0.0))):
        r, s, _, p = fit_mb_model(orn, kc_kc_inh=PERKC_SCALE,
            **{**PERKC_FIXED, **apl}, **PERKC_KWS, **kws
        )
        out[name] = (r, s, p)
    r, s, _, p = fit_mb_model(orn, kc_kc_inh=BOUTON_SCALE, **BOUTON_FIXED, **BOUTON_KWS,
        **kws
    )
    out['bouton_on'] = (r, s, p)
    simulate, params0 = bouton_simulator(orn, BOUTON_SCALE, **kws)
    off = {k: v for k, v in APL_OFF.items() if k in params0}
    assert set(off) == {'wAPLKC_scale', 'wAPLPN_scale'}
    r, s = simulate(off)
    out['bouton_off'] = (r, s, None)
    return out


def freeze() -> None:
    """Writes `FROZEN` from this build's KC-KC runs (with the flag off)."""
    orn = megamat_orn_deltas().iloc[:, :4]
    runs = frozen_runs(orn)
    arrays = dict()
    for name, (_, s, _) in runs.items():
        arrays[name] = s.to_numpy().astype(np.int32)
        arrays[f'{name}_kc_ids'] = s.index.get_level_values(KC_ID).to_numpy()
    bi = osm.build_info()
    meta = dict(olfsysm_commit=bi['git_hash'], git_dirty=bi['git_dirty'],
        cpp_dirty=bi['cpp_dirty'], source_sha256=bi['source_sha256'],
        build_host=bi['build_host'], odors=[str(c) for c in orn.columns]
    )
    arrays['meta'] = np.array(json.dumps(meta))
    FROZEN.parent.mkdir(exist_ok=True)
    np.savez_compressed(FROZEN, **arrays)
    print(f'wrote {FROZEN}: {meta}')


@pytest.fixture(scope='module')
def frozen():
    with np.load(FROZEN, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope='module')
def runs_off(orn):
    return frozen_runs(orn)


@pytest.fixture(scope='module')
def runs_on(orn):
    return frozen_runs(orn, allow_negative_kc_kc=True)


def n_diff(a: np.ndarray, b: np.ndarray) -> int:
    assert a.shape == b.shape
    return int((a != b).sum())


def test_flag_off_equals_frozen(frozen, runs_off):
    """(a) With the flag off, every KC-KC run gives the frozen spike counts exactly (0
    differing), on the same KCs, and no flag in the params."""
    meta = json.loads(str(frozen['meta']))
    assert meta['git_dirty'] == 0 and meta['cpp_dirty'] == 0
    for name, (_, s, p) in runs_off.items():
        assert np.array_equal(s.index.get_level_values(KC_ID).to_numpy(),
            frozen[f'{name}_kc_ids']
        )
        n = n_diff(s.to_numpy(), frozen[name])
        print(f'{name}: {n}/{s.size} spike counts differ from {meta["olfsysm_commit"][:10]}'
            f' ({int(s.to_numpy().sum())} spikes)'
        )
        assert n == 0
        if p is not None:
            assert FLAG not in p


def test_flag_on_nonnegative_equals_off(runs_off, runs_on):
    """(b) With the flag on and non-negative weights, the outputs are those of the flag
    off, and the params differ only by the flag."""
    for name, (r_off, s_off, p_off) in runs_off.items():
        r_on, s_on, p_on = runs_on[name]
        assert s_on.equals(s_off) and r_on.equals(r_off), name
        if p_off is not None:
            assert p_on.pop(FLAG) is True
            assert set(p_on) == set(p_off)
            assert p_on['kc_kc_inh'] == p_off['kc_kc_inh']


def test_negative_refused_with_flag_off(orn):
    """(a) Without the flag a negative scale or matrix entry is refused, in pyolfsysm
    (fit_mb_model, the simulator) and in olfsysm's own check; the flag needs kc_kc_inh;
    non-finite weights are refused with the flag too."""
    r0 = fit_mb_model(orn, kc_kc_inh=0.0, **PERKC_FIXED, **PERKC_KWS)[0]
    kc_ids = r0.index.get_level_values(KC_ID)
    neg_w = pd.DataFrame(0.0, index=kc_ids, columns=kc_ids)
    neg_w.iloc[0, 1] = -1.0
    nan_w = neg_w.copy()
    nan_w.iloc[0, 1] = np.nan
    for bad in (dict(kc_kc_inh=-1.0), dict(kc_kc_inh=1.0, _wKCKC=neg_w),
            dict(allow_negative_kc_kc=True),
            dict(kc_kc_inh=np.nan, allow_negative_kc_kc=True),
            dict(kc_kc_inh=-np.inf, allow_negative_kc_kc=True),
            dict(kc_kc_inh=1.0, kc_kc_tau=0.0, allow_negative_kc_kc=True),
            dict(kc_kc_inh=1.0, kc_kc_tau=-0.1, allow_negative_kc_kc=True),
            dict(kc_kc_inh=1.0, _wKCKC=nan_w, allow_negative_kc_kc=True)):
        with pytest.raises(ValueError):
            fit_mb_model(orn, **bad, **PERKC_FIXED, **PERKC_KWS)

    simulate, params0 = bouton_simulator(orn, BOUTON_SCALE)
    with pytest.raises(ValueError):
        simulate({'wKCKC_scale': NEG})
    # (other scales stay non-negative with the flag on)
    simulate2, _ = bouton_simulator(orn, BOUTON_SCALE, allow_negative_kc_kc=True)
    with pytest.raises(ValueError):
        simulate2({'wAPLKC_scale': -0.1})

    # olfsysm's check (in run_KC_sims; fit_sparseness has the same), on the weights
    # directly
    _, s, _, p = fit_mb_model(orn, kc_kc_inh=1.0, return_olfsysm_vars=True,
        **BOUTON_FIXED, **BOUTON_KWS
    )
    mp, rv = p['mp'], p['rv']
    assert not mp.kc.allow_negative_kc_kc
    w = p['wKCKC_unscaled']
    rv.kc.wKCKC = NEG * w
    with pytest.raises(ValueError, match='minCoeff'):
        osm.run_KC_sims(mp, rv, False)
    mp.kc.allow_negative_kc_kc = True
    osm.run_KC_sims(mp, rv, False)
    w_nan = w.copy()
    w_nan.data[0] = np.nan
    rv.kc.wKCKC = w_nan
    with pytest.raises(ValueError, match='allFinite'):
        osm.run_KC_sims(mp, rv, False)


def test_negative_scale_is_excitation(orn, runs_on):
    """(c) With the flag on a negative KC-KC scale runs and is excitation: the bouton
    model's spikes rise from KC-KC at zero (and the inhibition's at the mirrored scale),
    with APL on and with APL off; the simulator gives fit_mb_model's outputs at the
    negative scale."""
    totals = dict()
    for scale in (NEG, 0.0, -NEG):
        r, s, _, p = fit_mb_model(orn, kc_kc_inh=scale, allow_negative_kc_kc=True,
            **BOUTON_FIXED, **BOUTON_KWS
        )
        assert p['kc_kc_inh'] == scale and p['allow_negative_kc_kc'] is True
        assert np.isfinite(s.to_numpy()).all()
        totals[scale] = (int(s.to_numpy().sum()), int(r.to_numpy().sum()))
    print(f'bouton model, APL on: (spikes, responses) by KC-KC scale: {totals}')
    assert totals[NEG][0] > totals[0.0][0] > totals[-NEG][0]
    assert totals[NEG][1] >= totals[0.0][1] >= totals[-NEG][1]

    simulate, params0 = bouton_simulator(orn, BOUTON_SCALE, allow_negative_kc_kc=True)
    off = {k: v for k, v in APL_OFF.items() if k in params0}
    params = {**params0, 'wKCKC_scale': NEG}
    r, s = simulate(params)
    r2, s2, _, p2 = fit_mb_model(orn, allow_negative_kc_kc=True, **BOUTON_KWS,
        **fixed_params_to_fit_mb_model_kws(params)
    )
    assert p2['kc_kc_inh'] == NEG
    assert r.equals(r2) and s.equals(s2)
    assert s.to_numpy().sum() == totals[NEG][0]

    _, s_off_neg = simulate({**off, 'wKCKC_scale': NEG})
    _, s_off_zero = simulate({**off, 'wKCKC_scale': 0.0})
    print(f'bouton model, APL off: spikes {int(s_off_zero.to_numpy().sum())} at KC-KC 0, '
        f'{int(s_off_neg.to_numpy().sum())} at {NEG}'
    )
    assert s_off_neg.to_numpy().sum() > s_off_zero.to_numpy().sum()


def test_one_negative_weight_raises_its_kcs_drive(orn):
    """(c) One weight, onto KC `post` from KC `pre`, with APL's output at 0 (so KCs
    interact only through it): negative, `post` fires more to the odor `pre` fires most
    to; positive, less; no other KC changes. A negative scale on the matrix equals a
    positive scale on the negated matrix (the same term)."""
    apl_off = dict(PERKC_FIXED, wAPLKC=0.0)
    r0, s0, _, _ = fit_mb_model(orn, kc_kc_inh=0.0, **apl_off, **PERKC_KWS)
    kc_ids = s0.index.get_level_values(KC_ID)
    counts = s0.to_numpy()
    # the odor and KC with the most spikes, and a KC firing a few spikes to that odor
    i_pre, j = np.unravel_index(np.argmax(counts), counts.shape)
    few = np.flatnonzero((counts[:, j] >= 2) & (counts[:, j] <= 5))
    few = few[few != i_pre]
    assert len(few) > 0
    i_post = few[0]
    print(f'pre KC {kc_ids[i_pre]} ({counts[i_pre, j]} spikes to {s0.columns[j]}), post '
        f'KC {kc_ids[i_post]} ({counts[i_post, j]} spikes)'
    )

    w = pd.DataFrame(0.0, index=kc_ids, columns=kc_ids)
    w.iloc[i_post, i_pre] = 1.0
    scale = 20.0
    others = np.arange(len(kc_ids)) != i_post

    _, s_exc, _, _ = fit_mb_model(orn, kc_kc_inh=-scale, _wKCKC=w,
        allow_negative_kc_kc=True, **apl_off, **PERKC_KWS
    )
    _, s_inh, _, _ = fit_mb_model(orn, kc_kc_inh=scale, _wKCKC=w,
        allow_negative_kc_kc=True, **apl_off, **PERKC_KWS
    )
    exc, inh = s_exc.to_numpy(), s_inh.to_numpy()
    print(f'post KC to {s0.columns[j]}: {counts[i_post, j]} spikes, {exc[i_post, j]} '
        f'with weight {-scale}, {inh[i_post, j]} with {scale}'
    )
    assert np.array_equal(exc[others], counts[others])
    assert np.array_equal(inh[others], counts[others])
    assert exc[i_post, j] > counts[i_post, j] > inh[i_post, j]
    assert (exc[i_post] >= counts[i_post]).all() and (inh[i_post] <= counts[i_post]).all()

    # the same term with the sign on the matrix
    _, s_exc2, _, _ = fit_mb_model(orn, kc_kc_inh=scale, _wKCKC=-w,
        allow_negative_kc_kc=True, **apl_off, **PERKC_KWS
    )
    assert s_exc2.equals(s_exc)


def test_model_id():
    """Model IDs / LR cache keys do not change with the flag off (absent or False), and
    tell the flag and the sign of the scale apart."""
    kws = dict(weight_divisor=20, kc_kc_inh=2.0)
    assert format_model_params({**kws, 'allow_negative_kc_kc': False}) == \
        format_model_params(kws)
    assert onestep_lr_cache_key({**kws, 'allow_negative_kc_kc': False}) == \
        onestep_lr_cache_key(kws)
    ids = {format_model_params(x) for x in (kws, {**kws, 'allow_negative_kc_kc': True},
        {**kws, 'kc_kc_inh': -2.0, 'allow_negative_kc_kc': True}
    )}
    assert len(ids) == 3


if __name__ == '__main__':
    if sys.argv[1:] == ['--freeze']:
        freeze()
    else:
        raise SystemExit('usage: python test/test_kckc_negative.py --freeze')
