"""Tests of MB-C1 as feedback cells of their own (`fit_mb_model`'s `mbc1_inh`, olfsysm's
`mp.kc.use_mbc1`), alongside KC-KC inhibition: the cells' weights
(`connectome_MBC1_cell_weights`), their equivalence to the KC-KC path with MB-C1's KC x KC
matrix (`connectome_MBC1_KCKC_weights`), the threshold, the fixed-parameter simulator's
'wMBC1_scale', APL off and the refusals. On the hemibrain bouton model at fixed parameters
on 4 megamat odors (as test_mbc1_weights.py's last test), to keep them fast.
"""
import numpy as np
import pandas as pd
import pytest

from pyolfsysm.mb_model import (connectome_MBC1_weights, connectome_MBC1_KCKC_weights,
    connectome_MBC1_cell_weights, connectome_KCKC_weights, connectome_wPNKC, fit_mb_model,
    make_fixed_param_kc_simulator, fixed_params_to_fit_mb_model_kws, format_model_params,
    onestep_lr_cache_key, megamat_orn_deltas, MBC1_PARAMS, MBC1_ID, MBC1_POOLED, KC_ID,
    CLAW_ID
)
from pyolfsysm.tuning import APL_OFF

pytestmark = pytest.mark.filterwarnings('ignore')

# the hemibrain's two MB-C1_R cells
CELLS = [5813021291, 5813075020]
# the hemibrain bouton model at fixed parameters (test_mbc1_weights.py's)
KWS = dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
    use_connectome_APL_weights=True, pn_claw_to_apl=True, n_spikes_for_response=2,
    make_plots=False, silent=True
)
FIXED = dict(fixed_thr=275.0, wAPLKC=0.13)
# MB-C1's scale (`mbc1_inh`, in `kc_kc_inh`'s units), enough to lower the response rate
SCALE = 30.0
# the largest fraction of (KC, odor) spike counts allowed to differ (by one spike each)
# between two runs that are equal in exact arithmetic but sum in a different order (MB-C1's
# cells: wMBC1KC @ (wKCMBC1 @ spikes); the KC-KC path: (wMBC1KC @ wKCMBC1) @ spikes). A
# difference in the last bits of a KC's input only changes a spike when its Vm is that
# close to threshold. On viki (2026-11-23) none differed per cell, pooled or with KC-KC
# inhibition too, and 4 of 6928 (one spike each, one binary response) on the wiring's
# claws only.
MAX_DIFF_FRAC = 2e-3


@pytest.fixture(scope='module')
def orn():
    return megamat_orn_deltas().iloc[:, :4]


@pytest.fixture(scope='module')
def base(orn):
    """The model without MB-C1."""
    return fit_mb_model(orn, **KWS, **FIXED)


@pytest.fixture(scope='module')
def kc_ids(base):
    return base[0].index.get_level_values(KC_ID)


@pytest.fixture(scope='module')
def cells(orn):
    """The model with MB-C1 as its two cells, at `SCALE`."""
    return fit_mb_model(orn, mbc1_inh=SCALE, **KWS, **FIXED)


def n_diff(s1: pd.DataFrame, s2: pd.DataFrame) -> int:
    assert s1.index.equals(s2.index) and s1.columns.equals(s2.columns)
    return int((s1.to_numpy() != s2.to_numpy()).sum())


def assert_nearly_equal(s1: pd.DataFrame, s2: pd.DataFrame, what: str) -> None:
    """Spike counts equal but for at most `MAX_DIFF_FRAC` of them, by one spike each
    (summation order)."""
    n = n_diff(s1, s2)
    print(f'{what}: {n}/{s1.size} spike counts differ')
    assert n <= MAX_DIFF_FRAC * s1.size
    assert np.abs(s1.to_numpy() - s2.to_numpy()).max() <= 1


def test_cell_weights(base, kc_ids):
    """Per cell (or pooled), normalized so each cell's input sums to 1 and the product is
    MB-C1's KC x KC matrix; raw counts with normalize=False."""
    wiring = base[2]
    for pooled in (False, True):
        for claws_only in (False, True):
            src = dict(wPNKC=wiring) if claws_only else dict(kc_ids=kc_ids)
            w_out, w_in = connectome_MBC1_cell_weights(**src, pooled=pooled,
                claws_only=claws_only
            )
            m = connectome_MBC1_KCKC_weights(**src, pooled=pooled, claws_only=claws_only)
            cols = [MBC1_POOLED] if pooled else CELLS
            for w in (w_out, w_in):
                assert list(w.columns) == cols and w.columns.name == MBC1_ID
                assert w.index.name == KC_ID
                assert np.array_equal(w.index.to_numpy(), kc_ids.to_numpy())
                assert (w.to_numpy() >= 0).all()
            assert np.allclose(w_in.sum().to_numpy(), 1.0)
            prod = w_out.to_numpy() @ w_in.to_numpy().T
            assert np.allclose(prod, m.to_numpy(), rtol=1e-12, atol=0)
            assert np.isclose(prod.sum(axis=1).mean(), 1.0)

    raw_out, raw_in = connectome_MBC1_cell_weights(kc_ids, normalize=False)
    by_out, by_in, _, _ = connectome_MBC1_weights(by_cell=True)
    assert np.array_equal(raw_out.to_numpy(), by_out.reindex(kc_ids).fillna(0).to_numpy())
    assert np.array_equal(raw_in.to_numpy(), by_in.reindex(kc_ids).fillna(0).to_numpy())


@pytest.mark.parametrize('variant', [dict(), dict(mbc1_pooled=True),
    dict(mbc1_claws_only=True)], ids=['cells', 'pooled', 'claws_only']
)
def test_cells_equal_kckc_matrix(orn, base, kc_ids, cells, variant):
    """(a, b) MB-C1's cells, with tau = kc_kc_tau and no threshold, give the KC-KC path's
    outputs with MB-C1's KC x KC matrix as `_wKCKC` at the same scale: per cell (rank 2),
    pooled (rank 1) and on the wiring's claws only."""
    if len(variant) == 0:
        r1, s1, w1, p1 = cells
    else:
        r1, s1, w1, p1 = fit_mb_model(orn, mbc1_inh=SCALE, **variant, **KWS, **FIXED)
    assert p1['mbc1_inh'] == SCALE
    assert {k for k in p1 if k in MBC1_PARAMS} == {'mbc1_inh'} | set(variant)
    assert all(p1[k] is True for k in variant)
    assert 'kc_kc_inh' not in p1

    if variant.get('mbc1_claws_only'):
        m = connectome_MBC1_KCKC_weights(wPNKC=base[2], claws_only=True)
    else:
        m = connectome_MBC1_KCKC_weights(kc_ids,
            pooled=variant.get('mbc1_pooled', False)
        )
    r2, s2, w2, _ = fit_mb_model(orn, kc_kc_inh=SCALE, _wKCKC=m, **KWS, **FIXED)
    assert w1.equals(w2) and r1.index.equals(r2.index)
    assert_nearly_equal(s1, s2, f'MB-C1 {variant} vs its KC x KC matrix')

    # MB-C1 inhibits
    s0 = base[1]
    assert s1.to_numpy().sum() < s0.to_numpy().sum()
    assert (s1.to_numpy() <= s0.to_numpy()).mean() > 0.99


def test_cells_with_kckc_equal_summed_matrix(orn, kc_ids, cells):
    """(c) MB-C1's cells and KC-KC inhibition together (equal taus) give the KC-KC path
    with the summed matrix, s1 * (hemibrain KC->KC) + s2 * (MB-C1's matrix)."""
    s_kckc = 20.0
    r1, s1, _, p1 = fit_mb_model(orn, kc_kc_inh=s_kckc, mbc1_inh=SCALE, **KWS, **FIXED)
    assert p1['kc_kc_inh'] == s_kckc and p1['mbc1_inh'] == SCALE

    hb = connectome_KCKC_weights(kc_ids)
    mb = connectome_MBC1_KCKC_weights(kc_ids, pooled=False)
    assert hb.index.equals(mb.index) and hb.columns.equals(mb.columns)
    summed = s_kckc * hb + SCALE * mb
    r2, s2, _, _ = fit_mb_model(orn, kc_kc_inh=1.0, _wKCKC=summed, **KWS, **FIXED)
    assert_nearly_equal(s1, s2, 'MB-C1 + KC-KC vs the summed matrix')

    # both act: fewer spikes than MB-C1 alone
    assert s1.to_numpy().sum() < cells[1].to_numpy().sum()


def test_zero_scale_and_high_threshold_equal_off(orn, base, cells):
    """(d) mbc1_inh=0 gives the outputs without MB-C1; (e) so does a threshold above any
    cell's activity, and a threshold in between gives fewer spikes than none and more
    than the linear cells."""
    r0, s0, w0, p0 = base
    assert not any(k in p0 for k in MBC1_PARAMS)

    r1, s1, w1, p1 = fit_mb_model(orn, mbc1_inh=0.0, **KWS, **FIXED)
    assert p1['mbc1_inh'] == 0.0
    assert r1.equals(r0) and s1.equals(s0) and w1.equals(w0)

    r2, s2, _, p2 = fit_mb_model(orn, mbc1_inh=SCALE, mbc1_thr=1e9, **KWS, **FIXED)
    assert p2['mbc1_thr'] == 1e9
    assert r2.equals(r0) and s2.equals(s0)

    # (a cell's activity is the synapse-weighted mean of its input KCs' filtered spike
    # output, in Hz: about 1 Hz here)
    _, s3, _, p3 = fit_mb_model(orn, mbc1_inh=SCALE, mbc1_thr=1.0, **KWS, **FIXED)
    assert p3['mbc1_thr'] == 1.0
    n_cells, n3, n0 = (s.to_numpy().sum() for s in (cells[1], s3, s0))
    assert n_cells < n3 < n0


def test_hardcoded_weights(orn, kc_ids, cells):
    """`_wMBC1KC` / `_wKCMBC1`, in any KC order, are used as passed (here the loader's
    own, so the outputs are those of the default)."""
    w_out, w_in = connectome_MBC1_cell_weights(kc_ids)
    r, s, _, p = fit_mb_model(orn, mbc1_inh=SCALE, _wMBC1KC=w_out.iloc[::-1],
        _wKCMBC1=w_in.iloc[::-1], **KWS, **FIXED
    )
    assert r.equals(cells[0]) and s.equals(cells[1])
    assert p['mbc1_inh'] == SCALE

    ret = fit_mb_model(orn, mbc1_inh=SCALE, return_olfsysm_vars=True, **KWS, **FIXED)
    params = ret[3]
    rv = params['rv']
    assert np.array_equal(params['wMBC1KC_unscaled'], w_out.to_numpy())
    assert np.array_equal(rv.kc.wKCMBC1, w_in.to_numpy().T)
    assert np.array_equal(rv.kc.wMBC1KC, SCALE * w_out.to_numpy())
    assert params['mp'].kc.use_mbc1 and not params['mp'].kc.use_kc_kc_inh


def test_simulator(orn, base, cells):
    """(f) the fixed-parameter simulator's 'wMBC1_scale' gives `fit_mb_model`'s outputs
    with `mbc1_inh` at that value (and 0 those without MB-C1); (g) with APL's output
    weights at 0, MB-C1 still inhibits."""
    initial = dict(fixed_thr=FIXED['fixed_thr'], wAPLKC_scale=FIXED['wAPLKC'],
        wMBC1_scale=SCALE
    )
    # (the simulator checks it reproduces its initial fit_mb_model call)
    simulate, params0 = make_fixed_param_kc_simulator(orn, initial_params=initial, **KWS)
    assert params0['wMBC1_scale'] == SCALE
    assert 'wKCKC_scale' not in params0

    r, s = simulate(dict())
    assert r.equals(cells[0]) and s.equals(cells[1])

    _, s0 = simulate({'wMBC1_scale': 0.0})
    assert s0.equals(base[1])

    params = {**params0, 'wMBC1_scale': 2 * SCALE}
    r2, s2 = simulate(params)
    r3, s3, _, p3 = fit_mb_model(orn, **KWS, **fixed_params_to_fit_mb_model_kws(params))
    assert p3['mbc1_inh'] == 2 * SCALE
    assert r2.equals(r3) and s2.equals(s3)
    assert s2.to_numpy().sum() < s.to_numpy().sum()

    # APL off (APL's output weights at 0): MB-C1 still acts
    off = {k: v for k, v in APL_OFF.items() if k in params0}
    assert set(off) == {'wAPLKC_scale', 'wAPLPN_scale'}
    _, s_off0 = simulate({**off, 'wMBC1_scale': 0.0})
    r_off, s_off = simulate({**off})
    assert s_off.to_numpy().sum() < s_off0.to_numpy().sum()
    assert s_off0.to_numpy().sum() > s0.to_numpy().sum()
    r_off2, s_off2, _, _ = fit_mb_model(orn, **KWS,
        **fixed_params_to_fit_mb_model_kws({**params0, **off})
    )
    assert r_off.equals(r_off2) and s_off.equals(s_off2)

    # the scale can also be enabled with the kwarg (as kc_kc_inh can)
    initial2 = {k: v for k, v in initial.items() if k != 'wMBC1_scale'}
    simulate2, params2 = make_fixed_param_kc_simulator(orn, initial_params=initial2,
        mbc1_inh=SCALE, **KWS
    )
    assert params2['wMBC1_scale'] == SCALE
    with pytest.raises(ValueError):
        simulate2({'wMBC1_scale': -1.0})


def test_model_id():
    """Model IDs / LR cache keys only change when MB-C1 kwargs are passed (the flags
    only when True), and small scales are told apart."""
    kws = dict(weight_divisor=20)
    none_kws = {**kws, 'mbc1_inh': None, 'mbc1_tau': None, 'mbc1_thr': None,
        'mbc1_pooled': False, 'mbc1_claws_only': False
    }
    assert format_model_params(none_kws) == format_model_params(kws)
    assert onestep_lr_cache_key(none_kws) == onestep_lr_cache_key(kws)

    variants = (dict(), dict(mbc1_inh=0.0), dict(mbc1_inh=0.004), dict(mbc1_inh=0.005),
        dict(mbc1_inh=0.005, mbc1_tau=0.1), dict(mbc1_inh=0.005, mbc1_thr=1.0),
        dict(mbc1_inh=0.005, mbc1_pooled=True), dict(mbc1_inh=0.005, kc_kc_inh=0.005)
    )
    assert len({format_model_params({**kws, **x}) for x in variants}) == len(variants)
    assert onestep_lr_cache_key({**kws, 'mbc1_inh': 0.0}) != onestep_lr_cache_key(kws)


def test_refusals(orn, kc_ids):
    """(h) the male CNS refuses the loader's weights; invalid kwargs raise ValueError
    before any simulation."""
    with pytest.raises(NotImplementedError):
        connectome_MBC1_cell_weights(kc_ids, connectome='malecns-right')
    for side in ('malecns-left', 'malecns-right'):
        with pytest.raises(NotImplementedError):
            fit_mb_model(orn, pn2kc_connections=side, mbc1_inh=1.0, **KWS, **FIXED)

    w_out, w_in = connectome_MBC1_cell_weights(kc_ids)
    for bad in (dict(mbc1_tau=0.1), dict(mbc1_thr=1.0), dict(mbc1_pooled=True),
            dict(mbc1_claws_only=True), dict(mbc1_inh=-1.0), dict(mbc1_inh=np.nan),
            dict(mbc1_inh=1.0, mbc1_tau=0.0), dict(mbc1_inh=1.0, mbc1_thr=-1.0),
            dict(_wMBC1KC=w_out, _wKCMBC1=w_in),
            dict(mbc1_inh=1.0, _wMBC1KC=w_out),
            dict(mbc1_inh=1.0, _wMBC1KC=w_out, _wKCMBC1=w_in, mbc1_pooled=True)):
        with pytest.raises(ValueError):
            fit_mb_model(orn, **bad, **KWS, **FIXED)

    # claws only needs the claw wiring
    with pytest.raises(ValueError):
        fit_mb_model(orn, weight_divisor=20, mbc1_inh=1.0, mbc1_claws_only=True,
            make_plots=False, silent=True
        )
    with pytest.raises(ValueError):
        fit_mb_model(orn, mbc1_inh=1.0, n_claws_active_to_spike=2, **KWS, **FIXED)
