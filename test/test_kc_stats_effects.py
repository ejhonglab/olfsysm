"""Tests of pyolfsysm.kc_stats and pyolfsysm.effects on synthetic KC populations (moved with
the functions from the KC-remapping experiment queue's test_apl_effect_by_class.py,
test_apl_effect_decomposition.py and test_readout_low_end.py, 2026-11-12): the
odors-per-KC classes and the responder terms add up to the cosine and to the per-pair
effect, a naive per-unit sum is reproduced, and the effect's error splits into parts whose
squares add."""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from hong2p.rdm import PAIR_NAMES, pair_design, pair_distances
from pyolfsysm import effects as E
from pyolfsysm import kc_stats as K

ODORS = ['2h', 'IaA', 'pa', '2-but', 'eb', 'ep', '6al', 't2h', '1-8ol', '1-5ol', '1-6ol',
    'ms', 'paa', '-aPine', 'euc', 'B-myr'
]
CLASSES = list(K.OPK_CLASSES)


def _graded(seed=0, n_units=600, p_resp=0.12):
    """z-score-like values: noise around 0 (negatives included) plus responders well above
    1."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.4, (n_units, len(ODORS)))
    resp = rng.random(x.shape) < p_resp * rng.gamma(2.0, 0.5, (n_units, 1))
    return pd.DataFrame(np.where(resp, 1.2 + rng.gamma(2.0, 1.0, x.shape), x), columns=ODORS)


def _counts(seed=0, n_units=800, p_resp=0.07):
    rng = np.random.default_rng(seed)
    u = rng.random((n_units, len(ODORS))) * rng.gamma(3.0, 1 / 3.0, (n_units, 1))
    n = np.where(u < p_resp, 2 + rng.geometric(0.12, u.shape),
        np.where(u < p_resp + 0.05, 1, 0)
    )
    return pd.DataFrame(n.astype(float), columns=ODORS)


def test_amplitude_stats_and_responder_cv():
    x = pd.DataFrame(np.zeros((100, 3)), columns=['a', 'b', 'c'])
    x.iloc[:, 0] = 1.0
    x.iloc[0, 1] = 5.0
    st = K.amplitude_stats(x)
    assert st.loc['a', 'participation_ratio'] == pytest.approx(1.0)
    assert st.loc['b', 'participation_ratio'] == pytest.approx(0.01)
    assert st.loc['b', 'top1_share'] == 1.0 and np.isnan(st.loc['c', 'top1_share'])
    assert K.responder_cv(np.array([2.0, 2.0, np.nan])) == 0.0
    assert np.isnan(K.responder_cv(np.array([])))


def test_poisson_binomial_and_class_shares():
    P = np.array([[1.0] * 4 + [0.0] * 12, [0.5] * 16])
    bins = K.poisson_binomial_bins(P)
    assert bins.sum() == pytest.approx(1.0)
    # unit 0 responds to exactly 4 odors (class 4_7); unit 1 to Binomial(16, 0.5) odors
    from math import comb
    b16 = np.array([comb(16, k) for k in range(17)]) / 2 ** 16
    want = 0.5 * np.array([b16[0], b16[1], b16[2:4].sum(), b16[4:8].sum(), b16[8:].sum()])
    want[3] += 0.5
    assert np.allclose(bins, want)
    R = np.zeros((4, 16), dtype=bool)
    R[1, 0] = True
    R[2, :3] = True
    R[3, :] = True
    assert K.opk_class_shares(R).tolist() == [0.25, 0.25, 0.25, 0.0, 0.25]
    assert K.opk_class_labels() == CLASSES


def test_classes_and_terms_add_up_to_the_cosine():
    X = _graded()
    t = K.class_cosine_terms(X, order=ODORS)
    assert list(t.columns.get_level_values('class').unique()) == CLASSES + ['all']
    tot = t.xs('total', axis=1, level='term')
    cos = 1 - pair_distances(X, 'cosine', order=ODORS)
    assert np.allclose(tot[CLASSES].sum(axis=1), tot['all'], rtol=0, atol=1e-12)
    assert np.allclose(tot['all'], cos.reindex(t.index), rtol=0, atol=1e-12)
    for c in CLASSES + ['all']:
        assert np.allclose(t[(c, 'both')] + t[(c, 'one')] + t[(c, 'neither')], t[(c, 'total')],
            rtol=0, atol=1e-12)
    ct = K.cosine_terms(X, order=ODORS)
    for term in ('both', 'one', 'neither'):
        assert np.allclose(t[('all', term)], ct[term], rtol=0, atol=1e-12)
    # an array with its odors is the same
    ta = K.class_cosine_terms(X.to_numpy(), odors=ODORS, order=ODORS)
    assert np.array_equal(ta.to_numpy(), t.to_numpy(), equal_nan=True)
    with pytest.raises(ValueError):
        K.class_cosine_terms(X.to_numpy())


def test_naive_per_unit_sum():
    X = _graded(seed=3, n_units=60)
    t = K.class_cosine_terms(X, thresh=1.0, order=ODORS)
    x = X.to_numpy()
    R = x > 1.0
    n_kc = R.sum(axis=1)
    norms = np.sqrt((x ** 2).sum(axis=0))
    i, j = ODORS.index('IaA'), ODORS.index('ms')
    for c, (lo, hi) in zip(CLASSES, K.OPK_EDGES):
        both = one = neither = 0.0
        for u in range(len(x)):
            if not (lo <= n_kc[u] <= (16 if hi is None else hi)):
                continue
            p = x[u, i] * x[u, j]
            if R[u, i] and R[u, j]:
                both += p
            elif R[u, i] or R[u, j]:
                one += p
            else:
                neither += p
        den = norms[i] * norms[j]
        for term, v in (('both', both), ('one', one), ('neither', neither)):
            assert np.isclose(t.loc[('IaA', 'ms'), (c, term)], v / den, rtol=1e-12, atol=1e-15)


def test_cosine_terms_add_up():
    rng = np.random.default_rng(9)
    cols = ODORS[:6]
    m = pd.DataFrame(rng.normal(0, 0.5, (300, 6)) + np.where(rng.random((300, 6)) < 0.1, 2.0,
        0.0), columns=cols)
    t = K.cosine_terms(m)
    assert np.allclose(t[['both', 'one', 'neither']].sum(axis=1), t['cosine'])
    assert np.allclose(1 - t['cosine'].to_numpy(), pair_distances(m, 'cosine').to_numpy())
    pos = m.clip(lower=0)
    tz = K.cosine_terms(pos.where(pos > 1, 0.0))
    assert np.allclose(tz['one'], 0) and np.allclose(tz['neither'], 0)


def test_effect_classes_add_up_and_match_pair_effect():
    on = [_graded(s) for s in (0, 1, 2)]
    off = [_graded(s, p_resp=0.25) for s in (3, 4)]
    e = E.effect_by_class(on, off, order=ODORS)
    eff = e['effect'].xs('total', axis=1, level='term')
    assert np.allclose(eff[CLASSES].sum(axis=1), eff['all'], rtol=0, atol=1e-12)
    d_on = pd.concat([pair_distances(m, order=ODORS) for m in on], axis=1).mean(axis=1)
    d_off = pd.concat([pair_distances(m, order=ODORS) for m in off], axis=1).mean(axis=1)
    ref = E.pair_effect(d_on, d_off, ODORS)
    assert np.allclose(eff['all'], ref.reindex(eff.index), rtol=0, atol=1e-12)
    assert np.allclose(e['effect'], e['off'] - e['on'])
    for c in ('on', 'off'):
        assert np.isclose(e.attrs['class_shares'][c].sum(), 1.0)


def test_one_class_equals_effect_stats():
    on, off = _counts(1), _counts(2, p_resp=0.14)
    e = E.effect_by_class(on, off, resp_intact=on >= 2, resp_manipulated=off >= 2,
        edges=((0, None),), order=ODORS)
    assert list(e['effect'].columns.get_level_values('class').unique()) == ['0plus', 'all']
    d_on, d_off = pair_distances(on, order=ODORS), pair_distances(off, order=ODORS)
    eff = e['effect'][('0plus', 'total')]
    assert np.allclose(eff, E.pair_effect(d_on, d_off, ODORS).reindex(eff.index), rtol=0,
        atol=1e-12)
    data = 0.8 * E.pair_effect(d_on, d_off, ODORS) + 0.01 * np.sin(np.arange(120))
    st = E.effect_stats(d_on, d_off, data, ODORS, prefix='apl_effect_')
    err = eff - data.reindex(eff.index)
    assert np.isclose(st['apl_effect_rmse'], np.sqrt(np.mean(err ** 2)), rtol=1e-12)
    assert np.isclose(st['apl_effect_rmse_centered'], np.sqrt(np.mean((err - err.mean()) ** 2)),
        rtol=1e-12)
    assert st['apl_effect_n_pairs'] == 120


def test_responders_aligned_by_label():
    """Responder frames are aligned to the matrix by odor label, arrays by position."""
    on, off = _counts(1), _counts(2, p_resp=0.14)
    shuffled = list(reversed(ODORS))
    a = E.effect_by_class(on, off, resp_intact=on >= 2, resp_manipulated=off >= 2,
        order=ODORS)
    b = E.effect_by_class(on[shuffled], off, resp_intact=(on >= 2)[ODORS],
        resp_manipulated=(off >= 2).to_numpy(), order=ODORS)
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


SIX = ['a', 'b', 'c', 'd', 'e', 'f']


def _pairs():
    return pd.MultiIndex.from_tuples(list(combinations(SIX, 2)), names=PAIR_NAMES)


def _series(values, name='distance'):
    return pd.Series(np.asarray(values, dtype=float), index=_pairs(), name=name)


def test_decomposition_squares_add_and_parts():
    rng = np.random.default_rng(0)
    n = len(_pairs())
    data = _series(rng.normal(0.18, 0.05, n), name='apl_effect')
    off = _series(rng.uniform(0.4, 0.9, n))
    add = np.array([0.03, -0.02, 0.01, 0.0, -0.01, 0.02])
    per_odor_err = np.array([add[SIX.index(a)] + add[SIX.index(b)] for a, b in _pairs()])
    on = off + data.to_numpy() - 0.05 + per_odor_err
    stats, shares = E.effect_decomposition(on, off, data)
    total = (stats['effect_mean_error'] ** 2 + stats['effect_per_odor'] ** 2 +
        stats['effect_pair_specific'] ** 2)
    assert stats['effect_rmse'] ** 2 == pytest.approx(total)
    assert stats['effect_pair_specific'] == pytest.approx(0.0, abs=1e-12)
    assert stats['effect_per_odor'] > 0
    assert list(shares.columns) == ['model_share', 'data_share', 'share_error']
    X, _ = pair_design(_pairs(), SIX)
    noise = rng.normal(0, 0.02, n)
    noise -= X @ np.linalg.lstsq(X, noise, rcond=None)[0]
    stats2, _ = E.effect_decomposition(off + data.to_numpy() + noise, off, data)
    assert stats2['effect_per_odor'] == pytest.approx(0.0, abs=1e-12)
    assert stats2['effect_pair_specific'] == pytest.approx(float(np.sqrt(np.mean(noise ** 2))))
    parts = E.effect_pair_parts(off + data.to_numpy() + noise, off, data)
    assert np.allclose(parts['per_odor'] + parts['pair_specific'], parts['error'])
    assert np.allclose(parts['pair_specific'].to_numpy(), noise)
