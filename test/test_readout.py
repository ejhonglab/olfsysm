"""Tests of pyolfsysm.readout on synthetic spike counts and spike times (moved with the
functions from the KC-remapping experiment queue's test_session_zscore.py,
test_readout_low_end.py and test_e82_protocol.py, 2026-11-12): the readouts and their
calibration, the kernel weights of spikes against their limits and a brute-force integral,
the expected effect of measurement noise against sampling, and the per-KC session z-score
(its formula, its edge at 1 / q, its readings through noise and its calibration)."""
import numpy as np
import pandas as pd
import pytest

from hong2p.rdm import pair_distances
from pyolfsysm import readout as R
from pyolfsysm.effects import effect_stats

ODORS = ['2h', 'IaA', 'pa', '2-but', 'eb', 'ep', '6al', 't2h', '1-8ol', '1-5ol', '1-6ol',
    'ms', 'paa', '-aPine', 'euc', 'B-myr'
]
WINDOW = (2.0, 8.1)
KERNELS = [(r, d) for r in (0.2, 0.5, 1.0) for d in (1.0, 3.0, 5.0)]
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def _counts(seed=0, n_units=800, p_resp=0.07, odors=ODORS, geo=0.12):
    """Spike-count-like matrix: mostly 0, some 1, responders with geometric counts."""
    rng = np.random.default_rng(seed)
    u = rng.random((n_units, len(odors)))
    n = np.where(u < p_resp, 2 + rng.geometric(geo, u.shape),
        np.where(u < p_resp + 0.05, 1, 0)
    )
    return pd.DataFrame(n.astype(float), columns=odors)


# ---------------------------------------------------------------------------
# readouts
def test_floor_formula_and_default():
    c = _counts(n_units=400, odors=ODORS[:6], p_resp=0.08, geo=0.15)
    K = 7.0
    plain = R.readout(c, 'saturating', K)
    assert np.array_equal(plain.to_numpy(), (c / (c + K)).to_numpy())
    assert np.array_equal(R.readout(c, 'saturating', K, floor=0.0).to_numpy(), plain.to_numpy())
    x = c.to_numpy()
    for b in (-1.0, -0.5, 0.5, 3.0):
        r = R.readout(c, 'saturating', K, floor=b).to_numpy()
        assert np.allclose(r, np.where(x > 0, (x + b) / (x + b + K), 0.0))
        assert (r[x == 0] == 0).all()
    assert (R.readout(c, 'saturating', K, floor=-1.0).to_numpy()[x == 1] == 0).all()
    with pytest.raises(ValueError):
        R.readout(c, 'power', 0.5, floor=1.0)
    with pytest.raises(ValueError):
        R.readout(c, 'saturating', K, floor=-2.0)
    with pytest.raises(ValueError):
        R.readout(c, 'cube')


def test_readouts_keep_responders():
    c = _counts(1, n_units=300)
    lut = R.rank_match_map(c, np.linspace(1.0, 9.0, 50))
    assert lut.is_monotonic_increasing
    for kind, param in (('raw', None), ('sqrt', None), ('power', 0.4), ('saturating', 3.0),
            ('log1p', 2.0), ('binary', 1.0), ('rank_match', lut)):
        r = R.readout(c, kind, param).to_numpy()
        assert ((r > 0) == (c.to_numpy() > 0)).all(), kind
    assert R.readout_pair_distances(c, 'raw', order=ODORS).equals(pair_distances(c, order=ODORS))


def test_calibrate_readout():
    c = _counts(1, n_units=400, odors=ODORS[:6], p_resp=0.08, geo=0.15)
    k0 = R.calibrate_readout(c, 'saturating', 0.05, responders_min=2.0)
    assert k0 == R.calibrate_readout(c, 'saturating', 0.05, responders_min=2.0, floor=0.0)
    assert R.readout_stat(c, 'saturating', k0, 'participation_ratio', 2.0) == \
        R.readout_stat(c, 'saturating', k0, 'participation_ratio', 2.0, floor=0.0)
    target = R.readout_stat(c, 'saturating', 4.0, responders_min=2.0)
    assert R.calibrate_readout(c, 'saturating', target, responders_min=2.0) == \
        pytest.approx(4.0, rel=1e-6)
    # not bracketed: the nearer bound (power's upper bound is 1, the raw counts, whose
    # participation ratio is the lowest)
    assert R.calibrate_readout(c, 'power', 0.0) == 1.0
    assert R.calibrate_readout(c, 'power', 10.0) == pytest.approx(1e-3)


def test_expected_noisy_distances():
    c = R.readout(_counts(2, n_units=2000, odors=ODORS[:6], p_resp=0.08, geo=0.15),
        'saturating', 5.0) * 4.0
    d0 = R.expected_pair_distances_noisy(c, 0.0)
    assert np.allclose(d0.to_numpy(), pair_distances(c, 'cosine').to_numpy())
    s = 0.4
    exp = R.expected_pair_distances_noisy(c, s, n_trials=3)
    means = R.sample_pseudo_recordings(c, s, n_recordings=200, n_trials=3, seed=1)
    sampled = np.mean([pair_distances(pd.DataFrame(m, columns=c.columns), 'cosine').to_numpy()
        for m in means], axis=0)
    assert np.allclose(exp.to_numpy(), sampled, atol=3e-3)
    assert (exp.to_numpy() > d0.to_numpy()).all()


def test_sample_pseudo_recordings_shapes_and_seed():
    c = _counts(3, n_units=400, odors=ODORS[:6])
    m = R.sample_pseudo_recordings(c, 0.3, n_recordings=4, n_trials=3, seed=5)
    t = R.sample_pseudo_recordings(c, 0.3, n_recordings=4, n_trials=3, seed=5,
        return_trials=True)
    assert m.shape == (4,) + c.shape and t.shape == (4, 3) + c.shape
    assert np.allclose(t.mean(axis=1), m)
    trials = np.repeat(c.to_numpy()[None, None], 3, axis=1).repeat(2, axis=0)
    assert np.array_equal(R.sample_pseudo_recordings(trials, 0.0, return_trials=True), trials)
    with pytest.raises(ValueError):
        R.sample_pseudo_recordings(trials[0], 0.1)


def test_model_rdms_and_response_fractions():
    c = _counts(4, n_units=300)
    resp = (c >= 2).astype(float)
    rd = R.model_rdms(resp, c, order=ODORS)
    assert set(rd) == set(R.MODEL_RDM_VARIANTS) and R.PRIMARY_MODEL_RDM in rd
    assert rd['spike_counts_cosine'].equals(pair_distances(c, order=ODORS))
    assert np.allclose(R.response_fractions(resp).to_numpy(), resp.mean().to_numpy())


# ---------------------------------------------------------------------------
# kernel weights of spikes
def _brute(s, rise, decay, window=WINDOW, n=400001):
    """The kernel's area inside the window for a spike at s, by the trapezoid rule."""
    t = np.linspace(window[0], window[1], n)
    u = t - s
    k = np.where(u > 0, (1 - np.exp(-np.maximum(u, 0) / rise)) if rise > 0 else 1.0, 0.0)
    if np.isfinite(decay):
        k = k * np.exp(-np.maximum(u, 0) / decay)
    return _trapz(k, t)


def _spikes(n_kc=40, n_odor=5, n=600, t_max=1.9, seed=0):
    rng = np.random.default_rng(seed)
    st = pd.DataFrame(dict(kc=rng.integers(0, n_kc, n).astype(np.int32),
        odor=rng.integers(0, n_odor, n).astype(np.int32),
        time_s=np.round(rng.uniform(0.0, t_max, n), 4)))
    idx = pd.MultiIndex.from_arrays([np.arange(n_kc) + 1000, ['ab'] * n_kc],
        names=['kc_id', 'kc_type'])
    cols = pd.Index([f'o{j} @ -3' for j in range(n_odor)], name='odor')
    counts = np.zeros((n_kc, n_odor))
    np.add.at(counts, (st.kc.to_numpy(), st.odor.to_numpy()), 1.0)
    return st, idx, cols, pd.DataFrame(counts, index=idx, columns=cols)


def test_planner_weights():
    """A spike 50 ms after onset weighs 0.86, 0.62 and 0.17 of one at 2 s at decays 5, 3
    and 1 s with rises 1, 0.5 and 0.2 s."""
    got = [float(R.kernel_window_weights([0.05], r, d)[0])
        for r, d in ((1.0, 5.0), (0.5, 3.0), (0.2, 1.0))]
    assert np.round(got, 2).tolist() == [0.86, 0.62, 0.17]


@pytest.mark.parametrize('rise,decay', [(0.2, 1.0), (0.5, 3.0), (1.0, 5.0), (0.0, 3.0),
    (0.5, np.inf), (0.0, np.inf), (1e-4, 1e4)])
def test_weights_match_brute_force(rise, decay):
    times = np.array([-0.3, 0.0, 0.05, 0.7, 1.99, 2.0, 2.4, 5.0, 8.05, 8.1, 9.0])
    w = R.kernel_window_weights(times, rise, decay)
    want = np.array([_brute(s, rise, decay) for s in times]) / _brute(2.0, rise, decay)
    assert np.allclose(w, want, rtol=2e-5, atol=1e-6)
    assert w[times >= WINDOW[1]].max() == 0.0


def test_weights_tend_to_one():
    times = np.linspace(-0.5, 1.999, 500)
    assert np.array_equal(R.kernel_window_weights(times, 0.0, np.inf), np.ones(len(times)))
    for eps in (1e-3, 1e-5, 1e-7):
        w = R.kernel_window_weights(times, eps, 1.0 / eps)
        assert np.abs(w - 1).max() < 20 * eps
    assert np.allclose(R.kernel_window_weights([2.0, 5.05, 8.1], 0.0, np.inf), [1.0, 0.5, 0.0])


def test_weights_monotone_and_reference():
    times = np.append(np.linspace(-0.5, 1.0, 200), 2.0)
    for r, d in KERNELS:
        w = R.kernel_window_weights(times, r, d)
        assert np.all(np.diff(w[:-1]) > 0) and w[-1] == 1.0
        assert R.kernel_window_weights(np.linspace(-0.5, 2.0, 2001), r, d).max() < 1.02
        w2 = R.kernel_window_weights(times, r, d, ref_time=0.5)
        assert np.allclose(w2 / w, w2[0] / w[0])
        assert R.kernel_window_weights([0.5], r, d, ref_time=0.5)[0] == \
            pytest.approx(1.0, abs=1e-15)
    with pytest.raises(ValueError):
        R.kernel_window_weights([0.0], 0.5, 3.0, ref_time=9.0)
    with pytest.raises(ValueError):
        R.kernel_window_weights([0.0], -1.0, 3.0)


def test_weights_are_the_kernel_integral():
    """A spike's weight is the integral of hong2p's indicator kernel over the window, over
    that of a spike at the window's start."""
    from hong2p.imaging import indicator_kernel_integral as G

    s = np.array([0.05, 1.0, 3.0])
    for rise, decay in ((0.2, 1.0), (0.5, 3.0), (1.0, 5.0), (0.0, 2.0), (0.3, np.inf)):
        want = ((G(WINDOW[1] - s, rise, decay) - G(WINDOW[0] - s, rise, decay)) /
            G(WINDOW[1] - WINDOW[0], rise, decay))
        assert np.allclose(R.kernel_window_weights(s, rise, decay), want, rtol=1e-10)


def test_weighted_counts_reproduce_counts_for_a_step():
    st, idx, cols, counts = _spikes()
    x = R.kernel_weighted_counts(st, idx, cols, 0.0, np.inf)
    assert x.index.equals(idx) and x.columns.equals(cols)
    assert np.array_equal(x.to_numpy(), counts.to_numpy())
    y = R.kernel_weighted_counts(st, idx, cols, 1e-6, 1e6)
    assert np.allclose(y.to_numpy(), counts.to_numpy(), rtol=1e-4)
    assert not np.array_equal(y.to_numpy(), counts.to_numpy())


@pytest.mark.parametrize('rise,decay', KERNELS)
def test_train_at_one_time(rise, decay):
    idx = pd.Index(range(3), name='kc_id')
    cols = pd.Index(['a', 'b'], name='odor')
    st = pd.DataFrame(dict(kc=[1] * 7 + [2] * 3, odor=[0] * 7 + [1] * 3,
        time_s=[0.3] * 7 + [1.2] * 3))
    x = R.kernel_weighted_counts(st, idx, cols, rise, decay)
    w = R.kernel_window_weights([0.3, 1.2], rise, decay)
    assert x.iloc[1, 0] == pytest.approx(7 * w[0], rel=1e-14)
    assert x.iloc[2, 1] == pytest.approx(3 * w[1], rel=1e-14)
    assert (x.iloc[0] == 0).all()


def test_weighted_counts_onset_and_bounds():
    st, idx, cols, _ = _spikes()
    a = R.kernel_weighted_counts(st, idx, cols, 0.5, 3.0)
    b = R.kernel_weighted_counts(st.assign(time_s=st.time_s + 0.25), idx, cols, 0.5, 3.0,
        stim_onset=0.25)
    assert np.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12)
    with pytest.raises(ValueError):
        R.kernel_weighted_counts(st.assign(kc=st.kc + 1000), idx, cols, 0.5, 3.0)


# ---------------------------------------------------------------------------
# the per-KC session z-score
def test_session_zscore_formula():
    c = _counts()
    q, noise = 0.2, 1.5
    z = R.session_zscore(c, q=q, noise=noise, noise_ratio=0.4)
    x = c.to_numpy()
    n_stim = len(ODORS) + 1
    mu = q * x.sum(axis=1) / n_stim
    S = np.sqrt(q * (x ** 2).sum(axis=1) / n_stim - mu ** 2 + noise ** 2)
    assert np.allclose(z['center'][0], x / S[:, None])
    assert np.allclose(z['trial_sd'][0], 0.4 * noise / S)
    assert np.allclose(z['blank'][0], 0.0)
    zc = R.session_zscore(c, q=q, noise=noise, noise_ratio=0.4, center=True)
    assert np.allclose(zc['center'][0], (x - mu[:, None]) / S[:, None])
    assert np.allclose(zc['blank'][0], -mu / S)
    assert np.allclose(z['mean_sq'][0], (z['center'][0] ** 2).sum(axis=1) / n_stim +
        z['trial_sd'][0] ** 2)
    z0 = R.session_zscore(c, q=q, noise=1e-9, noise_ratio=0.0, center=True)
    assert z0['mean_sq'][0].max() <= 1 / q + 1e-6


def test_noise_free_edge_at_inv_q():
    q = 0.17
    x = np.zeros((3, len(ODORS)))
    x[0, 0] = 10.0
    x[1, :8] = 5.0
    x[2, :] = np.arange(1, 17)
    c = pd.DataFrame(x, columns=ODORS)
    zc = R.session_zscore(c, q=q, noise=0.0, center=True)
    n_stim = len(ODORS) + 1
    mean_r = (zc['center'][0].sum(axis=1) + zc['blank'][0]) / n_stim
    assert np.allclose(q * zc['mean_sq'][0] + (1 - q) * (q * mean_r / (1 - q)) ** 2, 1.0)
    z = R.session_zscore(c, q=q, noise=0.0)
    p = np.array([1, 8, 16]) / n_stim
    assert np.isclose(z['mean_sq'][0][0], 1 / (q * (1 - q * p[0])))
    assert np.isclose(z['mean_sq'][0][1], 1 / (q * (1 - q * p[1])))


def test_fixed_divisor_reproduces_present_readings():
    c, c2 = _counts(1), _counts(2)
    for amp, kind, K in (('linear', 'raw', None), ('saturating', 'saturating', 6.0)):
        present = R.readout_pair_distances(c, kind, K, order=ODORS)
        fixed = R.zcell_pair_distances(R.session_zscore(c, q=0.2, noise=0.0, amp=amp, K=K,
            divisor=2.5), order=ODORS)
        assert np.allclose(present.to_numpy(), fixed.reindex(present.index).to_numpy(),
            atol=1e-12)
        data = present * 0 + 0.1
        l_p = effect_stats(present, R.readout_pair_distances(c2, kind, K, order=ODORS), data,
            ODORS)['effect_rmse']
        l_f = effect_stats(fixed, R.zcell_pair_distances(R.session_zscore(c2, q=0.2,
            noise=0.0, amp=amp, K=K, divisor=2.5), order=ODORS), data, ODORS)['effect_rmse']
        assert np.isclose(l_p, l_f)
        sig = R.readout(c, kind, K)
        cz, s = 0.7, 0.385
        noisy = R.expected_pair_distances_noisy(cz * sig, s, 3, order=ODORS)
        fz = R.zcell_pair_distances(R.session_zscore(c, q=1.0, noise=s / cz, amp=amp, K=K,
            noise_ratio=1.0, divisor=1 / cz), order=ODORS)
        assert np.allclose(noisy.to_numpy(), fz.reindex(noisy.index).to_numpy(), atol=1e-12)


def test_readings_through_noise_match_sampling():
    c = _counts(3, n_units=3000)
    z = R.session_zscore(c, q=0.2, noise=1.0, noise_ratio=0.5)
    rng = np.random.default_rng(0)
    sd = (z['trial_sd'][0] / np.sqrt(3))[:, None]
    acc = {r: [] for r in R.ZCELL_READINGS}
    rates = []
    for _ in range(60):
        m = z['center'][0] + rng.normal(0, 1, z['center'][0].shape) * sd
        rates.append((m > 1).mean())
        for r, x in (('graded', m), ('thresholded', np.where(m > 1, m, 0.0)),
                ('binary', (m > 1).astype(float))):
            acc[r].append(pair_distances(pd.DataFrame(x, columns=ODORS), order=ODORS
                ).to_numpy())
    st = R.zcell_stats(z)
    assert abs(st['rate'] - np.mean(rates)) < 2e-3
    assert list(st['rate_per_odor'].index) == ODORS
    for r in R.ZCELL_READINGS:
        closed = R.zcell_pair_distances(z, r, order=ODORS).to_numpy()
        assert np.nanmax(np.abs(closed - np.nanmean(acc[r], axis=0))) < 0.02, r


def test_unlabeled_readings():
    """Arrays are labelled by `odors`, else by position."""
    c = _counts(5, n_units=300)
    zl = R.session_zscore(c, q=0.2, noise=1.0)
    za = R.session_zscore(c.to_numpy(), q=0.2, noise=1.0)
    assert za['odors'] is None
    d = R.zcell_pair_distances(za, odors=ODORS, order=ODORS)
    assert d.equals(R.zcell_pair_distances(zl, order=ODORS))
    assert list(R.zcell_stats(za)['rate_per_odor'].index) == list(range(16))
    with pytest.raises(ValueError):
        R.zcell_pair_distances(za, odors=ODORS[:3])


def test_calibration_reaches_targets():
    c = _counts(4, n_units=1500, p_resp=0.09)
    for amp in R.ZCELL_MAPS:
        cal = R.calibrate_session_zscore(c, q=0.18, amp=amp, target_rate=0.068,
            target_amp=2.0 if amp == 'saturating' else None, target_silent_sd=0.385)
        assert cal['converged'] and not cal['noise_at_bound']
        z = R.session_zscore(c, q=0.18, noise=cal['noise'], amp=amp, K=cal['K'],
            noise_ratio=cal['noise_ratio'])
        st = R.zcell_stats(z)
        assert abs(st['rate'] - 0.068) < 1e-6
        assert abs(st['nonresp_trial_sd'] - 0.385) < 1e-3
        if amp == 'saturating':
            assert not cal['K_at_bound']
            assert abs(st['resp_amp_mean'] - 2.0) < 1e-4


def test_trials_input():
    c = _counts(5)
    arr = np.broadcast_to(c.to_numpy(), (2, 3) + c.shape)
    z2 = R.session_zscore(c, q=0.2, noise=1.0)
    z4 = R.session_zscore(arr, q=0.2, noise=1.0)
    assert np.allclose(z4['center'][0], z2['center'][0])
    assert np.allclose(z4['center'][1], z2['center'][0])
    d2 = R.zcell_pair_distances(z2, order=ODORS)
    d4 = R.zcell_pair_distances(z4, odors=ODORS, order=ODORS)
    assert np.allclose(d2.to_numpy(), d4.to_numpy())


def test_drift_sets_units_only():
    c = _counts(6, n_units=1200, p_resp=0.09)
    kw = dict(q=0.128, amp='saturating', target_rate=0.068, target_amp=2.0,
        target_silent_sd=0.385)
    c0 = R.calibrate_session_zscore(c, **kw)
    c1 = R.calibrate_session_zscore(c, drift=1.0, **kw)
    z0 = R.session_zscore(c, q=0.128, noise=c0['noise'], amp='saturating', K=c0['K'],
        noise_ratio=c0['noise_ratio'])
    z1 = R.session_zscore(c, q=0.128, noise=c1['noise'], amp='saturating', K=c1['K'],
        noise_ratio=c1['noise_ratio'], drift=1.0)
    assert np.isclose(c0['K'], c1['K'], rtol=1e-5)
    assert np.allclose(z0['center'], z1['center'], atol=1e-6)
    assert np.allclose(z0['trial_sd'], z1['trial_sd'], atol=1e-6)
    silent = c.to_numpy().sum(axis=1) == 0
    assert np.allclose(z1['scale_ratio'][0][silent], np.sqrt(2.0))
