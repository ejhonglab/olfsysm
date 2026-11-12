"""The functions that moved here from the KC-remapping experiment queue (pyolfsysm.readout,
pyolfsysm.kc_stats, pyolfsysm.effects, pyolfsysm.tuning; 2026-11-12) give the outputs the
queue's own code gave before the move, on the same inputs.

test_data/frozen_from_queue/pyolfsysm.npz holds small inputs cut from real data (a fitted
model's spike counts and responses, its spike times, Yang's KC recordings, the data's
per-pair APL effect) and every case's output, frozen from the queue's code
(model_experiment_queue's test/frozen_step3/). The cases below call the moved functions
with the queue's parameters (the 16-odor APL panel's order, odor labels abbreviated as the
queue does: '2h @ -3' -> '2h', the response criterion of 1 z, APL blocking as the
manipulation); the tuning cases use a deterministic stand-in simulator
(`frozen_outputs.toy_simulator`), not the model. Exact in the environment the outputs were
frozen in, else to a relative 1e-9 (1e-6 for root-finding; `frozen_outputs.check`).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hong2p import rdm
from pyolfsysm import effects as E
from pyolfsysm import kc_stats as K
from pyolfsysm import readout as R
from pyolfsysm import tuning as T
from .conftest import test_data_dir
from .frozen_outputs import TOY_PARAMS, check, load, objects, toy_simulator

FROZEN = test_data_dir / 'frozen_from_queue' / 'pyolfsysm.npz'


def ab(df: pd.DataFrame) -> pd.DataFrame:
    """The queue's abbreviation of odor columns ('1-5ol @ -3' -> '1-5ol')."""
    out = df.copy()
    out.columns = pd.Index([str(c).split(' @ ')[0] for c in df.columns], name='odor')
    return out


def cases(O):
    order = O['odors']
    on, off = O['counts_on'], O['counts_off']

    def dist(df):
        return rdm.pair_distances(ab(df), order=order)

    def model():
        return dist(on), dist(off)

    def rpd(counts, *args, **kw):
        return R.readout_pair_distances(ab(counts), *args, order=order, **kw)

    def lut():
        return R.rank_match_map(on, O['ref_amp'])

    def z(**kw):
        kw = dict(dict(q=0.128, noise=2.0), **kw)
        return R.session_zscore(on.iloc[:200], **kw)

    def jitter():
        rng = np.random.default_rng(5)
        c = on.iloc[:100].to_numpy(dtype=float)
        return np.maximum(c[None, None] + rng.integers(-1, 2, (2, 3) + c.shape), 0.0)

    def zodors(zz):
        # the queue's labels: the abbreviated columns, or the APL16 odors for an array
        if zz['odors'] is None:
            return list(order)
        return [str(o).split(' @ ')[0] for o in zz['odors']]

    def zpd(zz, reading='graded', thresh=1.0):
        return R.zcell_pair_distances(zz, reading, thresh, odors=zodors(zz), order=order)

    def zst(zz, thresh=1.0):
        return R.zcell_stats(zz, thresh, odors=zodors(zz))

    def hold(**kw):
        p = dict(TOY_PARAMS) if 'params' not in kw else kw.pop('params')
        res = T.hold_rates(toy_simulator(), p, **dict(dict(target_off=420, target_on=160,
            size=200 * 16), **kw))
        # the keys the queue's wrapper (E77_input_screen.hold_rates) adds
        res.update(breadth=None, kws=None, breadth_sha256=None)
        return res

    def population():
        sims = {lab: toy_simulator(seed=s) for lab, s in (('real', 0), ('P1', 1), ('M1', 2))}
        ratios = {'real': (0.8, 1.2), 'P1': (0.7, 1.0), 'M1': (0.9, 1.5)}
        theta = dict(fixed_thr=25.0, wAPLKC_scale=0.6, wAPLPN_scale=1.2, wKCKC_scale=0.2)
        params = {lab: dict(theta, wKCAPL_scale=theta['wAPLKC_scale'] * ratios[lab][0],
            wPNAPL_scale=theta['wAPLKC_scale'] * ratios[lab][1]) for lab in sims}
        outs = T.simulate_population(sims, params)
        ds = [{c: dist(o[c][1].set_axis(order, axis=1)) for c in ('on', 'off')} for o in outs]
        return outs, T.mean_distances(ds)

    def bisect(target, bracket, step):
        sim = toy_simulator(seed=7)
        p = dict(TOY_PARAMS)
        return T.bisect_factor(lambda f: sim(dict(p, fixed_thr=p['fixed_thr'] * f)), target,
            200 * 16, bracket, step, 'frozen')

    def by_class():
        a = [O['trials_ctrl'].mean(axis=0), O['trials_ctrl2'].mean(axis=0)]
        b = [O['trials_tnt'].mean(axis=0)]
        return E.effect_by_class(a, b, odors=order, order=order)

    sat = lambda c: R.readout(c, 'saturating', 3.0)
    return {
        'readout': lambda: [R.readout(on, k, p) for k, p in (('raw', None), ('sqrt', None),
            ('power', 0.5), ('saturating', 3.0), ('log1p', 2.0), ('binary', None),
            ('binary', 3.0))] + [R.readout(on, 'saturating', 3.0, floor=0.5),
            R.readout(on, 'saturating', 4.0, floor=-1.0), R.readout(on, 'rank_match', lut())],
        'readout_pair_distances': lambda: [rpd(on, 'raw'), rpd(off, 'saturating', 3.0),
            rpd(on, 'saturating', 3.0, floor=0.5), rpd(on, 'binary', 2.0)],
        'readout_stat': lambda: [R.readout_stat(on, 'saturating', 3.0),
            R.readout_stat(on, 'saturating', 3.0, 'top1_share'),
            R.readout_stat(on, 'saturating', 3.0, responders_min=2.0),
            R.readout_stat(on, 'power', 0.5, 'responder_cv'),
            R.readout_stat(on, 'saturating', 3.0, floor=0.5, responders_min=2.0)],
        'calibrate_readout': lambda: [R.calibrate_readout(on, 'saturating', O['cal_target'],
            responders_min=2.0), R.calibrate_readout(on, 'power', 0.05),
            R.calibrate_readout(on, 'log1p', 0.04, stat='participation_ratio'),
            R.calibrate_readout(on, 'saturating', 0.5, stat='responder_cv'),
            R.calibrate_readout(on, 'saturating', O['cal_target'], responders_min=2.0,
                floor=0.3)],
        'rank_match_map': lut,
        'kernel_window_weights': lambda: [R.kernel_window_weights(np.linspace(-1, 10, 45), r,
            d, **kw) for r, d, kw in ((0.5, 3.0, {}), (0.0, np.inf, {}),
            (1.0, 5.0, dict(window=(2.0, 6.0))), (0.3, 20.0, dict(ref_time=0.5)),
            (0.0, 4.0, {}), (2.0, np.inf, {}))],
        'kernel_weighted_counts': lambda: [R.kernel_weighted_counts(O['spikes'],
            pd.Index(range(O['n_kc'])), pd.Index(order), 1.0, 5.0),
            R.kernel_weighted_counts(O['spikes'], pd.Index(range(O['n_kc'])), pd.Index(order),
                0.0, np.inf, stim_onset=0.5, window=(2.0, 8.1))],
        'expected_pair_distances_noisy': lambda: [R.expected_pair_distances_noisy(ab(on), 0.0,
            order=order), R.expected_pair_distances_noisy(ab(on), 0.7, 3, order=order),
            R.expected_pair_distances_noisy(ab(O['tm_mat']), 0.4, order=order)],
        'sample_pseudo_recordings': lambda: [R.sample_pseudo_recordings(on.iloc[:50], 0.4,
            n_recordings=3, seed=1), R.sample_pseudo_recordings(on.iloc[:40].to_numpy(), 0.3,
            n_recordings=2, n_trials=2, seed=2, return_trials=True),
            R.sample_pseudo_recordings(jitter()[:, :, :30], 0.5, seed=3)],
        'session_zscore': lambda: [z(), z(amp='saturating', K=5.0), z(center=True, n_blank=2),
            z(divisor=3.0), z(drift=0.5, noise_ratio=0.3),
            R.session_zscore(jitter(), q=0.128, noise=1.5)],
        'zcell_pair_distances': lambda: [zpd(z(), r) for r in ('graded', 'thresholded',
            'binary')] + [zpd(R.session_zscore(jitter(), q=0.128, noise=1.5), 'thresholded',
            thresh=0.8)],
        'zcell_stats': lambda: [zst(z()), zst(R.session_zscore(jitter(), q=0.128, noise=1.5)),
            zst(z(amp='saturating', K=5.0), thresh=0.8)],
        'calibrate_session_zscore': lambda: R.calibrate_session_zscore(on, q=0.128,
            amp='linear', target_rate=0.068),
        'calibrate_session_zscore_sat': lambda: R.calibrate_session_zscore(on.iloc[:150],
            q=0.128, amp='saturating', target_rate=0.068, target_amp=2.5,
            target_silent_sd=0.4, n_K_grid=9, max_iter=4),
        'response_fractions': lambda: R.response_fractions(ab(O['resp_on'])),
        'model_rdms': lambda: R.model_rdms(ab(O['resp_on']), ab(on), order),
        'amplitude_stats': lambda: [K.amplitude_stats(on), K.amplitude_stats(O['tm_mat'])],
        'responder_cv': lambda: [K.responder_cv(on.to_numpy()[on.to_numpy() >= 2]),
            K.responder_cv(np.array([np.nan, 1.0, 3.0])), K.responder_cv(np.array([]))],
        'poisson_binomial_bins': lambda: [K.poisson_binomial_bins(np.clip(
            O['tm_mat'].to_numpy() / 6, 0, 1)), K.poisson_binomial_bins(np.clip(
            O['tm_mat'].to_numpy() / 6, 0, 1), edges=((0, 1), (2, None)))],
        'opk_class': lambda: [K.opk_class_labels(), K.opk_class_labels(((0, 2), (3, None))),
            K.opk_class_shares(O['resp_on']), K.opk_class_shares(O['resp_off'].to_numpy() > 0)],
        'class_cosine_terms_array': lambda: K.class_cosine_terms_array(O['tm_mat'].to_numpy(),
            O['tm_mat'].to_numpy() > 1.0),
        'class_cosine_terms': lambda: [K.class_cosine_terms(ab(O['tm_mat']), order=order),
            K.class_cosine_terms(ab(sat(on)), resp=ab(on >= 2), order=order),
            K.class_cosine_terms(O['trials_ctrl'].mean(axis=0), thresh=0.8,
                edges=((0, 1), (2, None)), odors=order, order=order)],
        'cosine_terms': lambda: [K.cosine_terms(ab(O['tm_mat']), order=order),
            K.cosine_terms(ab(on), thresh=2.5, order=order)],
        'apl_effect': lambda: E.pair_effect(*model(), order, name='apl_effect'),
        'apl_effect_stats': lambda: E.effect_stats(*model(), O['data_effect'], order,
            prefix='apl_effect_'),
        'apl_effect_pair_parts': lambda: E.effect_pair_parts(*model(), O['data_effect'],
            order),
        'apl_effect_decomposition': lambda: E.effect_decomposition(*model(),
            O['data_effect'], order, prefix='apl_effect_'),
        'apl_effect_by_class': by_class,
        'apl_effect_by_class_resp': lambda: E.effect_by_class(ab(sat(on)), ab(sat(off)),
            resp_intact=ab(on >= 2), resp_manipulated=ab(off >= 2), order=order),
        'bisect_factor': lambda: [bisect(500, (0.1, 10.0), 1.1), bisect(2900, (0.5, 2.0),
            1.25)],
        'hold_rates': lambda: hold(label='frozen'),
        'hold_rates_targets': lambda: hold(target_off=600, target_on=250,
            params=dict(TOY_PARAMS, fixed_thr=40.0), label='frozen2', apl_bracket=(0.5, 4.0)),
        'simulate_population': population,
    }


SOLVER = {'calibrate_readout', 'calibrate_session_zscore', 'calibrate_session_zscore_sat'}
_STATE = {}


def _frozen():
    if not _STATE:
        fz = load(FROZEN)
        _STATE.update(fz=fz, O=objects(fz['in']))
    return _STATE['fz'], _STATE['O']


@pytest.mark.parametrize('case', sorted(cases(_frozen()[1])))
def test_frozen_case(case):
    fz, O = _frozen()
    check(fz, case, cases(O)[case](), solver=case in SOLVER)


def test_every_frozen_case_covered():
    fz, O = _frozen()
    assert {k.split('/')[0] for k in fz['out']} == set(cases(O))
