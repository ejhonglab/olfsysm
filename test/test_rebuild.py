"""Tests of pyolfsysm.rebuild: the breadth weighting and the glomerulus gain moved from the KC-remapping queue (against
their definitions), the archive's round trip and integrity check, the threshold re-tuning on the tuning tests' toy
simulator, and, on the model (the plain hemibrain wiring with connectome APL weights), a rebuild from an archive
reproducing the fixed-parameter simulator's APL-on and APL-off outputs exactly. The queue's fitted models are checked
from their archives by its archive_tools/rebuild_archive.py (and test/test_rebuild_archive.py there)."""
import json

import numpy as np
import pandas as pd
import pytest

from pyolfsysm import rebuild as R
from pyolfsysm import tuning as T
from .frozen_outputs import TOY_PARAMS, toy_simulator

pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')


def _claw_weights(n_kcs=5, claws=(1, 3, 2, 4, 1), seed=0):
    rng = np.random.default_rng(seed)
    rows = [(kc, c) for kc, n in zip(range(100, 100 + n_kcs), claws) for c in range(n)]
    idx = pd.MultiIndex.from_tuples(rows, names=['kc_id', 'claw_id'])
    return pd.Series(rng.uniform(0.5, 2.0, len(rows)), index=idx, name='w')


def test_breadth_weights_definition():
    w0 = _claw_weights()
    breadth = pd.Series([0.0, 3.0, 10.0, 1.0], index=pd.Index([100, 101, 102, 103], name='kc_id'))
    for k in (0.0, 1.0, 2.0):
        w = R.breadth_weights(w0, breadth, k)
        f = (breadth + 1) / (breadth + 1).mean()
        for (kc, c), v in w.items():
            # KC 104 has no breadth: factor 1
            expect = w0[(kc, c)] * (f[kc] ** k if kc in f.index else 1.0)
            assert v == pytest.approx(expect, rel=1e-15)
        assert w.index.equals(w0.index)
    assert R.breadth_weights(w0, breadth, 0.0).equals(w0)


def test_glomerulus_gain():
    cols = pd.MultiIndex.from_tuples([('DM1', 1), ('DM1', 2), ('VA2', 3), ('DL1', 4)], names=['glomerulus', 'bouton'])
    W = pd.DataFrame(np.arange(12).reshape(3, 4) % 2, index=pd.Index([1, 2, 3], name='claw'), columns=cols)
    G = R.glomerulus_gain(W, ['DM1', 'DL1'], 1.5)
    assert G.dtypes.eq(float).all()
    assert np.array_equal(G.to_numpy(), W.to_numpy() * np.array([1.5, 1.5, 1.0, 1.5]))
    assert R.glomerulus_gain(W, [], 3.0).equals(W.astype(float))


def _pieces():
    w = _claw_weights()
    kcs = pd.Index(sorted(set(w.index.get_level_values('kc_id'))), name='kc_id')
    cols = pd.MultiIndex.from_tuples([('DM1', 1), ('VA2', 2), ('DL1', 3)], names=['glomerulus', 'bouton'])
    wiring = pd.DataFrame(np.random.default_rng(1).integers(0, 2, (len(w), 3)).astype(float), index=w.index, columns=cols)
    bout = pd.Series([1.0, 2.0, 0.5], index=cols, name='b')
    weights = dict(wAPLKC=w, wKCAPL=w * 0.5, wAPLPN=bout, wPNAPL=bout * 3.0)
    params = dict(fixed_thr=100.0, wAPLKC_scale=0.2, wKCAPL_scale=1e-4, wAPLPN_scale=0.3, wPNAPL_scale=5e-4,
                  wKCKC_scale=20.0)
    odors = pd.MultiIndex.from_tuples([('p', 'a @ -3'), ('p', 'b @ -2')], names=['panel', 'odor'])
    orn = pd.DataFrame([[10.0, 0.0], [5.5, 80.25], [1e-3, 2.0]], index=pd.Index(['DM1', 'VA2', 'DL1'], name='glomerulus'),
                       columns=odors)
    out_idx = pd.MultiIndex.from_arrays([kcs, ['g'] * len(kcs)], names=['kc_id', 'kc_type'])
    oc = pd.Index(['a @ -3', 'b @ -2'], name='odor')
    s = pd.DataFrame([[0, 3], [5, 1], [2, 2], [0, 0], [7, 4]], index=out_idx, columns=oc).astype(float)
    outputs = {'on': ((s >= 2).astype(float), s), 'off': ((s * 2 >= 2).astype(float), s * 2)}
    breadth = pd.Series([1.0, 2.0, 0.0, 4.0, 5.0], index=kcs)
    model_kws = dict(one_row_per_claw=True, prat_boutons=True, n_spikes_for_response=2, kc_kc_tau=0.05, claw_thr=40.0)
    return dict(params=params, wiring=wiring, weights=weights, model_kws=model_kws, orn_deltas=orn, outputs=outputs,
                breadth=breadth)


def test_model_kwargs_and_scaled_weights():
    P = _pieces()
    kws = R.model_kwargs(P['params'], P['wiring'], P['weights'], P['model_kws'])
    assert kws['fixed_thr'] == 100.0 and kws['wAPLKC'] == 0.2 and kws['kc_kc_inh'] == 20.0
    assert kws['_wPNKC'] is P['wiring'] and kws['_wAPLKC'] is P['weights']['wAPLKC']
    assert kws['_wPNAPL'] is P['weights']['wPNAPL'] and kws['claw_thr'] == 40.0
    with pytest.raises(ValueError, match='must not hold'):
        R.model_kwargs(P['params'], P['wiring'], P['weights'], dict(P['model_kws'], fixed_thr=1.0))
    with pytest.raises(ValueError, match='both wAPLPN and wPNAPL'):
        R.model_kwargs(P['params'], P['wiring'], {k: P['weights'][k] for k in ('wAPLKC', 'wKCAPL', 'wAPLPN')})
    sw = R.scaled_weights(P['weights'], P['params'])
    assert set(sw) == set(R.WEIGHT_KEYS)
    assert sw['wAPLPN'].equals(P['weights']['wAPLPN'] * 0.3)


def test_archive_round_trip(tmp_path):
    P = _pieces()
    meta = dict(experiment='E0_test', tag='t', wiring=dict(connectome='hemibrain', shuffle=None))
    js = R.write_archive(tmp_path / 'a', params=P['params'], wiring=P['wiring'], weights=P['weights'],
                         model_kws=P['model_kws'], orn_deltas=P['orn_deltas'], outputs=P['outputs'],
                         breadth=P['breadth'], meta=meta)
    assert js == tmp_path / 'a' / R.ARCHIVE_JSON
    rec = json.loads(js.read_text())
    assert rec['format'] == R.ARCHIVE_FORMAT and rec['params'] == P['params'] and rec['experiment'] == 'E0_test'
    assert rec['response_criterion']['n_spikes_for_response'] == 2 and rec['apl_off'] == T.APL_OFF
    assert R.apl_off_overrides({'fixed_thr': 1.0, 'wAPLKC_scale': 1.0, 'wKCAPL_scale': 1.0}) == {'wAPLKC_scale': 0.0}
    # no file of an archive looks like an experiment folder's <variant>_<condition>_spike_counts.parquet
    assert not list((tmp_path / 'a').glob('*_spike_counts.parquet')) and not list((tmp_path / 'a').glob('*_responses.parquet'))
    for p in (js, js.parent):
        a = R.load_archive(p)
        assert a['params'] == P['params'] and a['model_kws'] == P['model_kws']
        assert a['wiring'].equals(P['wiring']) and a['orn_deltas'].equals(P['orn_deltas'])
        assert a['breadth'].equals(P['breadth'])
        for k, w in P['weights'].items():
            assert a['weights'][k].equals(w) and a['weights'][k].index.equals(w.index) and a['weights'][k].name == w.name
        for c in ('on', 'off'):
            for x, y in zip(a['outputs'][c], P['outputs'][c]):
                assert x.equals(y)
        assert a['meta']['wiring'] == meta['wiring']
    # a changed file is refused
    f = tmp_path / 'a' / 'wAPLKC.parquet'
    df = pd.read_parquet(f)
    df.iloc[0, 0] += 1.0
    df.to_parquet(f)
    with pytest.raises(ValueError, match='sha256'):
        R.load_archive(tmp_path / 'a')
    assert R.load_archive(tmp_path / 'a', verify=False)['weights']['wAPLKC'].iloc[0] != P['weights']['wAPLKC'].iloc[0]


def test_write_archive_refusals(tmp_path):
    P = _pieces()
    kw = dict(params=P['params'], wiring=P['wiring'], weights=P['weights'], model_kws=P['model_kws'])
    with pytest.raises(ValueError, match='meta must not set'):
        R.write_archive(tmp_path / 'b', **kw, meta=dict(params={}))
    with pytest.raises(TypeError, match='only scalars'):
        R.write_archive(tmp_path / 'c', **dict(kw, model_kws=dict(P['model_kws'], x=np.zeros(3))))
    with pytest.raises(ValueError, match='unknown parameters'):
        R.write_archive(tmp_path / 'd', **dict(kw, params=dict(P['params'], foo=1.0)))


def test_tune_threshold_on_the_toy_simulator():
    sim = toy_simulator(seed=3)
    p = dict(TOY_PARAMS)
    n1 = T.n_responses(sim(p)[0])
    res = R.tune_threshold(sim, p, target_n=n1)
    assert res['factor'] == 1.0 and res['params'] == p and len(res['history']) == 1
    for target in (300, 900):
        res = R.tune_threshold(sim, p, target_n=target)
        assert not res['at_bound'] and res['n_responses'] == T.n_responses(res['outputs'][0])
        assert abs(res['n_responses'] - target) <= min(abs(h['n_responses'] - target) for h in res['history'])
        assert res['params']['fixed_thr'] == p['fixed_thr'] * res['factor']
        assert {k: v for k, v in res['params'].items() if k != 'fixed_thr'} == \
            {k: v for k, v in p.items() if k != 'fixed_thr'}
        assert res['outputs'][1].equals(sim(res['params'])[1])
    size = 200 * 16
    res = R.tune_threshold(sim, p, target_rate=0.1, apl_off=True)
    assert res['target_n'] == round(0.1 * size) and res['rate'] == res['n_responses'] / size
    # tuned with APL off, the returned parameters keep the APL scales
    assert res['params']['wAPLKC_scale'] == p['wAPLKC_scale']
    assert res['outputs'][1].equals(sim(dict(res['params'], **T.APL_OFF))[1])
    with pytest.raises(ValueError, match='one of'):
        R.tune_threshold(sim, p)


def test_rebuild_reproduces_the_model(tmp_path):
    """The plain hemibrain wiring with connectome APL weights (one row per KC) at fixed parameters: the archived
    pieces rebuild the fixed-parameter simulator's APL-on and APL-off outputs exactly, through fit_mb_model and through
    the simulator, and the re-tuned threshold reaches its target."""
    from pyolfsysm.mb_model import fit_mb_model, make_fixed_param_kc_simulator, megamat_orn_deltas
    orn = megamat_orn_deltas()
    kws = dict(weight_divisor=20, use_connectome_APL_weights=True, make_plots=False, silent=True)
    r0, s0, wPNKC, pd0 = fit_mb_model(orn, **kws)
    weights = {k: pd0[k] / pd0[f'{k}_scale'] for k in ('wAPLKC', 'wKCAPL')}
    params = dict(fixed_thr=float(pd0['fixed_thr']), wAPLKC_scale=float(pd0['wAPLKC_scale']) * 1.1,
                  wKCAPL_scale=float(pd0['wKCAPL_scale']) * 1.1)
    sim, p = make_fixed_param_kc_simulator(orn, initial_params=params, _wPNKC=wPNKC, _wAPLKC=weights['wAPLKC'],
                                           _wKCAPL=weights['wKCAPL'], **kws)
    outputs = {'on': sim(p), 'off': sim(dict(p, wAPLKC_scale=0.0))}
    del sim
    a = tmp_path / 'm'
    R.write_archive(a, params=params, wiring=wPNKC, weights=weights, model_kws=kws, orn_deltas=orn, outputs=outputs)
    res = R.check_archive(a, record=True)
    assert res['ok'], res
    for route in ('fit_mb_model', 'simulator'):
        for c in ('on', 'off'):
            assert res[route][c]['spike_count_diffs'] == 0 and res[route][c]['n_entries'] == s0.size
    assert json.loads((a / R.ARCHIVE_JSON).read_text())['checks'][0]['ok']
    # APL off has more responses than APL on
    assert T.n_responses(outputs['off'][0]) > T.n_responses(outputs['on'][0])
    # the rebuilt simulator, re-tuned to a lower response rate with APL on
    arch = R.load_archive(a)
    sim, p = R.make_simulator(arch['params'], arch['wiring'], arch['weights'], arch['orn_deltas'],
                              model_kws=arch['model_kws'])
    target = T.n_responses(outputs['on'][0]) // 2
    t = R.tune_threshold(sim, p, target_n=target)
    assert t['factor'] > 1.0 and abs(t['n_responses'] - target) <= max(5, target // 50)
