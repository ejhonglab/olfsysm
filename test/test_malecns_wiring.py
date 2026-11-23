"""Tests of the male CNS's own wiring: `connectome_wPNKC` / `connectome_APL_weights` /
`fit_mb_model` with `pn2kc_connections` (`connectome`) 'malecns-left' or 'malecns-right',
read from the model tables in `malecns_data_dir` (copied from hong-data 8dc9cea; README.md
there).
"""
import hashlib
import re

import numpy as np
import pandas as pd
import pytest

from pyolfsysm.mb_model import (connectome_wPNKC, connectome_APL_weights, fit_mb_model,
    format_model_params, load_malecns_table, claw_compartments_shell_sphere,
    malecns_data_dir, malecns_connectome_options, connectome_options,
    pn2kc_connections_options, megamat_orn_deltas, task_gloms, claw_coord_cols,
    glomerulus_col, fixed_params_to_fit_mb_model_kws, KC_ID, CLAW_ID, KC_TYPE, PN_ID,
    BOUTON_ID
)

pytestmark = pytest.mark.filterwarnings('ignore')

SIDES = sorted(malecns_connectome_options)
# claws, KCs, boutons of each side's wiring (README.md in malecns_data_dir; E97's counts)
EXPECTED = {'malecns-left': (10077, 1864, 471), 'malecns-right': (10008, 1884, 434)}


@pytest.fixture(scope='module', params=SIDES)
def side(request):
    return request.param


@pytest.fixture(scope='module')
def wirings():
    return {c: connectome_wPNKC(c, prat_claws=True, prat_boutons=True) for c in SIDES}


def test_options():
    assert set(SIDES) == {'malecns-left', 'malecns-right'}
    assert set(SIDES) <= connectome_options and set(SIDES) <= pn2kc_connections_options


def test_data_files_match_readme():
    """The copied tables are those whose sha256 the folder's README lists."""
    readme = (malecns_data_dir / 'README.md').read_text()
    listed = dict((name, h) for h, name in re.findall(r'([0-9a-f]{64})\s+(\S+\.parquet)',
        readme))
    assert len(listed) == 10
    for name, h in listed.items():
        assert hashlib.sha256((malecns_data_dir / name).read_bytes()).hexdigest() == h, name


def test_wiring(side, wirings):
    w = wirings[side]
    n_claws, n_kcs, n_boutons = EXPECTED[side]
    assert w.shape == (n_claws, n_boutons)
    assert list(w.index.names) == [KC_ID, CLAW_ID, KC_TYPE] + claw_coord_cols + [
        'compartment']
    assert list(w.columns.names) == [glomerulus_col, PN_ID, BOUTON_ID]
    idx = w.index.to_frame(index=False)
    assert idx[KC_ID].nunique() == n_kcs
    assert not idx[[KC_ID, CLAW_ID]].duplicated().any()
    assert idx[KC_ID].is_monotonic_increasing
    assert set(idx[KC_TYPE]) == {'g', 'ab', "a'b'"}
    assert w.columns.is_monotonic_increasing
    assert set(w.columns.get_level_values(glomerulus_col)) <= set(task_gloms)
    # one bouton per claw, weight 1, every bouton with a claw
    assert set(np.unique(w.to_numpy())) == {0.0, 1.0}
    assert ((w > 0).sum(axis='columns') == 1).all() and (w.sum() > 0).all()
    # the hemibrain's compartment rule, over this side's claws
    comp = claw_compartments_shell_sphere(idx[claw_coord_cols].to_numpy())
    assert np.array_equal(idx['compartment'].to_numpy(), comp)
    assert set(np.unique(comp)) == {0, 1}
    # the table's bouton of each claw
    t = load_malecns_table(side, 'pn_kc').sort_values([KC_ID, CLAW_ID])
    cols = w.columns.to_frame(index=False).iloc[w.to_numpy().argmax(axis=1)]
    assert np.array_equal(cols[PN_ID].to_numpy(), t[PN_ID].to_numpy())
    assert np.array_equal(cols[BOUTON_ID].to_numpy(), t[BOUTON_ID].to_numpy())
    assert (cols[glomerulus_col].to_numpy() == t['model_glomerulus'].to_numpy()).all()


def test_glomerulus_wiring_is_boutons_summed(side, wirings):
    w = wirings[side]
    g = connectome_wPNKC(side, prat_claws=True)
    assert g.index.equals(w.index)
    assert list(g.columns.names) == [glomerulus_col]
    summed = w.T.groupby(level=glomerulus_col).sum().T.astype(np.int64)
    assert summed.reindex(columns=g.columns).equals(g)


def test_claw_weight_synapses(side, wirings):
    raw = connectome_wPNKC(side, prat_claws=True, prat_boutons=True,
        claw_weight='synapses_raw'
    )
    assert ((raw > 0) == (wirings[side] > 0)).all().all()
    assert (raw.sum(axis='columns') >= 5).all()
    norm = connectome_wPNKC(side, prat_claws=True, prat_boutons=True,
        claw_weight='synapses'
    )
    assert np.isclose(norm.to_numpy()[norm.to_numpy() > 0].mean(), 1.0)


def test_apl_weights(side, wirings):
    w = wirings[side]
    wAPLKC, wKCAPL, wAPLPN, wPNAPL = connectome_APL_weights(side, prat_claws=True,
        prat_boutons=True, wPNKC=w
    )
    for x in (wAPLKC, wKCAPL):
        assert x.index.equals(w.index) and x.name == 'weight'
    for x in (wAPLPN, wPNAPL):
        assert x.index.equals(w.columns) and x.name == 'weight'
    for x, name in ((wAPLKC, 'apl_kc'), (wKCAPL, 'kc_apl'), (wAPLPN, 'apl_pn'),
            (wPNAPL, 'pn_apl')):
        assert (x >= 0).all() and (x > 0).mean() > 0.5
        t = load_malecns_table(side, name)
        assert np.isclose(x.sum(), t['weight'].sum())
    # a subset of the claws (as the wiring restricted to some glomeruli) keeps its weights
    sub = w.iloc[::7]
    sub = sub.loc[:, (sub > 0).any()]
    a = connectome_APL_weights(side, prat_claws=True, prat_boutons=True, wPNKC=sub)
    assert np.array_equal(a[0].to_numpy(), wAPLKC.iloc[::7].to_numpy())
    assert np.array_equal(a[2].to_numpy(), wAPLPN.reindex(sub.columns).to_numpy())
    # without boutons
    g = connectome_wPNKC(side, prat_claws=True)
    b = connectome_APL_weights(side, prat_claws=True, wPNKC=g)
    assert b[2] is None and b[3] is None
    assert np.array_equal(b[0].to_numpy(), wAPLKC.to_numpy())


def test_apl_weights_refuse_other_wiring(wirings):
    with pytest.raises(ValueError):
        connectome_APL_weights('malecns-right', prat_claws=True, prat_boutons=True,
            wPNKC=wirings['malecns-left']
        )


@pytest.mark.parametrize('kws', [
        dict(prat_claws=False),
        dict(prat_claws=True, prat_boutons=True, multibouton_claws='first'),
        dict(prat_claws=True, prat_boutons=True, multibouton_claws='split'),
        dict(prat_claws=True, dist_weight='raw'),
        dict(prat_claws=True, Btn_separate=True),
    ], ids=str
)
def test_unsupported(kws):
    with pytest.raises(NotImplementedError):
        connectome_wPNKC('malecns-left', **kws)


def test_unsupported_apl():
    with pytest.raises(NotImplementedError):
        connectome_APL_weights('malecns-left', prat_claws=True, apl_units=True)


def test_model_ids():
    base = dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True)
    ids = {c: format_model_params(dict(base, pn2kc_connections=c))
        for c in SIDES + ['hemibrain']
    }
    assert len(set(ids.values())) == 3
    assert all(c in ids[c] for c in SIDES)


BOUTON_MODEL = dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
    use_connectome_APL_weights=True, pn_claw_to_apl=True, n_spikes_for_response=2
)


@pytest.mark.parametrize('connectome,kws', [
        ('malecns-left', BOUTON_MODEL),
        ('malecns-right', dict(one_row_per_claw=True, prat_claws=True,
            use_connectome_APL_weights=True, n_spikes_for_response=2)),
    ], ids=['left-boutons', 'right-claws']
)
def test_fit_mb_model_runs(connectome, kws, wirings):
    """`fit_mb_model` runs on each side's wiring at fixed parameters (4 megamat odors),
    with the wiring and the KCs of the male CNS."""
    orn = megamat_orn_deltas().iloc[:, :4]
    responses, spike_counts, wPNKC, params = fit_mb_model(orn, fixed_thr=275.0,
        wAPLKC=0.13, pn2kc_connections=connectome, make_plots=False, silent=True, **kws
    )
    w = wirings[connectome] if kws.get('prat_boutons') else connectome_wPNKC(connectome,
        prat_claws=True
    )
    assert wPNKC.index.equals(w.index)
    assert set(responses.index.get_level_values(KC_ID)) == set(
        w.index.get_level_values(KC_ID))
    assert responses.shape[1] == 4 and spike_counts.shape == responses.shape
    rate = float(responses.mean().mean())
    assert 0 < rate < 0.5


APL_WEIGHTS = ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL')


def test_hardcoded_weights(side):
    """`fit_mb_model` takes hardcoded weights (`_wPNKC`, `_wAPLKC` / `_wKCAPL` / `_wAPLPN` /
    `_wPNAPL`) with `pn2kc_connections` 'malecns-left' / 'malecns-right' (refused until
    2026-11-23): the side's own wiring and normalized APL weights, from a run at fixed
    parameters, reproduce that run's spike counts exactly, and doubling the KC<>APL scales
    lowers the response rate. KC-KC weights still come only from `_wKCKC` on these sides,
    and wirings other than the hemibrain and the male CNS still refuse hardcoded weights.
    """
    orn = megamat_orn_deltas().iloc[:, :4]
    kws = dict(BOUTON_MODEL, pn2kc_connections=side, make_plots=False, silent=True)
    r0, s0, w0, p0 = fit_mb_model(orn, fixed_thr=275.0, wAPLKC=0.13, **kws)
    hard = {f'_{k}': p0[k] / p0[f'{k}_scale'] for k in APL_WEIGHTS}
    fixed = fixed_params_to_fit_mb_model_kws(dict(fixed_thr=p0['fixed_thr'],
        **{f'{k}_scale': p0[f'{k}_scale'] for k in APL_WEIGHTS}
    ))
    r1, s1, w1, _ = fit_mb_model(orn, _wPNKC=w0, **hard, **fixed, **kws)
    assert w1.equals(w0)
    assert s1.equals(s0) and r1.equals(r0)
    assert r0.values.sum() > 0

    stronger = dict(fixed, wAPLKC=2 * fixed['wAPLKC'], wKCAPL=2 * fixed['wKCAPL'])
    r2, _, _, _ = fit_mb_model(orn, _wPNKC=w0, **hard, **stronger, **kws)
    assert r2.values.sum() < r0.values.sum()

    # no KC-KC synapses of the male CNS in olfsysm (they are passed as `_wKCKC`), and the
    # hemibrain's are not used in their place
    with pytest.raises(NotImplementedError):
        fit_mb_model(orn, _wPNKC=w0, **hard, **fixed, kc_kc_inh=1.0, **kws)

    with pytest.raises(AssertionError, match='pn2kc_connections'):
        fit_mb_model(orn, _wPNKC=w0, pn2kc_connections='uniform', n_claws=7,
            make_plots=False, silent=True
        )
    with pytest.raises(AssertionError, match='pn2kc_connections'):
        fit_mb_model(orn, **hard, **fixed, **dict(kws, pn2kc_connections='uniform',
            n_claws=7)
        )
