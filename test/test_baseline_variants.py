"""Tests of the bouton base's variants for olfdb's baseline models (`pyolfsysm.baselines`):
`pn2kc_connections='uniform-glomeruli-boutons'`, `uniform_apl_weights` and
`shuffle_connectome_seed` / `shuffle_apl_pairs` (`fit_mb_model` kwargs).
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import chisquare

from pyolfsysm.mb_model import (fit_mb_model, connectome_wPNKC, uniform_bouton_wPNKC,
    connectome_shuffle_perms, permute_rows, megamat_orn_deltas, format_model_params,
    get_fitmbmodel_default, glomerulus_col, KC_ID, WIRING_CONTROL_PARAMS
)
from pyolfsysm.baselines import BOUTON_BASE_KWS, BASELINE_VARIANTS, baseline_kws


pytestmark = pytest.mark.filterwarnings('ignore')

# the bouton base, tuned on fewer odors (faster)
BASE = {k: v for k, v in BOUTON_BASE_KWS.items()}
N_ODORS = 4


@pytest.fixture(scope='module')
def orn_deltas() -> pd.DataFrame:
    return megamat_orn_deltas().iloc[:, :N_ODORS]


@pytest.fixture(scope='module')
def bouton_wPNKC() -> pd.DataFrame:
    return connectome_wPNKC(prat_claws=True, prat_boutons=True)


def _fit(orn_deltas, **kws):
    responses, spike_counts, wPNKC, params = fit_mb_model(orn_deltas, make_plots=False,
        silent=True, return_olfsysm_vars=True, **kws
    )
    rv = params.pop('rv')
    params.pop('mp')
    weights = {
        'wAPLKC': np.asarray(rv.kc.wAPLKC_unscaled, dtype=float).squeeze().copy(),
        'wKCAPL': np.asarray(rv.kc.wKCAPL_unscaled, dtype=float).squeeze().copy(),
        'wAPLPN': np.asarray(rv.pn.wAPLPN_unscaled, dtype=float).squeeze().copy(),
        'wPNAPL': np.asarray(rv.pn.wPNAPL_unscaled, dtype=float).squeeze().copy(),
        'wPNKC': np.asarray(rv.kc.wPNKC, dtype=float).copy(),
    }
    del rv
    return responses, spike_counts, wPNKC, params, weights


@pytest.fixture(scope='module')
def base_fit(orn_deltas):
    return _fit(orn_deltas, **BASE)


def test_uniform_glomeruli_wPNKC(bouton_wPNKC):
    """glomerulus_probs='uniform': 7 claws per KC, glomeruli drawn uniformly, seeded.
    """
    n_claws = 7
    wPNKC = uniform_bouton_wPNKC(bouton_wPNKC, n_claws=n_claws, seed=0,
        glomerulus_probs='uniform'
    )
    assert wPNKC.columns.equals(bouton_wPNKC.columns)
    assert wPNKC.index.is_monotonic_increasing
    assert (wPNKC.sum(axis='columns') == 1).all()
    n_kcs = bouton_wPNKC.index.get_level_values(KC_ID).nunique()
    assert len(wPNKC) == n_kcs * n_claws
    assert (wPNKC.groupby(level=KC_ID).size() == n_claws).all()

    gloms = bouton_wPNKC.columns.get_level_values(glomerulus_col)
    n_gloms = gloms.nunique()
    n_claws_per_glom = wPNKC.sum().groupby(level=glomerulus_col).sum()
    assert len(n_claws_per_glom) == n_gloms
    # roughly uniform over glomeruli (not in proportion to their # of boutons)
    assert chisquare(n_claws_per_glom).pvalue > 0.01
    n_boutons = pd.Series(gloms).value_counts()
    p_boutons = (n_boutons / n_boutons.sum()).reindex(n_claws_per_glom.index)
    assert chisquare(n_claws_per_glom, len(wPNKC) * p_boutons).pvalue < 1e-6

    assert uniform_bouton_wPNKC(bouton_wPNKC, n_claws=n_claws, seed=0,
        glomerulus_probs='uniform').equals(wPNKC)
    assert not uniform_bouton_wPNKC(bouton_wPNKC, n_claws=n_claws, seed=1,
        glomerulus_probs='uniform').equals(wPNKC)
    # default unchanged
    assert uniform_bouton_wPNKC(bouton_wPNKC, n_claws=n_claws, seed=0).equals(
        uniform_bouton_wPNKC(bouton_wPNKC, n_claws=n_claws, seed=0,
            glomerulus_probs='boutons')
    )
    with pytest.raises(ValueError):
        uniform_bouton_wPNKC(bouton_wPNKC, glomerulus_probs='glomeruli')


def test_connectome_shuffle_perms():
    kept = connectome_shuffle_perms(100, 30, 0, keep_apl_pairs=True)
    broken = connectome_shuffle_perms(100, 30, 0, keep_apl_pairs=False)
    for k, n in (('wPNKC', 100), ('wAPLKC', 100), ('wKCAPL', 100), ('wAPLPN', 30),
            ('wPNAPL', 30)):
        assert np.array_equal(np.sort(kept[k]), np.arange(n))
        assert np.array_equal(np.sort(broken[k]), np.arange(n))
    for k in ('wPNKC', 'wAPLKC', 'wAPLPN'):
        assert np.array_equal(kept[k], broken[k])
    assert kept['wKCAPL'] is kept['wAPLKC'] and kept['wPNAPL'] is kept['wAPLPN']
    assert not np.array_equal(broken['wKCAPL'], broken['wAPLKC'])
    assert not np.array_equal(broken['wPNAPL'], broken['wAPLPN'])
    # PN->KC independent of KC<>APL
    assert not np.array_equal(kept['wPNKC'], kept['wAPLKC'])

    again = connectome_shuffle_perms(100, 30, 0, keep_apl_pairs=False)
    assert all(np.array_equal(again[k], broken[k]) for k in broken)
    other = connectome_shuffle_perms(100, 30, 1)
    assert not np.array_equal(other['wPNKC'], kept['wPNKC'])

    s = pd.Series(np.arange(5.0), index=list('abcde'), name='weight')
    p = permute_rows(s, np.array([4, 0, 1, 2, 3]))
    assert p.index.equals(s.index) and p.name == 'weight'
    assert list(p.values) == [4, 0, 1, 2, 3]


def test_option_validation(orn_deltas):
    with pytest.raises(ValueError):
        # shuffling needs the connectome claws
        fit_mb_model(orn_deltas, **dict(BASE, prat_claws=False,
            pn2kc_connections='uniform-boutons', shuffle_connectome_seed=0
        ))
    with pytest.raises(ValueError):
        fit_mb_model(orn_deltas, **dict(BASE, shuffle_apl_pairs=False))
    with pytest.raises(ValueError):
        fit_mb_model(orn_deltas, **dict(BASE, prat_boutons=False, pn_claw_to_apl=False,
            uniform_apl_weights=True
        ))
    with pytest.raises(NotImplementedError):
        fit_mb_model(orn_deltas, **dict(BASE, apl_unit_coupling=1.0,
            shuffle_connectome_seed=0
        ))


def test_defaults_unchanged(orn_deltas, base_fit):
    """The new kwargs at their defaults give the bouton base to the last digit.
    """
    responses, spike_counts, wPNKC, params, weights = base_fit
    explicit = _fit(orn_deltas, **BASE, uniform_apl_weights=False,
        shuffle_connectome_seed=None, shuffle_apl_pairs=True
    )
    assert responses.equals(explicit[0])
    assert spike_counts.equals(explicit[1])
    assert wPNKC.equals(explicit[2])
    for k in ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale',
            'wPNAPL_scale'):
        assert np.array_equal(params[k], explicit[3][k]), k
    assert not any(k in params for k in WIRING_CONTROL_PARAMS)
    assert set(params) == set(explicit[3])
    # and nothing new in the model's ID
    assert format_model_params(BASE) == format_model_params(dict(BASE,
        uniform_apl_weights=False, shuffle_connectome_seed=None, shuffle_apl_pairs=True
    ))


def test_uniform_glomeruli_uniform_apl(orn_deltas, bouton_wPNKC):
    kws = baseline_kws('uniform_glomeruli_claws_uniform_apl')
    kws.pop('target_sparsity')
    responses, _, wPNKC, params, weights = _fit(orn_deltas, target_sparsity=0.05, **kws)
    expected = uniform_bouton_wPNKC(bouton_wPNKC, n_claws=7, seed=0,
        glomerulus_probs='uniform'
    )
    assert wPNKC.equals(expected)
    assert np.array_equal(weights['wPNKC'], expected.values)
    for k in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL'):
        assert (weights[k] == 1).all(), k
    assert len(weights['wAPLPN']) == bouton_wPNKC.shape[1]
    assert params['uniform_apl_weights'] is True
    assert 'shuffle_connectome_seed' not in params
    assert 'uniform-glomeruli-boutons' in format_model_params(kws)
    assert 'uniform' in format_model_params(kws) and 'apl' in format_model_params(kws)


def test_uniform_apl_weights_connectome(orn_deltas, base_fit):
    _, _, wPNKC, params, weights = _fit(orn_deltas, **BASE, uniform_apl_weights=True)
    assert wPNKC.equals(base_fit[2])
    for k in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL'):
        assert (weights[k] == 1).all(), k
        # (the base's are synapse counts)
        assert not (base_fit[4][k] == 1).all(), k


@pytest.mark.parametrize('pairs', [True, False])
def test_shuffle_connectome(orn_deltas, base_fit, pairs):
    _, _, base_wPNKC, _, base_w = base_fit
    seed = 0
    _, _, wPNKC, params, weights = _fit(orn_deltas, **BASE,
        shuffle_connectome_seed=seed, shuffle_apl_pairs=pairs
    )
    perms = connectome_shuffle_perms(base_wPNKC.shape[0], base_wPNKC.shape[1], seed,
        keep_apl_pairs=pairs
    )
    # PN->KC: claws (and their index: KC, claw ID, coordinates) stay; their input rows
    # are permuted
    assert wPNKC.index.equals(base_wPNKC.index)
    assert wPNKC.columns.equals(base_wPNKC.columns)
    assert wPNKC.equals(permute_rows(base_wPNKC, perms['wPNKC']))
    assert np.array_equal(weights['wPNKC'], wPNKC.values)
    # each bouton keeps its total output, each KC its claw count, total preserved
    assert np.array_equal(wPNKC.sum().values, base_wPNKC.sum().values)
    assert (wPNKC.groupby(level=KC_ID).size()
        == base_wPNKC.groupby(level=KC_ID).size()).all()
    assert wPNKC.values.sum() == base_wPNKC.values.sum()
    # identities change
    changed = (wPNKC.values != base_wPNKC.values).any(axis=1).mean()
    assert changed > 0.9
    per_kc_gloms = lambda w: w.T.groupby(level=glomerulus_col).sum().T.groupby(
        level=KC_ID).sum()
    assert not per_kc_gloms(wPNKC).equals(per_kc_gloms(base_wPNKC))

    # APL weights (normalized by olfsysm, which permuting leaves as is: same mean)
    for k in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL'):
        assert np.allclose(weights[k], base_w[k][perms[k]], rtol=1e-12, atol=0), k
        assert np.isclose(weights[k].sum(), base_w[k].sum())
        assert not np.array_equal(weights[k], base_w[k]), k
    # KC<>APL permuted independently of PN->KC
    assert not np.array_equal(perms['wAPLKC'], perms['wPNKC'])

    def pairs_of(w, a, b):
        return sorted(zip(np.round(w[a], 9), np.round(w[b], 9)))

    for a, b in (('wAPLKC', 'wKCAPL'), ('wAPLPN', 'wPNAPL')):
        same_pairs = pairs_of(weights, a, b) == pairs_of(base_w, a, b)
        assert same_pairs == pairs, (a, b)

    assert params['shuffle_connectome_seed'] == seed
    assert params['shuffle_apl_pairs'] is pairs
    model_id = format_model_params(dict(BASE, shuffle_connectome_seed=seed,
        shuffle_apl_pairs=pairs
    ))
    assert 'shuffle' in model_id
    assert ('pairs' in model_id) == (not pairs)


def test_baseline_variants():
    assert BASELINE_VARIANTS['bouton_base'] == BOUTON_BASE_KWS
    assert BOUTON_BASE_KWS['target_sparsity'] == 0.05
    assert get_fitmbmodel_default('pn2kc_connections') == 'hemibrain'
    for name, kws in BASELINE_VARIANTS.items():
        assert {k: v for k, v in kws.items() if k in BOUTON_BASE_KWS and k != 'prat_claws'
            } == {k: v for k, v in BOUTON_BASE_KWS.items() if k != 'prat_claws'}, name
    assert BASELINE_VARIANTS['shuffled_connectome_pairs_broken_seed0'][
        'shuffle_apl_pairs'] is False
    with pytest.raises(ValueError):
        baseline_kws('nope')
    assert baseline_kws('bouton_base', seed=3)['seed'] == 3
