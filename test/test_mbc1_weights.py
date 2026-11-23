"""Tests of the hemibrain MB-C1 loader: `connectome_MBC1_weights` (MB-C1's synapse counts
per claw, KC and PN bouton, from Pratyush's tables in `mbc1_synapse_paths`) and
`connectome_MBC1_KCKC_weights` (MB-C1 as a KC x KC matrix, usable as `fit_mb_model`'s
`_wKCKC`).
"""
import numpy as np
import pandas as pd
import pytest

from pyolfsysm.mb_model import (connectome_wPNKC, connectome_MBC1_weights,
    connectome_MBC1_KCKC_weights, fit_mb_model, megamat_orn_deltas, mbc1_synapse_paths,
    prat_hemibrain_seg_dir, KC_ID, CLAW_ID, KC_TYPE, MBC1_ID
)

pytestmark = pytest.mark.filterwarnings('ignore')

# synapses with both sides in the right calyx, CA(R), per table (both MB-C1 cells)
CALYX_SYNAPSES = {'KC>MBC1': 4906, 'MBC1>KC': 12445, 'PN>MBC1': 953, 'MBC1>PN': 448}
# the hemibrain's two MB-C1_R cells
CELLS = [5813021291, 5813075020]


@pytest.fixture(scope='module')
def wiring():
    return connectome_wPNKC(prat_claws=True, prat_boutons=True)


@pytest.fixture(scope='module')
def claw_weights(wiring):
    dropped = dict()
    weights = connectome_MBC1_weights(prat_claws=True, prat_boutons=True, wPNKC=wiring,
        dropped_counts=dropped
    )
    return weights, dropped


def test_tables():
    """The tables are Pratyush's v5 hemibrain outputs, KC claw IDs from the 2025-12-09
    re-run (those of the wiring's PNbouton-to-KCclaw table)."""
    assert set(mbc1_synapse_paths) == set(CALYX_SYNAPSES)
    for name, path in mbc1_synapse_paths.items():
        assert path.exists() and path.parent == prat_hemibrain_seg_dir, name
    assert '2025-12-09' in mbc1_synapse_paths['KC>MBC1'].name
    assert '2025-12-09' in mbc1_synapse_paths['MBC1>KC'].name


def test_per_kc():
    """Without a claw wiring: all of each KC's calyx synapses with MB-C1, by cell too."""
    w_out, w_in, w_out_pn, w_in_pn = connectome_MBC1_weights()
    assert w_out_pn is None and w_in_pn is None
    assert w_out.sum() == CALYX_SYNAPSES['MBC1>KC']
    assert w_in.sum() == CALYX_SYNAPSES['KC>MBC1']
    for w in (w_out, w_in):
        assert w.index.name == KC_ID and w.index.is_monotonic_increasing
        assert not w.index.duplicated().any() and (w > 0).all()

    by_out, by_in, _, _ = connectome_MBC1_weights(by_cell=True)
    for by, w in ((by_out, w_out), (by_in, w_in)):
        assert list(by.columns) == CELLS and by.columns.name == MBC1_ID
        assert by.index.equals(w.index)
        assert np.array_equal(by.sum(axis=1).to_numpy(), w.to_numpy())
        # about half of the synapses each
        assert ((by.sum() / w.sum()) > 0.4).all()

    # a one-row-per-KC wPNKC gives its index
    kc_index = pd.Index(w_out.index[::5], name=KC_ID)
    sub_out, _, _, _ = connectome_MBC1_weights(wPNKC=pd.DataFrame(index=kc_index))
    assert sub_out.index.equals(kc_index)
    assert np.array_equal(sub_out.to_numpy(), w_out.reindex(kc_index).to_numpy())


def test_per_claw(wiring, claw_weights):
    """On the hemibrain claw x bouton wiring: weights on its rows and bouton columns;
    kept and dropped synapses add up to each KC's calyx synapses; MB-C1 mostly in from
    and out onto gamma KCs."""
    (w_out, w_in, w_out_pn, w_in_pn), dropped = claw_weights
    assert w_out.index.equals(wiring.index) and w_in.index.equals(wiring.index)
    assert w_out_pn.index.equals(wiring.columns) and w_in_pn.index.equals(wiring.columns)
    for w in (w_out, w_in, w_out_pn, w_in_pn):
        assert (w >= 0).all() and np.array_equal(w.to_numpy(), w.round().to_numpy())
        assert w.sum() > 0

    kc_ids = pd.Index(wiring.index.get_level_values(KC_ID).unique(), name=KC_ID)
    all_out, all_in, _, _ = connectome_MBC1_weights()
    for w, name, all_w in ((w_out, 'wMBC1KC', all_out), (w_in, 'wKCMBC1', all_in)):
        d = dropped[name]
        assert d.index.equals(kc_ids)
        assert list(d.columns) == ['nan_claw', 'unassigned_claw', 'other_claw']
        kept = w.groupby(level=KC_ID).sum().reindex(kc_ids)
        total = all_w.reindex(kc_ids).fillna(0)
        assert np.array_equal((kept + d.sum(axis=1)).to_numpy(), total.to_numpy())
        # about half of the wiring's KCs' synapses with MB-C1 are on its claws
        assert kept.sum() / total.sum() > 0.5

    kc_types = wiring.index.get_level_values(KC_TYPE)
    assert w_out[kc_types == 'g'].sum() / w_out.sum() > 0.75
    assert w_in[kc_types == 'g'].sum() / w_in.sum() > 0.75

    assert 0.7 * CALYX_SYNAPSES['MBC1>PN'] < w_out_pn.sum() <= CALYX_SYNAPSES['MBC1>PN']
    assert 0.7 * CALYX_SYNAPSES['PN>MBC1'] < w_in_pn.sum() <= CALYX_SYNAPSES['PN>MBC1']

    by = connectome_MBC1_weights(prat_claws=True, prat_boutons=True, wPNKC=wiring,
        by_cell=True
    )
    for b, w in zip(by, (w_out, w_in, w_out_pn, w_in_pn)):
        assert list(b.columns) == CELLS and b.index.equals(w.index)
        assert np.array_equal(b.sum(axis=1).to_numpy(), w.to_numpy())


def test_claw_ids_of_the_wiring(wiring, claw_weights, monkeypatch):
    """The 2025-12-09 tables put more synapses on the wiring's claws than the 2025-12-05
    ones (which number some claws differently)."""
    (w_out, w_in, _, _), _ = claw_weights
    for name in ('KC>MBC1', 'MBC1>KC'):
        monkeypatch.setitem(mbc1_synapse_paths, name, mbc1_synapse_paths[name].parent /
            mbc1_synapse_paths[name].name.replace('2025-12-09-19-47-36',
                '2025-12-05-13-39-22')
        )
        assert mbc1_synapse_paths[name].exists()
    old_out, old_in, _, _ = connectome_MBC1_weights(prat_claws=True, wPNKC=wiring)
    assert w_out.sum() > old_out.sum() and w_in.sum() > old_in.sum()


def test_kckc_matrix(wiring, claw_weights):
    """MB-C1 as a KC x KC matrix: by default one pooled neuron (rank 1) from all of each
    KC's calyx synapses; each cell a pathway of its own (rank 2) with `pooled=False`, the
    wiring's claws only with `claws_only=True`; normalized as `connectome_KCKC_weights`."""
    kc_ids = pd.Index(wiring.index.get_level_values(KC_ID).unique(), name=KC_ID)
    all_out, all_in, _, _ = connectome_MBC1_weights()
    by_out, by_in, _, _ = connectome_MBC1_weights(by_cell=True)

    m = connectome_MBC1_KCKC_weights(wPNKC=wiring)
    assert m.index.name == f'{KC_ID}_post' and m.columns.name == f'{KC_ID}_pre'
    assert np.array_equal(m.index.to_numpy(), kc_ids.to_numpy())
    assert np.array_equal(m.columns.to_numpy(), kc_ids.to_numpy())
    assert np.isclose(m.to_numpy().sum(axis=1).mean(), 1.0)
    assert np.linalg.matrix_rank(m.to_numpy()) == 1
    # the same from the wiring's KC IDs
    assert m.equals(connectome_MBC1_KCKC_weights(kc_ids))

    # unnormalized, pooled: (all synapses MB-C1 -> post) x (all synapses pre -> MB-C1)
    raw = connectome_MBC1_KCKC_weights(wPNKC=wiring, normalize=False)
    out_kc = all_out.reindex(kc_ids).fillna(0)
    in_kc = all_in.reindex(kc_ids).fillna(0)
    post, pre = out_kc.idxmax(), in_kc.idxmax()
    assert raw.loc[post, pre] == out_kc[post] * in_kc[pre]
    assert raw.to_numpy().sum() == out_kc.sum() * in_kc.sum()

    # each cell a pathway of its own: no cross terms between the cells
    sep = connectome_MBC1_KCKC_weights(wPNKC=wiring, pooled=False)
    assert np.linalg.matrix_rank(sep.to_numpy()) == 2
    assert np.isclose(sep.to_numpy().sum(axis=1).mean(), 1.0)
    raw_sep = connectome_MBC1_KCKC_weights(wPNKC=wiring, pooled=False, normalize=False)
    cell_out = by_out.reindex(kc_ids).fillna(0)
    cell_in = by_in.reindex(kc_ids).fillna(0)
    assert raw_sep.loc[post, pre] == (cell_out.loc[post] * cell_in.loc[pre]).sum()
    assert (raw_sep.to_numpy() <= raw.to_numpy()).all()

    # the wiring's claws only: no more than all of each KC's synapses
    raw_claws = connectome_MBC1_KCKC_weights(wPNKC=wiring, claws_only=True,
        normalize=False
    )
    (w_out, w_in, _, _), _ = claw_weights
    assert raw_claws.index.equals(raw.index) and raw_claws.columns.equals(raw.columns)
    assert raw_claws.to_numpy().sum() == w_out.sum() * w_in.sum()
    assert (raw_claws.to_numpy() <= raw.to_numpy()).all()
    assert raw_claws.to_numpy().sum() < raw.to_numpy().sum()

    every = connectome_MBC1_KCKC_weights()
    assert every.index.equals(every.columns.rename(every.index.name))
    assert np.array_equal(every.index.to_numpy(),
        all_out.index.union(all_in.index).to_numpy()
    )


def test_kckc_matrix_in_fit_mb_model():
    """MB-C1's matrix, passed as `_wKCKC` with `kc_kc_inh`, inhibits the hemibrain
    bouton model at fixed parameters (4 megamat odors)."""
    orn = megamat_orn_deltas().iloc[:, :4]
    kws = dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
        use_connectome_APL_weights=True, pn_claw_to_apl=True, n_spikes_for_response=2,
        fixed_thr=275.0, wAPLKC=0.13, make_plots=False, silent=True
    )
    r0, s0, w0, _ = fit_mb_model(orn, **kws)
    m = connectome_MBC1_KCKC_weights(wPNKC=w0)
    r1, s1, w1, _ = fit_mb_model(orn, kc_kc_inh=30.0, _wKCKC=m, **kws)
    assert w1.equals(w0)
    assert r1.index.equals(r0.index)
    assert 0 < r1.values.sum() < r0.values.sum()
    assert (s1.values <= s0.values).mean() > 0.99


def test_refusals(wiring):
    with pytest.raises(NotImplementedError):
        connectome_MBC1_weights('malecns-left')
    with pytest.raises(ValueError):
        connectome_MBC1_weights('not-a-connectome')
    with pytest.raises(ValueError):
        connectome_MBC1_weights(prat_claws=True)
    with pytest.raises(ValueError):
        connectome_MBC1_weights(prat_boutons=True)
    with pytest.raises(ValueError):
        connectome_MBC1_weights(wPNKC=wiring)
    with pytest.raises(ValueError):
        connectome_MBC1_weights(dropped_counts=dict())
    with pytest.raises(ValueError):
        connectome_MBC1_KCKC_weights([1, 2, 3])
    with pytest.raises(ValueError):
        connectome_MBC1_KCKC_weights([1, 2], wPNKC=wiring)
    with pytest.raises(ValueError):
        connectome_MBC1_KCKC_weights(claws_only=True)
    with pytest.raises(NotImplementedError):
        connectome_MBC1_KCKC_weights(connectome='malecns-right')
