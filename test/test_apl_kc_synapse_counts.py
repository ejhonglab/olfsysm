#!/usr/bin/env python3
"""Tests of `connectome_APL_KC_synapse_counts` and `connectome_APL_weights(dropped_counts=...)`.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from pyolfsysm.mb_model import (connectome_APL_KC_synapse_counts, connectome_APL_weights,
    connectome_wPNKC, APL_KC_COUNT_COLUMNS, KC_ID, CLAW_ID, KC_TYPE
)


PRAT_KWS = dict(prat_claws=True, prat_boutons=True)


@lru_cache(maxsize=None)
def _wPNKC() -> pd.DataFrame:
    return connectome_wPNKC(**PRAT_KWS)


def _synapse(kc, claw, roi='CA(R)', post_roi=None, direction='APL>KC', apl='APL_R'):
    pre, post = (0, kc) if direction == 'APL>KC' else (kc, 0)
    inst = dict(instance_pre=apl, instance_post='KCab-m_R') if direction == 'APL>KC' else (
        dict(instance_pre='KCab-m_R', instance_post=apl))
    return dict(bodyId_pre=pre, bodyId_post=post, roi_pre=roi,
        roi_post=roi if post_roi is None else post_roi, anatomical_claw_corrected=claw,
        **inst
    )


@pytest.mark.parametrize('direction', ['APL>KC', 'KC>APL'])
def test_synapse_counts_made_up(direction):
    """A made-up wiring and table: a kept claw, a dropped claw, no claw (-1 and no
    claw ID), a -1 claw the wiring has, the peduncle, other ROIs, another APL, a KC with
    none and a KC outside the wiring."""
    index = pd.MultiIndex.from_tuples([(1, 0, 'ab'), (1, 1, 'ab'), (2, -1, 'g'),
        (2, 3, 'g'), (3, 0, 'ab')], names=[KC_ID, CLAW_ID, KC_TYPE]
    )
    s = lambda *a, **k: _synapse(*a, direction=direction, **k)
    rows = [
        s(1, 0), s(1, 0), s(1, 1),             # kept on claws 0 and 1
        s(1, 5),                               # a claw the wiring drops
        s(1, -1), s(1, np.nan),                # no claw
        s(1, np.nan, roi='PED(R)'), s(1, np.nan, roi='PED(R)'),
        s(1, np.nan, roi='PED(R)', post_roi='aL(R)'),   # not both sites in the peduncle
        s(1, 0, roi='CA(R)', post_roi='SLP(R)'),        # not both sites in the calyx
        s(1, 0, apl='APL fragment?'),          # another APL
        s(2, -1), s(2, -1), s(2, 3),           # -1 kept on the wiring's claw -1
        s(9, 0),                               # a KC outside the wiring
    ]
    per_kc, per_claw = connectome_APL_KC_synapse_counts(index, direction,
        synapses=pd.DataFrame(rows)
    )
    assert list(per_kc.columns) == list(APL_KC_COUNT_COLUMNS)
    assert list(per_kc.index) == [1, 2, 3]
    expected = pd.DataFrame({
        'n_claws': [2, 2, 1], 'kept': [3, 3, 0], 'kept_unassigned': [0, 2, 0],
        'dropped_claw': [1, 0, 0], 'no_claw': [2, 0, 0], 'no_claw_nan': [1, 0, 0],
        'peduncle': [2, 0, 0]}, index=per_kc.index
    )
    pd.testing.assert_frame_equal(per_kc, expected, check_dtype=False)
    assert per_claw.index.equals(index)
    assert per_claw.tolist() == [2.0, 1.0, 2.0, 1.0, 0.0]


def test_synapse_counts_duplicate_claw_rows():
    """A (KC, claw) with two rows of the wiring has its count split over them."""
    index = pd.MultiIndex.from_tuples([(1, 0, 'a'), (1, 0, 'b'), (1, 1, 'a')],
        names=[KC_ID, CLAW_ID, 'other']
    )
    rows = [_synapse(1, 0), _synapse(1, 0), _synapse(1, 0), _synapse(1, 1)]
    per_kc, per_claw = connectome_APL_KC_synapse_counts(index, synapses=pd.DataFrame(rows))
    assert per_claw.tolist() == [1.5, 1.5, 1.0]
    assert per_kc.loc[1, 'kept'] == 4
    assert per_kc.loc[1, 'n_claws'] == 3


def test_synapse_counts_bad_direction():
    with pytest.raises(ValueError):
        connectome_APL_KC_synapse_counts(_wPNKC().index, 'APL>PN')


def test_dropped_counts_needs_prat_claws():
    with pytest.raises(NotImplementedError):
        connectome_APL_weights(dropped_counts={})


def test_synapse_counts_match_loader():
    """On the model's claw wiring: the kept counts are `connectome_APL_weights`' weights
    exactly, what it drops (`dropped_counts`) is the counts' 'no_claw' and 'dropped_claw',
    and passing `dropped_counts` changes nothing it returns."""
    wPNKC = _wPNKC()
    plain = connectome_APL_weights(wPNKC=wPNKC, **PRAT_KWS)
    dropped = {}
    weights = connectome_APL_weights(wPNKC=wPNKC, dropped_counts=dropped, **PRAT_KWS)
    for w1, w2 in zip(plain, weights):
        assert w1.equals(w2)
    assert set(dropped) == {'wAPLKC', 'wKCAPL'}

    for name, direction, w in (('wAPLKC', 'APL>KC', weights[0]),
            ('wKCAPL', 'KC>APL', weights[1])):

        per_kc, per_claw = connectome_APL_KC_synapse_counts(wPNKC.index, direction)
        assert per_claw.index.equals(w.index)
        assert np.array_equal(per_claw.to_numpy(), w.to_numpy(dtype=float)), name

        d = dropped[name]
        assert d.index.equals(per_kc.index)
        assert list(d.columns) == ['nan_claw', 'unassigned_claw', 'other_claw']
        assert (d.nan_claw == per_kc.no_claw_nan).all(), name
        assert (d.nan_claw + d.unassigned_claw == per_kc.no_claw).all(), name
        assert (d.other_claw == per_kc.dropped_claw).all(), name

        kept_per_kc = w.groupby(level=KC_ID).sum().reindex(per_kc.index)
        assert np.allclose(kept_per_kc.to_numpy(), per_kc.kept.to_numpy()), name
        assert (per_kc.n_claws == wPNKC.groupby(level=KC_ID).size().reindex(
            per_kc.index)).all()
        assert (per_kc.kept_unassigned <= per_kc.kept).all()
        assert (per_kc.no_claw_nan <= per_kc.no_claw).all()
        assert per_kc.peduncle.sum() > 0
