"""The effect of a manipulation on odor-odor distances, pair by pair, and how well a model
reproduces a measured one.

The effect is, for each odor pair, the distance in the intact condition minus the distance
under the manipulation (e.g. with APL on minus with APL output blocked: the APL effect).
`effect_stats` compares a model's per-pair effect with the data's (the root mean squared
error, split exactly into its mean part and its pattern part, and the correlations of the
pattern), `effect_decomposition` splits the pattern part further into what each odor
contributes additively and a pair-specific rest, and `effect_by_class` splits each
condition's similarities, and so the effect, by the units' odors-per-KC class.

Distances are pair Series (`hong2p.rdm`), aligned in canonical order (`order`). The data's
effect is an argument. Moved from the KC-remapping experiment queue
(model_experiment_queue's kcq.harness, where blocking APL output is the manipulation),
2026-11-12.
"""
from typing import Dict, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd

from hong2p.rdm import PAIR_NAMES, align_pairs, canonical_pairs, odor_shares, pair_design
from hong2p.rdm import pair_index, pair_list
from hong2p.stats import pearson, spearman

from pyolfsysm.kc_stats import (OPK_EDGES, class_columns, class_cosine_terms_array,
    opk_class_labels, opk_class_shares, ordered_matrix, responders
)


def pair_effect(intact: pd.Series, manipulated: pd.Series, order: Optional[Sequence] = None,
    name: str = 'effect') -> pd.Series:
    """Returns the per-pair effect of a manipulation on distances: the intact distance
    minus the manipulated distance of each pair (canonical pair order; pairs missing from
    either side dropped)."""
    a, b = align_pairs(canonical_pairs(intact, order), canonical_pairs(manipulated, order))
    return (a - b).rename(name)


def effect_stats(intact: pd.Series, manipulated: pd.Series, data: pd.Series,
    order: Optional[Sequence] = None, prefix: str = 'effect_') -> Dict[str, float]:
    """Returns how well a model's per-pair effect (`pair_effect`) matches the data's, over
    the pairs both have.

    Args:
        intact, manipulated: the model's pair distances in the two conditions.
        data: the data's per-pair effect.
        order: canonical order of the odors (`hong2p.rdm.order_items`).
        prefix: of the keys.

    Keys (after `prefix`): 'rmse' (the root mean squared difference between the model's
    and the data's per-pair effect; lower is better), 'rmse_centered' (the same with each
    side's mean removed, the pattern part: rmse^2 = mean_error^2 + rmse_centered^2),
    'mean' (the model's mean effect), 'mean_error' (model minus data), 'spearman' and
    'pearson' (the per-pair pattern), 'n_pairs'.
    """
    m, d = align_pairs(pair_effect(intact, manipulated, order), data)
    err = m - d
    mean_error = float(err.mean())
    return {f'{prefix}rmse': float(np.sqrt(np.mean(err ** 2))),
        f'{prefix}rmse_centered': float(np.sqrt(np.mean((err - mean_error) ** 2))),
        f'{prefix}mean': float(m.mean()), f'{prefix}mean_error': mean_error,
        f'{prefix}spearman': spearman(m, d), f'{prefix}pearson': pearson(m, d),
        f'{prefix}n_pairs': int(len(m))
    }


def effect_pair_parts(intact: pd.Series, manipulated: pd.Series, data: pd.Series,
    order: Optional[Sequence] = None) -> pd.DataFrame:
    """Returns, per pair, the parts of the effect's error (`effect_decomposition`).

    Columns 'model' and 'data' (each side's per-pair effect), 'error' (model minus data,
    each with its mean removed: the pattern part's error), 'per_odor' (its projection onto
    the additive space over odors, `hong2p.rdm.pair_design`) and 'pair_specific' (the
    residual; error = per_odor + pair_specific). Indexed by the pairs both sides have.
    """
    m, d = align_pairs(pair_effect(intact, manipulated, order), data)
    err = (m - m.mean()) - (d - d.mean())
    X, _ = pair_design(m.index)
    coef, *_ = np.linalg.lstsq(X, err.to_numpy(dtype=float), rcond=None)
    proj = X @ coef
    return pd.DataFrame({'model': m, 'data': d, 'error': err, 'per_odor': proj,
        'pair_specific': err.to_numpy(dtype=float) - proj}, index=m.index
    )


def effect_decomposition(intact: pd.Series, manipulated: pd.Series, data: pd.Series,
    order: Optional[Sequence] = None, prefix: str = 'effect_'
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Returns the effect's error (`effect_stats`) split into three parts whose squares
    add, rmse**2 = mean_error**2 + per_odor**2 + pair_specific**2, and the odors' shares.

    The pattern part's error (model minus data, each with its mean removed) is projected
    onto the additive space over odors (a pair's value the sum of its two odors' shares,
    `hong2p.rdm.odor_shares`): the projection's root mean square is the per-odor part, the
    residual's the pair-specific part.

    Returns (dict with the `effect_stats` keys plus 'per_odor', 'pair_specific',
    'share_pearson' (the model's per-odor shares against the data's),
    'model_additive_r2' and 'data_additive_r2' (variance over pairs each side's additive
    model explains), all after `prefix`; DataFrame indexed by odor with the model's and
    the data's shares and their difference).
    """
    stats = effect_stats(intact, manipulated, data, order, prefix)
    parts = effect_pair_parts(intact, manipulated, data, order)
    sm, r2m = odor_shares(parts['model'])
    sd, r2d = odor_shares(parts['data'])
    shares = pd.DataFrame({'model_share': sm, 'data_share': sd.reindex(sm.index)})
    shares['share_error'] = shares.model_share - shares.data_share
    stats.update({f'{prefix}per_odor': float(np.sqrt(np.mean(parts['per_odor'] ** 2))),
        f'{prefix}pair_specific': float(np.sqrt(np.mean(parts['pair_specific'] ** 2))),
        f'{prefix}share_pearson': float(np.corrcoef(shares.model_share,
            shares.data_share)[0, 1]),
        f'{prefix}model_additive_r2': r2m, f'{prefix}data_additive_r2': r2d
    })
    return stats, shares


def effect_by_class(intact, manipulated, *, resp_intact=None, resp_manipulated=None,
    thresh_intact: float = 1.0, thresh_manipulated: float = 1.0, edges=OPK_EDGES,
    odors: Optional[Sequence] = None, order: Optional[Sequence] = None,
    names: Sequence[str] = PAIR_NAMES) -> pd.DataFrame:
    """Returns the per-pair effect split by odors-per-KC class.

    Each condition's `kc_stats.class_cosine_terms` (classes set by that condition's
    responses), averaged over its recordings, and the effect, manipulated similarity
    minus intact similarity (= intact distance minus manipulated distance, as
    `pair_effect`), per class and term. Class 'all' term 'total' of the effect is the
    `pair_effect` of the two conditions' cosine distances (mean over recordings).

    Args:
        intact, manipulated: each condition's (units x odors) values, or a list of them
            (recordings; averaged with equal weight, NaN pairs of a recording skipped).
        resp_intact, resp_manipulated: each condition's responders (one per matrix, as
            `class_cosine_terms`' `resp`), or None for values above `thresh_intact` /
            `thresh_manipulated`.
        edges, odors, order: as `kc_stats.class_cosine_terms`.

    Returns DataFrame indexed by canonical pairs, columns (block, class, term), block 'on'
    (intact), 'off' (manipulated) (the similarity contributions) and 'effect' (off minus
    on); attrs['class_shares'] {'on', 'off'}: the share of units in each class (mean over
    recordings).
    """
    blocks, shares = {}, {}
    for cond, mats, resps, thresh in (('on', intact, resp_intact, thresh_intact),
            ('off', manipulated, resp_manipulated, thresh_manipulated)):
        if isinstance(mats, (list, tuple)):
            resps = [None] * len(mats) if resps is None else list(resps)
            if len(resps) != len(mats):
                raise ValueError(f'{cond}: {len(mats)} matrices but {len(resps)} responder '
                    'matrices'
                )
        else:
            mats, resps = [mats], [resps]
        arrs, sh, srt = [], [], None
        for m, r in zip(mats, resps):
            X, o, idx = ordered_matrix(m, odors, order)
            if srt is not None and o != srt:
                raise ValueError('the matrices have different odors')
            srt = o
            R = responders(X, r, idx, o, thresh)
            arrs.append(class_cosine_terms_array(X, R, edges))
            sh.append(opk_class_shares(R, edges))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            a = np.nanmean(np.stack(arrs), axis=0) if len(arrs) > 1 else arrs[0]
        pl = pair_list(srt, order)
        blocks[cond] = pd.DataFrame(a.reshape(len(pl), -1), index=pair_index(pl, names),
            columns=class_columns(edges)
        )
        shares[cond] = pd.Series(np.mean(sh, axis=0), index=opk_class_labels(edges))
    if not blocks['on'].index.equals(blocks['off'].index):
        raise ValueError('the two conditions have different odors')
    blocks['effect'] = blocks['off'] - blocks['on']
    out = pd.concat(blocks, axis=1, names=['block'])
    out.attrs['class_shares'] = shares
    return out
