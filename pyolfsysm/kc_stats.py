"""Statistics of a KC population's responses: how concentrated each odor's response is
across KCs, the number of odors each KC responds to (odors-per-KC classes), and each odor
pair's cosine similarity split by responders and by those classes.

Inputs are (units x odors) matrices: a model's spike counts or a readout of them
(`pyolfsysm.readout`), or a recording's trial-mean responses. Odors are labelled by the
matrix's columns (or `odors` for arrays) and ordered by `order`
(`hong2p.rdm.order_items`).

Moved from the KC-remapping experiment queue (model_experiment_queue's kcq.harness),
2026-11-12.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from hong2p.rdm import PAIR_NAMES, order_items, pair_index, pair_list


def amplitude_stats(mat: pd.DataFrame) -> pd.DataFrame:
    """Returns, per odor, how concentrated its response vector is across units.

    Args:
        mat: rows units, columns odors (a model's counts or readout, or a recording's
            trial means).

    Columns: 'top1_share' / 'top5_share', the share of the odor's squared response vector
    (the quantity a cosine distance weighs) carried by the top 1 % / 5 % of units;
    'participation_ratio', (sum x^2)^2 / (n sum x^4), the effective share of units
    carrying it (1 when all units carry equal squared amplitude, 1 / n when one unit
    carries all); 'n_units'.
    """
    x = mat.astype(float).to_numpy()
    n = x.shape[0]
    sq = x ** 2
    tot = sq.sum(axis=0)
    srt = -np.sort(-sq, axis=0)
    k1, k5 = max(1, int(np.ceil(0.01 * n))), max(1, int(np.ceil(0.05 * n)))
    with np.errstate(invalid='ignore', divide='ignore'):
        out = pd.DataFrame(dict(top1_share=srt[:k1].sum(axis=0) / tot,
            top5_share=srt[:k5].sum(axis=0) / tot,
            participation_ratio=tot ** 2 / (n * (sq ** 2).sum(axis=0)), n_units=n),
            index=mat.columns
        )
    return out


def responder_cv(values: np.ndarray) -> float:
    """Returns the coefficient of variation (sd / mean) of responders' amplitudes, pooled
    (non-finite values dropped; NaN if none or a mean <= 0)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.std() / v.mean()) if len(v) and v.mean() > 0 else np.nan


# odors-per-KC classes: (lo, hi) inclusive numbers of odors responded to, hi None for no
# upper limit
OPK_EDGES = ((0, 0), (1, 1), (2, 3), (4, 7), (8, None))
OPK_CLASSES = ('0', '1', '2_3', '4_7', '8plus')
# the parts of a cosine similarity by unit: both entries responders, one, neither, total
CLASS_TERMS = ('both', 'one', 'neither', 'total')


def poisson_binomial_bins(P: np.ndarray, edges=OPK_EDGES) -> np.ndarray:
    """Returns the expected distribution of the number of odors a unit responds to, in bins.

    Each odor is responded to independently with its probability (P: units x odors); the
    distribution's mean over units is summed into `edges` bins ((lo, hi) inclusive, hi None
    for no upper limit).
    """
    P = np.asarray(P, dtype=float)
    n_u, n_o = P.shape
    dist = np.zeros((n_u, n_o + 1))
    dist[:, 0] = 1.0
    for j in range(n_o):
        p = P[:, j:j + 1]
        dist[:, 1:] = dist[:, 1:] * (1 - p) + dist[:, :-1] * p
        dist[:, 0] = dist[:, 0] * (1 - p[:, 0])
    mean = dist.mean(axis=0)
    return np.array([mean[lo:(n_o + 1 if hi is None else hi + 1)].sum() for lo, hi in edges])


def opk_class_labels(edges=OPK_EDGES) -> List[str]:
    """Returns labels of odors-per-KC classes: '0', '2_3', '8plus', ... (`OPK_CLASSES` for
    `OPK_EDGES`)."""
    return [f'{lo}' if hi == lo else (f'{lo}plus' if hi is None else f'{lo}_{hi}')
        for lo, hi in edges
    ]


def opk_class_shares(resp, edges=OPK_EDGES) -> np.ndarray:
    """Returns the share of units in each odors-per-KC class of a (units x odors) bool
    responder matrix (DataFrame or array)."""
    R = np.asarray(resp.to_numpy() if isinstance(resp, pd.DataFrame) else resp, dtype=bool)
    n_kc = R.sum(axis=1)
    return np.array([np.mean((n_kc >= lo) & (n_kc <= (R.shape[1] if hi is None else hi)))
        for lo, hi in edges
    ])


def ordered_matrix(mat, odors: Optional[Sequence] = None,
    order: Optional[Sequence] = None) -> Tuple[np.ndarray, List, List[int]]:
    """Returns (the units x odors float array with its columns in canonical order, that
    order, the column positions taken) of a DataFrame (odors its columns) or an array
    whose columns are `odors`."""
    if isinstance(mat, pd.DataFrame):
        labels = list(mat.columns)
        X = mat.to_numpy(dtype=float)
    else:
        X = np.asarray(mat, dtype=float)
        if odors is None:
            raise ValueError('an array needs its odors')
        labels = list(odors)
    if X.ndim != 2 or X.shape[1] != len(labels) or len(set(labels)) != len(labels):
        raise ValueError(f'a (units x odors) matrix with {len(labels)} distinct odor '
            f'columns, not {X.shape}'
        )
    srt = order_items(labels, order)
    idx = [labels.index(o) for o in srt]
    return X[:, idx], srt, idx


def class_cosine_terms_array(X: np.ndarray, R: np.ndarray, edges=OPK_EDGES) -> np.ndarray:
    """Returns the core of `class_cosine_terms` on arrays.

    Args:
        X: (units x odors) values, columns in the order the pairs are wanted in (pairs
            (i, j), i < j, row-major).
        R: (units x odors) bool responders.

    Returns (pairs, classes + 1, 4): per pair, each class's contribution to the cosine
    similarity (the last class all units), split into the terms `CLASS_TERMS`.
    """
    X = np.asarray(X, dtype=float)
    R = np.asarray(R, dtype=bool)
    n_kc = R.sum(axis=1)
    XR = np.where(R, X, 0.0)
    XS = X - XR
    norms = np.sqrt((X ** 2).sum(axis=0))
    iu = np.triu_indices(X.shape[1], 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        den = np.outer(norms, norms)[iu]
    masks = [(n_kc >= lo) & (n_kc <= (X.shape[1] if hi is None else hi)) for lo, hi in edges]
    masks.append(np.ones(X.shape[0], dtype=bool))
    out = np.empty((len(iu[0]), len(masks), len(CLASS_TERMS)))
    for c, m in enumerate(masks):
        r, s = XR[m], XS[m]
        rs = r.T @ s
        with np.errstate(invalid='ignore', divide='ignore'):
            out[:, c, 0] = (r.T @ r)[iu] / den
            out[:, c, 1] = (rs + rs.T)[iu] / den
            out[:, c, 2] = (s.T @ s)[iu] / den
    out[:, :, 3] = out[:, :, :3].sum(axis=2)
    return out


def class_columns(edges=OPK_EDGES) -> pd.MultiIndex:
    """Returns the (class, term) columns of `class_cosine_terms`."""
    return pd.MultiIndex.from_product([opk_class_labels(edges) + ['all'], list(CLASS_TERMS)],
        names=['class', 'term']
    )


def responders(X: np.ndarray, resp, idx: List[int], order: List, thresh: float
    ) -> np.ndarray:
    """Returns the responders of an `ordered_matrix` X: X > thresh, or `resp` in X's column
    order (a DataFrame by its odor labels, an array by the positions `idx` of the matrix
    it belongs to)."""
    if resp is None:
        return X > thresh
    if isinstance(resp, pd.DataFrame):
        R = resp[order].to_numpy(dtype=bool)
    else:
        R = np.asarray(resp, dtype=bool)[:, idx]
    if R.shape != X.shape:
        raise ValueError(f'responders {R.shape} for a matrix {X.shape}')
    return R


def class_cosine_terms(mat, resp=None, *, thresh: float = 1.0, edges=OPK_EDGES,
    odors: Optional[Sequence] = None, order: Optional[Sequence] = None,
    names: Sequence[str] = PAIR_NAMES) -> pd.DataFrame:
    """Returns each odor pair's cosine similarity across units split by the units'
    odors-per-KC class and, within a class, into three terms: both entries responders
    ('both'), one a responder and the other not ('one'), neither ('neither'), with their
    sum ('total').

    Each part is that part of the dot product over the product of the two odors' lengths
    over all units, so the classes' totals add up to the cosine (class 'all', computed
    over all units).

    Args:
        mat: (units x odors) values: a DataFrame (odors its columns) or an array whose
            columns are `odors`.
        resp: (units x odors) responders, in `mat`'s column order (a bool array or
            DataFrame); default `mat > thresh` (e.g. a model read out as n / (n + K) whose
            responders are its two-spike entries passes resp=counts >= 2).
        thresh: the response criterion when `resp` is None.
        edges: the classes, (lo, hi) inclusive numbers of odors responded to (hi None: no
            upper limit).
        order: canonical order of the odors (`hong2p.rdm.order_items`).

    Returns DataFrame indexed by canonical pairs, columns (class, term)
    (`opk_class_labels(edges)` + 'all' x `CLASS_TERMS`); NaN for pairs with an odor of
    zero length.
    """
    X, srt, idx = ordered_matrix(mat, odors, order)
    R = responders(X, resp, idx, srt, thresh)
    arr = class_cosine_terms_array(X, R, edges)
    pl = pair_list(srt, order)
    return pd.DataFrame(arr.reshape(len(pl), -1), index=pair_index(pl, names),
        columns=class_columns(edges)
    )


def cosine_terms(mat: pd.DataFrame, thresh: float = 1.0, order: Optional[Sequence] = None,
    *, names: Sequence[str] = PAIR_NAMES) -> pd.DataFrame:
    """Returns each odor pair's cosine similarity across units split into three terms by
    unit: both entries responders (above `thresh`), one a responder and the other not,
    neither.

    Each term is that part of the dot product divided by the product of the two odors'
    lengths, so the three add up to the cosine. Columns 'both', 'one', 'neither',
    'cosine'; index canonical pairs.
    """
    m = mat.astype(float)
    odors = order_items(m.columns, order)
    X = m[odors].to_numpy()
    R = (X > thresh).astype(float)
    S = 1.0 - R
    norms = np.linalg.norm(X, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        den = np.outer(norms, norms)
        both = ((X * R).T @ (X * R)) / den
        one = ((X * R).T @ (X * S) + (X * S).T @ (X * R)) / den
        neither = ((X * S).T @ (X * S)) / den
    pl = pair_list(odors, order)
    idx = {o: i for i, o in enumerate(odors)}
    rows = [(both[idx[a], idx[b]], one[idx[a], idx[b]], neither[idx[a], idx[b]])
        for a, b in pl
    ]
    out = pd.DataFrame(rows, columns=['both', 'one', 'neither'], index=pair_index(pl, names))
    out['cosine'] = out.sum(axis=1)
    return out
