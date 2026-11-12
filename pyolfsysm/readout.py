"""The model's KC output read as imaging sees it.

The model's KCs are compared with calcium imaging of real KCs, whose response to an odor
is a fluorescence change (often z-scored), not a spike count. These read a model's (KCs x
odors) spike counts, or the times of its spikes, the way the recordings are made:
- elementwise readouts of the counts (`readout`: compressive maps, a binarized response,
  a rank-matched amplitude), and the parameter of a one-parameter readout calibrated on
  the APL-on counts alone (`calibrate_readout`), so that what changes with a manipulation
  (APL off) stays a prediction;
- spikes weighted by the area of the indicator kernel they start inside the response
  window (`kernel_window_weights`, `kernel_weighted_counts`);
- measurement noise: its expected effect on cosine distances, and pseudo-recordings
  sampled through it (`expected_pair_distances_noisy`, `sample_pseudo_recordings`);
- a z-score of each KC over its whole session, as some recordings are normalized
  (`session_zscore`, its distances and statistics, `calibrate_session_zscore`);
- the odor-odor distances and response fractions of a model output (`model_rdms`,
  `response_fractions`).

Targets (a participation ratio, a response rate) are arguments, and odors are ordered by
`order` (`hong2p.rdm.order_items`). Moved from the KC-remapping experiment queue
(model_experiment_queue's kcq.harness), 2026-11-12.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from hong2p.rdm import PAIR_NAMES, order_items, pair_distances, pair_index, pair_list

from pyolfsysm.kc_stats import amplitude_stats, poisson_binomial_bins, responder_cv


# ---------------------------------------------------------------------------
# readouts of the spike counts
READOUT_KINDS = ('raw', 'sqrt', 'power', 'saturating', 'log1p', 'binary', 'rank_match')


def readout(counts: pd.DataFrame, kind: str = 'raw', param=None, floor: float = 0.0
    ) -> pd.DataFrame:
    """Returns a readout of a model's spike counts (rows KCs, columns odors), elementwise.

    Args:
        counts: spike counts.
        kind: 'raw' (the counts); 'sqrt' (n ** 0.5); 'power' (n ** param,
            0 < param <= 1); 'saturating' (n / (n + param), half-saturation at `param`
            spikes); 'log1p' (log(1 + n / param)); 'binary' (1 where n >= param, default
            2, i.e. the model's `responses`); 'rank_match' (each count mapped through the
            lookup `param` from `rank_match_map`, a monotone count -> amplitude map).
        param: the readout's parameter (see `kind`); None for 'raw' and 'sqrt'.
        floor: 'saturating' only: the one-spike floor b of the readout with a low end,
            (n + b) / (n + b + param) for n >= 1 and 0 for n = 0, so b raises the weight
            of the weakest responses (b = -1 reads one spike as 0). The default 0 is the
            plain saturating readout.

    Every readout keeps 0 at 0 and is monotone in the count, so which KCs respond, and the
    response rates, are the same under every readout.
    """
    x = counts.astype(float)
    if floor and kind != 'saturating':
        raise ValueError(f'floor is defined for the saturating readout only, not {kind!r}')
    if kind == 'raw':
        return x
    if kind == 'sqrt':
        return x ** 0.5
    if kind == 'power':
        return x ** float(param)
    if kind == 'saturating':
        if not floor:
            return x / (x + float(param))
        b = float(floor)
        if b < -1:
            raise ValueError('floor must be >= -1 (one spike reads (1 + b) / (1 + b + K)), '
                f'not {b}'
            )
        with np.errstate(invalid='ignore', divide='ignore'):
            return ((x + b) / (x + b + float(param))).where(x > 0, 0.0)
    if kind == 'log1p':
        return np.log1p(x / float(param))
    if kind == 'binary':
        return (x >= (2.0 if param is None else float(param))).astype(float)
    if kind == 'rank_match':
        lut = param
        vals = np.interp(x.to_numpy(), lut.index.to_numpy(dtype=float),
            lut.to_numpy(dtype=float)
        )
        return pd.DataFrame(np.where(x.to_numpy() > 0, vals, 0.0), index=x.index,
            columns=x.columns
        )
    raise ValueError(f'unknown readout {kind!r} (one of {READOUT_KINDS})')


def readout_pair_distances(counts: pd.DataFrame, kind: str = 'raw', param=None,
    floor: float = 0.0, order: Optional[Sequence] = None) -> pd.Series:
    """Returns the cosine odor-odor distances across KCs after `readout` ('raw' gives the
    distances of the spike counts themselves), a canonical pair Series."""
    return pair_distances(readout(counts, kind, param, floor=floor), measure='cosine',
        order=order
    )


def readout_stat(on_counts: pd.DataFrame, kind: str, param,
    stat: str = 'participation_ratio', responders_min: Optional[float] = None,
    floor: float = 0.0) -> float:
    """Returns a concentration statistic of the readout of `on_counts`.

    Args:
        stat: an `amplitude_stats` column (its mean over odors), or 'responder_cv'
            (`responder_cv` of the readout among responders).
        responders_min: zero the readout of entries below this many spikes first (e.g.
            2, the model's response criterion), to compare with data statistics on
            responders only; 'responder_cv' always uses responders (default 2 spikes).
        floor: the saturating readout's one-spike floor (`readout`).
    """
    r = readout(on_counts, kind, param, floor=floor)
    if stat == 'responder_cv':
        thr = 2.0 if responders_min is None else responders_min
        return responder_cv(r.to_numpy()[on_counts.to_numpy() >= thr])
    if responders_min is not None:
        r = r.where(on_counts >= responders_min, 0.0)
    return float(amplitude_stats(r)[stat].mean())


def calibrate_readout(on_counts: pd.DataFrame, kind: str, target: float,
    stat: str = 'participation_ratio', bounds: Tuple[float, float] = (1e-3, 1e3),
    iters: int = 60, responders_min: Optional[float] = None, floor: float = 0.0) -> float:
    """Returns the parameter of a one-parameter readout ('power', 'saturating' or 'log1p')
    at which `readout_stat` of the APL-on counts equals `target`.

    E.g. `target` the control flies' participation ratio. Bisection on a log scale. The
    parameter is fixed on the APL-on counts alone and then used unchanged for APL off, so
    the effect of APL is not fitted. Returns the bound nearest to the target when it is not
    bracketed (for 'power' the upper bound is 1, raw counts). `floor`: the saturating
    readout's one-spike floor, held fixed (`readout`).
    """
    if kind == 'power':
        bounds = (max(bounds[0], 1e-3), 1.0)
    f = lambda p: readout_stat(on_counts, kind, p, stat, responders_min, floor)
    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    flo, fhi = f(np.exp(lo)) - target, f(np.exp(hi)) - target
    if flo * fhi > 0:
        return float(np.exp(lo) if abs(flo) < abs(fhi) else np.exp(hi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(np.exp(mid)) - target
        if fm * flo <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return float(np.exp(0.5 * (lo + hi)))


def rank_match_map(on_counts: pd.DataFrame, reference: np.ndarray) -> pd.Series:
    """Returns a monotone count -> amplitude lookup for `readout(..., 'rank_match', lut)`.

    Each positive count of the APL-on matrix is placed at its mid-rank quantile among the
    positive counts and given the amplitude at that quantile of `reference` (e.g. the
    recordings' trial-mean amplitudes among responders); counts above the APL-on maximum
    keep the top amplitude (np.interp clamps), 0 stays 0. Fixed on APL on, used unchanged
    for APL off.
    """
    c = on_counts.astype(float).to_numpy().ravel()
    c = c[c > 0]
    vals, cnt = np.unique(c, return_counts=True)
    q = (np.cumsum(cnt) - 0.5 * cnt) / cnt.sum()
    ref = np.sort(np.asarray(reference, dtype=float))
    amp = np.quantile(ref, q)
    return pd.Series(amp, index=vals, name='amplitude')


# ---------------------------------------------------------------------------
# spikes read through the indicator kernel: a spike at time s counts by the area inside the
# response window of the kernel (1 - exp(-(t - s) / rise)) exp(-(t - s) / decay), t > s
def kernel_window_weights(times, rise: float, decay: float,
    window: Tuple[float, float] = (2.0, 8.1), ref_time: Optional[float] = None
    ) -> np.ndarray:
    """Returns each spike's weight in a response window: the area inside `window` of the
    kernel (1 - exp(-u / rise)) exp(-u / decay) (u the time since the spike), divided by
    the area of a spike at `ref_time`.

    Args:
        times: spike times in seconds from the odor's onset (array-like).
        rise, decay: the kernel's time constants in seconds; rise 0 (an instant rise) and
            decay np.inf (no decay) are the limits, so rise 0 with decay inf is a step and
            every spike before the window has weight exactly 1 (with `ref_time` before the
            window too).
        window: the response window (start, end) in seconds from onset (e.g. the mean
            2.0 to 8.1 s after onset minus the pre-onset mean).
        ref_time: the spike time of weight 1; default the window's start. Must be before
            the window's end.

    A spike inside the window counts by the part of its kernel after it, one after the
    window by 0. The baseline (before onset) holds no kernel of a spike after onset, so it
    is not subtracted; spikes before onset get the same window area with no baseline
    correction. With a finite rise a spike shortly before the window's start can weigh a
    little more than one at it (whose kernel is still rising when the window opens): at
    most 1.8 % for rise 1 s, decay 5 s. Returns float weights, one per spike.
    """
    t = np.asarray(times, dtype=float)
    w0, w1 = float(window[0]), float(window[1])
    rise, decay = float(rise), float(decay)
    if not w1 > w0:
        raise ValueError(f'window must have end > start, not {window}')
    if rise < 0 or not decay > 0:
        raise ValueError(f'rise must be >= 0 and decay > 0, not {rise}, {decay}')
    ref = w0 if ref_time is None else float(ref_time)
    if not ref < w1:
        raise ValueError(f'ref_time ({ref}) must be before the window end ({w1})')

    def area(s: np.ndarray) -> np.ndarray:
        # u from lo to hi: the window in time since the spike, clipped at the spike
        lo, hi = np.maximum(w0 - s, 0.0), np.maximum(w1 - s, 0.0)
        if np.isinf(decay):
            # (the window's length after the spike, computed so that it is the same float
            # for every spike before the window: w1 - w0 exactly)
            first = np.maximum(w1, s) - np.maximum(w0, s)
            tau = rise
        else:
            # decay * (exp(-lo / decay) - exp(-hi / decay)), accurate for long decays
            first = -decay * np.exp(-lo / decay) * np.expm1(-(hi - lo) / decay)
            tau = rise * decay / (rise + decay)
        if tau == 0:
            return first
        return first - tau * (np.exp(-lo / tau) - np.exp(-hi / tau))

    a_ref = float(area(np.array([ref]))[0])
    if not a_ref > 0:
        raise ValueError(f'a spike at ref_time {ref} has no kernel area inside the window '
            f'{window}'
        )
    return area(t) / a_ref


def kernel_weighted_counts(spike_times: pd.DataFrame, index: pd.Index, columns: pd.Index,
    rise: float, decay: float, window: Tuple[float, float] = (2.0, 8.1),
    ref_time: Optional[float] = None, stim_onset: float = 0.0) -> pd.DataFrame:
    """Returns a model's spike counts read through the indicator kernel: KCs x odors sums of
    each spike's `kernel_window_weights`.

    Args:
        spike_times: one row per spike, as `mb_model.simulate(..., spike_times=True)`
            returns: 'kc' and 'odor' (the positions of the spike's KC and odor in the
            spike-count matrix), 'time_s' (seconds in olfsysm's time, where the odor's
            onset is 0); other columns are ignored.
        index, columns: the spike-count matrix's KC index and odor columns (the rows of
            'kc' and 'odor' refer to them by position).
        rise, decay, window, ref_time: the kernel and the window (`kernel_window_weights`).
        stim_onset: the odor's onset in 'time_s' units (the window and `ref_time` are
            relative to it).

    Returns a float DataFrame (`index` x `columns`); with every weight 1 (e.g. rise 0 and
    decay inf with every spike before the window) it equals the spike counts exactly. The
    responses (two or more spikes over the whole simulation) are not affected: only the
    amplitudes are.
    """
    n_kc, n_odor = len(index), len(columns)
    kc = spike_times['kc'].to_numpy(dtype=np.int64)
    od = spike_times['odor'].to_numpy(dtype=np.int64)
    if len(kc) and (kc.min() < 0 or kc.max() >= n_kc or od.min() < 0 or
            od.max() >= n_odor):
        raise ValueError(f'spike positions outside a {n_kc} x {n_odor} matrix')
    w = kernel_window_weights(spike_times['time_s'].to_numpy(dtype=float) -
        float(stim_onset), rise, decay, window=window, ref_time=ref_time
    )
    a = np.zeros((n_kc, n_odor), dtype=float)
    np.add.at(a, (kc, od), w)
    return pd.DataFrame(a, index=index, columns=columns)


# ---------------------------------------------------------------------------
# measurement noise
def expected_pair_distances_noisy(mat: pd.DataFrame, noise_sd: float, n_trials: int = 1,
    order: Optional[Sequence] = None, *, names: Sequence[str] = PAIR_NAMES) -> pd.Series:
    """Returns the cosine distances of a (units x odors) matrix, in expectation over
    independent zero-mean measurement noise of sd `noise_sd` per entry and trial, after
    averaging `n_trials` trials.

    Closed form: independent noise adds N s^2 / n_trials to each odor's squared length (N
    units, s = `noise_sd`) and nothing, in expectation, to the products between different
    odors, so the distance is 1 - a.b / sqrt((|a|^2 + N s^2 / n)(|b|^2 + N s^2 / n)) (the
    ratio of the expectations, which the sampled mean approaches for many units).
    `noise_sd` 0 gives `hong2p.rdm.pair_distances(mat, 'cosine')`.
    """
    m = mat.astype(float)
    odors = order_items(m.columns, order)
    X = m[odors].to_numpy().T
    n_units = X.shape[1]
    sq = (X ** 2).sum(axis=1) + n_units * float(noise_sd) ** 2 / n_trials
    with np.errstate(invalid='ignore', divide='ignore'):
        S = (X @ X.T) / np.sqrt(np.outer(sq, sq))
    pl = pair_list(odors, order)
    idx = {o: i for i, o in enumerate(odors)}
    return pd.Series([1 - S[idx[a], idx[b]] for a, b in pl], index=pair_index(pl, names))


def sample_pseudo_recordings(signal, noise_sd: float, *, n_recordings: int = 10,
    n_trials: int = 3, seed: int = 0, return_trials: bool = False) -> np.ndarray:
    """Returns pseudo-recordings of a model's readout through measurement noise:
    independent Gaussian noise of sd `noise_sd` added to every entry of every trial
    (seeded).

    Args:
        signal: the noise-free signal: a (units x odors) DataFrame or array (the same
            signal on every trial), or an array (n_recordings, n_trials, units, odors) of
            per-trial signals (e.g. jittered trials read out per trial), whose first two
            dimensions then set `n_recordings` and `n_trials`.
        noise_sd: noise sd per entry and trial.
        return_trials: return the noisy trials instead of their means.

    Returns (n_recordings, units, odors) noisy trial means, or (n_recordings, n_trials,
    units, odors) noisy trials with `return_trials`.
    """
    sig = (signal.to_numpy(dtype=float) if isinstance(signal, pd.DataFrame)
        else np.asarray(signal, dtype=float)
    )
    if sig.ndim == 2:
        sig = np.broadcast_to(sig, (n_recordings, n_trials) + sig.shape)
    elif sig.ndim != 4:
        raise ValueError('signal must be (units, odors) or (recordings, trials, units, '
            f'odors), not {sig.shape}'
        )
    rng = np.random.default_rng(seed)
    trials = sig + rng.normal(0.0, float(noise_sd), size=sig.shape)
    return trials if return_trials else trials.mean(axis=1)


# ---------------------------------------------------------------------------
# each KC z-scored over its whole session (as some recordings are normalized: each KC's
# fluorescence divided by its sd over the whole recording), on the model's side
ZCELL_MAPS = ('linear', 'saturating')
ZCELL_READINGS = ('graded', 'thresholded', 'binary')


def _zcell_amplitude(x: np.ndarray, amp: str, K: Optional[float]) -> np.ndarray:
    if amp == 'linear':
        return x
    if amp == 'saturating':
        if K is None or not K > 0:
            raise ValueError(f'the saturating map needs K > 0, not {K}')
        return x / (x + float(K))
    raise ValueError(f'unknown amplitude map {amp!r} (one of {ZCELL_MAPS})')


def session_zscore(counts, *, q: float, noise, amp: str = 'linear',
    K: Optional[float] = None, noise_ratio: float = 0.385, n_trials: int = 3,
    n_blank: int = 1, divisor: Optional[float] = None, center: bool = False,
    drift: float = 0.0) -> Dict[str, object]:
    """Returns a model's KC responses read as recordings z-scored per KC over the session:
    each KC's session (the panel's odors and `n_blank` blank stimuli, `n_trials` trials
    each) assembled from its spike counts and divided by the KC's sd over the session.

    Each trial's response window holds the share `q` of the session's frames; inside it
    the KC's signal is its amplitude a (`amp`: 'linear', a = n; 'saturating',
    a = n / (n + K)), outside it 0; every frame carries independent noise of sd `noise`
    (amplitude units: the baseline-noise SD) and the trace a slow drift of sd
    drift x noise, which the pre-odor baseline removes from every response but which adds
    to the session's variance. The KC's session sd is
    S = sqrt(q mean(a^2) - (q mean(a))^2 + noise^2 (1 + drift^2)) (means over the
    session's trials), and a trial reads a / S (the window relative to the pre-odor
    baseline, both in the KC's session z units), or with `center` (a - q mean(a)) / S (the
    window mean of the plain session z-score, no baseline subtracted), plus measurement
    noise of sd noise_ratio x noise / S (a silent KC's trial sd is `noise_ratio`). The
    per-KC mean of squared trial values cannot exceed about 1 / q. drift changes no
    reading when the noise is calibrated (noise sqrt(1 + drift^2) and
    noise_ratio / sqrt(1 + drift^2) at drift 0 read identically); it sets the units, so
    that S / noise is the whole-trace SD over the baseline-noise SD.

    Args:
        counts: spike counts, (units x odors) DataFrame or array (the same counts on every
            trial), or (recordings, trials, units, odors) per-trial counts (jittered
            trials; each recording a session).
        q: share of the session's frames inside the response windows.
        noise: frame-noise sd in amplitude units, one value for all units (or an array
            over units).
        amp, K: the amplitude map (`ZCELL_MAPS`); K the saturating map's half-saturation.
        noise_ratio: sd of a response's measurement noise relative to the frame-noise sd.
        n_trials: trials per odor when `counts` is 2-D.
        n_blank: blank (solvent) stimuli in the session, amplitude 0.
        divisor: the control: every unit's amplitude divided by this one value, nothing
            subtracted (no per-KC normalization), measurement noise sd
            noise_ratio x noise / divisor.
        center: subtract the session mean (see above); ignored with `divisor`.
        drift: the slow drift's sd relative to `noise`.

    Returns dict: 'trials' ((recordings, trials, units, odors) noise-free trial values in
    z; trials is 1 for 2-D counts, whose trials are identical), 'center' (their mean over
    trials, (recordings, units, odors)), 'trial_sd' ((recordings, units) sd of one trial's
    measurement noise in z), 'n_trials', 'blank' ((recordings, units) the blank's value
    in z), 'mean_sq' ((recordings, units) expected per-unit mean of squared trial values
    over the session, the blank's included), 'scale_ratio' ((recordings, units) S / noise,
    the whole-trace SD over the baseline-noise SD), 'odors' (column labels or None), and
    the parameters.
    """
    odors = list(counts.columns) if isinstance(counts, pd.DataFrame) else None
    x = (counts.to_numpy(dtype=float) if isinstance(counts, pd.DataFrame)
        else np.asarray(counts, dtype=float)
    )
    if x.ndim == 2:
        x = x[None, None]
        T = int(n_trials)
    elif x.ndim == 4:
        T = x.shape[1]
    else:
        raise ValueError('counts must be (units, odors) or (recordings, trials, units, '
            f'odors), not {x.shape}'
        )
    a = _zcell_amplitude(x, amp, K)
    n_stim = a.shape[3] + int(n_blank)
    # session means over the trials: the blank trials add zeros
    mean_a = a.mean(axis=1).sum(axis=-1) / n_stim
    mean_a2 = (a ** 2).mean(axis=1).sum(axis=-1) / n_stim
    sig = np.broadcast_to(np.asarray(noise, dtype=float), mean_a.shape[-1:])[None, :]
    if divisor is not None:
        S = np.full(mean_a.shape, float(divisor))
        mu = np.zeros_like(mean_a)
    else:
        mu = q * mean_a
        S = np.sqrt(np.maximum(q * mean_a2 - mu ** 2, 0.0) + sig ** 2 *
            (1 + float(drift) ** 2)
        )
        if not center:
            mu = np.zeros_like(mu)
    with np.errstate(invalid='ignore', divide='ignore'):
        vals = (a - mu[:, None, :, None]) / S[:, None, :, None]
        blank = -mu / S
        trial_sd = noise_ratio * sig / S
    vals = np.nan_to_num(vals)
    blank, trial_sd = np.nan_to_num(blank), np.nan_to_num(trial_sd)
    mean_sq = (((vals ** 2).mean(axis=1).sum(axis=-1) + n_blank * blank ** 2) / n_stim +
        trial_sd ** 2
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        scale_ratio = S / sig
    return dict(trials=vals, center=vals.mean(axis=1), trial_sd=trial_sd, n_trials=T,
        blank=blank, mean_sq=mean_sq, scale_ratio=scale_ratio, odors=odors, q=float(q),
        noise=noise, amp=amp, K=K, noise_ratio=float(noise_ratio), n_blank=int(n_blank),
        divisor=divisor, center_mean=bool(center), drift=float(drift)
    )


def tail_moments(m: np.ndarray, sd: np.ndarray, thresh: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns P(x > thresh), E[x 1{x > thresh}] and E[x^2 1{x > thresh}] for
    x ~ N(m, sd^2), elementwise (sd broadcast against m; sd 0 is the noise-free
    indicator)."""
    from scipy.special import ndtr

    m = np.asarray(m, dtype=float)
    sd = np.broadcast_to(np.asarray(sd, dtype=float), m.shape)
    pos = sd > 0
    with np.errstate(invalid='ignore', divide='ignore'):
        a = np.where(pos, (thresh - m) / np.where(pos, sd, 1.0), 0.0)
        P = np.where(pos, ndtr(-a), (m > thresh).astype(float))
        phi = np.exp(-0.5 * a ** 2) / np.sqrt(2 * np.pi)
        E1 = np.where(pos, m * P + sd * phi, m * P)
        E2 = np.where(pos, (m ** 2 + sd ** 2) * P + sd * phi * (m + thresh), m ** 2 * P)
    return P, E1, E2


def _zcell_odors(z: Dict[str, object], n_odors: int, odors: Optional[Sequence]
    ) -> List:
    if odors is None:
        odors = z.get('odors')
    if odors is None:
        return list(range(n_odors))
    odors = list(odors)
    if len(odors) != n_odors:
        raise ValueError(f'{len(odors)} odor labels for {n_odors} odors')
    return odors


def zcell_pair_distances(z: Dict[str, object], reading: str = 'graded',
    thresh: float = 1.0, odors: Optional[Sequence] = None,
    order: Optional[Sequence] = None, *, names: Sequence[str] = PAIR_NAMES) -> pd.Series:
    """Returns the cosine odor-odor distances of a `session_zscore` reading, in expectation
    over its measurement noise, mean over its recordings.

    Closed form, the ratio of the expectations as `expected_pair_distances_noisy`:
    independent noise adds to each odor's squared length and nothing to the products of
    different odors.

    Args:
        z: `session_zscore` output.
        reading: 'graded' (all values, negative ones included), 'thresholded' (trial means
            <= `thresh` set to 0) or 'binary' (trial mean > `thresh`).
        thresh: the response criterion, in z.
        odors: labels of the odor axis (default z['odors'], else positions).
        order: canonical order of the odors (`hong2p.rdm.order_items`).

    Returns a canonical pair Series.
    """
    C = z['center']
    s = z['trial_sd'] / np.sqrt(z['n_trials'])
    odors = _zcell_odors(z, C.shape[2], odors)
    srt = order_items(odors, order)
    idx = [odors.index(o) for o in srt]
    D = []
    for r in range(C.shape[0]):
        m = C[r][:, idx]
        sd = s[r][:, None]
        if reading == 'graded':
            E1, E2 = m, m ** 2 + sd ** 2
        elif reading in ('thresholded', 'binary'):
            P, E1, E2 = tail_moments(m, sd, thresh)
            if reading == 'binary':
                E1 = E2 = P
        else:
            raise ValueError(f'unknown reading {reading!r} (one of {ZCELL_READINGS})')
        n2 = E2.sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            D.append(1 - (E1.T @ E1) / np.sqrt(np.outer(n2, n2)))
    D = np.nanmean(np.stack(D), axis=0) if len(D) > 1 else D[0]
    pl = pair_list(srt, order)
    pos = {o: i for i, o in enumerate(srt)}
    return pd.Series([D[pos[a], pos[b]] for a, b in pl], index=pair_index(pl, names))


def zcell_stats(z: Dict[str, object], thresh: float = 1.0,
    odors: Optional[Sequence] = None) -> Dict[str, object]:
    """Returns statistics of a `session_zscore` reading under a response criterion (trial
    mean > `thresh`), in expectation over the measurement noise, pooled over its
    recordings.

    Keys: 'rate' (share of entries responding) and 'rate_per_odor' (Series, indexed by
    `odors`, default z['odors'] else positions), 'resp_amp_mean' and 'resp_amp_cv' (the
    responders' trial means), 'share_above_4' and 'share_05_15' (entries with a trial mean
    above 4 z, and in (0.5, 1.5] z), 'opk' (the odors-per-KC distribution over
    `kc_stats.OPK_EDGES`), 'silent_share' and 'opk_8plus', 'nonresp_trial_sd' (root mean
    trial-to-trial variance of the non-responding entries: measurement noise plus the
    trials' own variation), the per-unit mean square's upper edge ('mean_sq_max', '_p999',
    '_p99', '_median', 'edge_gap' = max - 99.9th percentile), 'max_trial_mean' and
    'max_trial' (noise-free); by breadth (each unit at its expected number of odors
    responded to, rounded, 11 or more pooled), 'scale_ratio_by_breadth' (the median
    whole-trace SD over the baseline-noise SD) and 'amp_base_by_breadth' (the responders'
    mean amplitude in baseline-noise units, z x that ratio).
    """
    from scipy.special import ndtr

    C = z['center']
    T = z['n_trials']
    t = z['trial_sd']
    sd = t / np.sqrt(T)
    sdb = sd[:, :, None]
    P, E1, E2 = tail_moments(C, sdb, thresh)
    odors = _zcell_odors(z, C.shape[2], odors)
    with np.errstate(invalid='ignore', divide='ignore'):
        amp_mean = float(E1.sum() / P.sum())
        amp_var = float(E2.sum() / P.sum()) - amp_mean ** 2
        P4 = tail_moments(C, sdb, 4.0)[0]
        band = (ndtr((1.5 - C) / np.where(sdb > 0, sdb, np.inf)) -
            ndtr((0.5 - C) / np.where(sdb > 0, sdb, np.inf))
        )
    if not (sd > 0).all():
        band = np.where(sdb > 0, band, ((C > 0.5) & (C <= 1.5)).astype(float))
    # the trials' own variation (jittered trials) adds to the measurement noise
    tr = z['trials']
    var_tr = tr.var(axis=1, ddof=1) if tr.shape[1] > 1 else np.zeros_like(C)
    w = 1 - P
    nonresp_var = float((w * (t[:, :, None] ** 2 + var_tr)).sum() / w.sum())
    opk = np.mean([poisson_binomial_bins(P[r]) for r in range(P.shape[0])], axis=0)
    ms = z['mean_sq'].ravel()
    breadth = np.minimum(np.rint(P.sum(axis=2)), 11).ravel()
    sr = np.asarray(z['scale_ratio'], dtype=float).ravel()
    e1b = (E1 * z['scale_ratio'][:, :, None]).sum(axis=2).ravel()
    pb = P.sum(axis=2).ravel()
    sr_b, amp_b = {}, {}
    for b in range(12):
        m = breadth == b
        if m.any():
            sr_b[b] = float(np.median(sr[m]))
            if b > 0 and pb[m].sum() > 0:
                amp_b[b] = float(e1b[m].sum() / pb[m].sum())
    return dict(rate=float(P.mean()), rate_per_odor=pd.Series(P.mean(axis=(0, 1)),
        index=odors), resp_amp_mean=amp_mean,
        resp_amp_cv=(float(np.sqrt(max(amp_var, 0.0)) / amp_mean) if amp_mean > 0
            else np.nan),
        share_above_4=float(P4.mean()), share_05_15=float(band.mean()), opk=opk,
        silent_share=float(opk[0]), opk_8plus=float(opk[-1]),
        nonresp_trial_sd=float(np.sqrt(nonresp_var)), mean_sq_max=float(ms.max()),
        mean_sq_p999=float(np.quantile(ms, 0.999)), mean_sq_p99=float(np.quantile(ms, 0.99)),
        mean_sq_median=float(np.median(ms)), edge_gap=float(ms.max() - np.quantile(ms, 0.999)),
        max_trial_mean=float(C.max()), max_trial=float(tr.max()),
        scale_ratio_by_breadth=sr_b, amp_base_by_breadth=amp_b
    )


def _zcell_rate(on_counts, q, noise, amp, K, noise_ratio, n_trials, n_blank, center,
    drift=0.0, thresh=1.0) -> Tuple[float, Dict]:
    z = session_zscore(on_counts, q=q, noise=noise, amp=amp, K=K, noise_ratio=noise_ratio,
        n_trials=n_trials, n_blank=n_blank, center=center, drift=drift
    )
    P = tail_moments(z['center'], (z['trial_sd'] / np.sqrt(z['n_trials']))[:, :, None],
        thresh
    )[0]
    return float(P.mean()), z


def calibrate_session_zscore(on_counts, *, q: float, amp: str = 'linear',
    target_rate: float, target_amp: Optional[float] = None,
    target_silent_sd: Optional[float] = None, K: Optional[float] = None,
    noise_ratio: float = 0.385, n_trials: int = 3, n_blank: int = 1, center: bool = False,
    drift: float = 0.0, noise_bounds: Tuple[float, float] = (1e-4, 1e4),
    K_bounds: Tuple[float, float] = (1e-2, 1e3), n_K_grid: int = 25, tol: float = 1e-4,
    max_iter: int = 20, thresh: float = 1.0) -> Dict[str, object]:
    """Returns the parameters of `session_zscore` set on the APL-on counts alone (then used
    unchanged for APL off, so the effect of APL stays a prediction).

    The frame noise is set so that the APL-on response rate under the criterion (expected
    trial mean > `thresh` z) is `target_rate`; for the saturating map with `target_amp`,
    also K, so that the responders' mean amplitude (z) is `target_amp`; with
    `target_silent_sd`, the noise ratio so that the trial-to-trial sd of the
    non-responding entries (`zcell_stats`' 'nonresp_trial_sd') is that value (iterated to
    a fixed point; it barely moves the other two). The linear map has the noise as its
    only free parameter, so its responders' amplitude is a prediction. The rate falls
    monotonically with the noise (root in log noise, Brent); K is the root of the
    amplitude error nearest the grid point with the smallest error among the sign changes
    on a log grid over `K_bounds` (the amplitude, with the noise re-solved at every K,
    need not be monotone in K), or the grid point with the smallest error when there is no
    sign change ('K_at_bound').

    Args:
        on_counts: APL-on spike counts, (units x odors) or (recordings, trials, units,
            odors).
        K: the saturating map's K when it is not calibrated (no `target_amp`).
        thresh: the response criterion, in z.

    Returns dict: 'noise', 'K', 'noise_ratio', 'rate', 'resp_amp_mean', 'nonresp_trial_sd'
    (achieved), 'noise_at_bound', 'K_at_bound', 'converged', 'iterations', 'q', 'amp',
    'center', 'drift'.
    """
    from scipy.optimize import brentq

    lo_n, hi_n = np.log(noise_bounds[0]), np.log(noise_bounds[1])

    def solve_noise(Kv, nr):
        f = lambda ln: _zcell_rate(on_counts, q, np.exp(ln), amp, Kv, nr, n_trials,
            n_blank, center, drift, thresh)[0] - target_rate
        flo, fhi = f(lo_n), f(hi_n)
        if flo * fhi > 0:
            return float(np.exp(lo_n if abs(flo) < abs(fhi) else hi_n)), True
        return float(np.exp(brentq(f, lo_n, hi_n, xtol=1e-10, rtol=1e-10))), False

    def stats_at(Kv, nr):
        noise, nb = solve_noise(Kv, nr)
        z = session_zscore(on_counts, q=q, noise=noise, amp=amp, K=Kv, noise_ratio=nr,
            n_trials=n_trials, n_blank=n_blank, center=center, drift=drift
        )
        return noise, nb, z

    def amp_of(z):
        C = z['center']
        sd = (z['trial_sd'] / np.sqrt(z['n_trials']))[:, :, None]
        P, E1, _ = tail_moments(C, sd, thresh)
        return float(E1.sum() / P.sum()) if P.sum() > 0 else np.nan

    fit_K = amp == 'saturating' and target_amp is not None
    if amp == 'saturating' and not fit_K and K is None:
        raise ValueError('the saturating map needs K, or target_amp to calibrate it')
    nr = float(noise_ratio)
    Kv = K
    K_at_bound = False
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        fk = lambda lk: amp_of(stats_at(np.exp(lk), nr)[2]) - target_amp
        if fit_K and it > 1 and not K_at_bound:
            # later rounds (the noise ratio moved a little): the root near the previous K
            lo_k, hi_k = np.log(Kv) - 0.7, np.log(Kv) + 0.7
            if fk(lo_k) * fk(hi_k) <= 0:
                Kv = float(np.exp(brentq(fk, lo_k, hi_k, xtol=1e-8, rtol=1e-8)))
            else:
                K_at_bound = True
        if fit_K and (it == 1 or K_at_bound):
            grid = np.linspace(np.log(K_bounds[0]), np.log(K_bounds[1]), n_K_grid)
            err = np.array([amp_of(stats_at(np.exp(g), nr)[2]) - target_amp for g in grid])
            ok = np.isfinite(err)
            best = int(np.nanargmin(np.where(ok, np.abs(err), np.nan)))
            brackets = [i for i in range(len(grid) - 1)
                if ok[i] and ok[i + 1] and err[i] * err[i + 1] <= 0
            ]
            if brackets:
                i = min(brackets, key=lambda i: abs(i + 0.5 - best))
                Kv = float(np.exp(brentq(fk, grid[i], grid[i + 1], xtol=1e-8, rtol=1e-8)))
                K_at_bound = False
            else:
                Kv, K_at_bound = float(np.exp(grid[best])), True
        noise, noise_at_bound, z = stats_at(Kv, nr)
        if target_silent_sd is None:
            converged = True
            break
        st = zcell_stats(z, thresh)
        new_nr = nr * float(target_silent_sd) / st['nonresp_trial_sd']
        if abs(new_nr - nr) < tol * nr:
            nr = new_nr
            noise, noise_at_bound, z = stats_at(Kv, nr)
            converged = True
            break
        nr = new_nr
    st = zcell_stats(z, thresh)
    return dict(noise=noise, K=Kv, noise_ratio=nr, rate=st['rate'],
        resp_amp_mean=st['resp_amp_mean'], nonresp_trial_sd=st['nonresp_trial_sd'],
        noise_at_bound=bool(noise_at_bound), K_at_bound=bool(K_at_bound),
        converged=converged, iterations=it, q=float(q), amp=amp, center=bool(center),
        drift=float(drift)
    )


# ---------------------------------------------------------------------------
# a model output's odor-odor distances and response fractions
MODEL_RDM_VARIANTS: Dict[str, Tuple[str, str, bool]] = {
    # name -> (source, measure, drop_silent)
    'spike_counts_cosine': ('spike_counts', 'cosine', False),
    'spike_counts_pearson': ('spike_counts', 'pearson', False),
    'spike_counts_pearson_nosilent': ('spike_counts', 'pearson', True),
    'responses_cosine': ('responses', 'cosine', False),
    'responses_pearson': ('responses', 'pearson', False),
    'responses_pearson_nosilent': ('responses', 'pearson', True),
}
PRIMARY_MODEL_RDM = 'spike_counts_cosine'


def model_rdms(responses: pd.DataFrame, spike_counts: pd.DataFrame,
    order: Optional[Sequence] = None) -> Dict[str, pd.Series]:
    """Returns every `MODEL_RDM_VARIANTS` pair Series of one model output (its binarized
    `responses` and its `spike_counts`, KCs x odors)."""
    src = {'responses': responses.astype(float), 'spike_counts': spike_counts.astype(float)}
    return {name: pair_distances(src[s], measure=meas, drop_silent=ds, order=order)
        for name, (s, meas, ds) in MODEL_RDM_VARIANTS.items()
    }


def response_fractions(responses: pd.DataFrame) -> pd.Series:
    """Returns the fraction of model KCs responding to each odor (columns)."""
    return responses.astype(float).mean(axis='index')
