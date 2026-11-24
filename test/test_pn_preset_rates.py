"""Tests of PN rates given directly (`fit_mb_model`'s `pn_rates`, olfsysm's
`rv.pn.preset_rates`; Yang's OK of 2026-11-24, for the KC-remapping queue's proposal "the
antennal lobe held to the recorded PN dendrites").

(a) Unset (the default), the outputs are those of the build before the option (olfsysm
45e5cf9), frozen in test_data/pn_preset_rates/unset.npz: the spike counts of the
one-row-per-KC hemibrain model at fixed parameters and tuned (with its tuned threshold and
APL weight), and of the hemibrain bouton model, and every odor's PN traces (and the bouton
model's bouton traces) bit for bit, by sha256; a table of NaN changes nothing either.
(b) With every entry set to the model's own odor-evoked rates (olfsysm's `pn_evoked_rates`
of an unset run, or the same reading in numpy, which differs in the last bits), every
output is the unset run's, bit for bit, at fixed parameters, tuned and in the bouton model.
(c) A set PN's trace is the model's PN dynamics for that odor at a shifted ORN drive (a
numpy port of the PN stage, solved on its own, gives the same trace), with the given rate
to within 1e-9 Hz, and only its trace changes: from odor onset on, raised at every step by
a higher rate; with APL's output at 0 only the KCs it reaches change, and in the bouton
model with APL's output to the boutons at 0 only its boutons. (d) A NaN mask: the NaN
entries keep the model's traces bit for bit, the others get their rates; the bounds hold.
(e) Tuning and the fixed-parameter simulator see the given rates. (f) Bad tables, shapes,
values and settings are refused, in pyolfsysm and in olfsysm's own checks; model IDs and
cache keys do not change without the option.

To re-freeze (only with a build whose outputs are known good, e.g. olfsysm 45e5cf9, with
which it was frozen): `python test/test_pn_preset_rates.py --freeze`. The freezing uses
only what 45e5cf9's pyolfsysm has.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

import olfsysm as osm
from pyolfsysm.mb_model import (fit_mb_model, fixed_params_to_fit_mb_model_kws,
    format_model_params, make_fixed_param_kc_simulator, megamat_orn_deltas,
    onestep_lr_cache_key, KC_ID
)

pytestmark = pytest.mark.filterwarnings('ignore')

# (not from conftest, so that the file can also be run as a script, to freeze)
FROZEN = Path(__file__).resolve().parent / 'test_data' / 'pn_preset_rates' / 'unset.npz'

COMMON = dict(make_plots=False, silent=True)
# the one-row-per-KC hemibrain model (olfsysm's branch without claws)
PERKC_KWS = dict(weight_divisor=20, **COMMON)
PERKC_FIXED = dict(fixed_thr=333.0, wAPLKC=2.6)
PERKC_TUNED = dict(target_sparsity=0.1)
# the hemibrain bouton model (claws and boutons; test_kckc_negative.py's)
BOUTON_KWS = dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
    multibouton_claws='first', use_connectome_APL_weights=True, pn_claw_to_apl=True,
    n_spikes_for_response=2, **COMMON
)
BOUTON_FIXED = dict(fixed_thr=275.0, wAPLKC=0.13)
N_BOUTON_ODORS = 4

RUN_KWS = dict(perkc=dict(**PERKC_FIXED, **PERKC_KWS),
    tuned=dict(**PERKC_TUNED, **PERKC_KWS), bouton=dict(**BOUTON_FIXED, **BOUTON_KWS)
)

# olfsysm's tolerance on a given rate (Hz; PN_PRESET_RATE_TOL)
TOL = 1e-9


def sha(x) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x, dtype=np.float64)
        ).tobytes()).hexdigest()


def sims_sha(sims) -> np.ndarray:
    return np.array([sha(x) for x in sims])


def run_orn(orn, name):
    return orn.iloc[:, :N_BOUTON_ODORS] if name == 'bouton' else orn


def frozen_runs(orn, **kws):
    """{name: (spike counts, wPNKC, param_dict)} of the frozen runs
    (`return_olfsysm_vars=True`), with `kws` passed to each: perkc (fixed parameters, the
    17 megamat odors), tuned (the same model tuned to `PERKC_TUNED`) and bouton (the
    bouton model, its first `N_BOUTON_ODORS` odors)."""
    out = dict()
    for name, run_kws in RUN_KWS.items():
        _, s, w, p = fit_mb_model(run_orn(orn, name), return_olfsysm_vars=True,
            **run_kws, **kws
        )
        out[name] = (s, w, p)
    return out


def freeze() -> None:
    """Writes `FROZEN` from this build's runs (without the option)."""
    orn = megamat_orn_deltas()
    arrays = dict()
    for name, (s, _, p) in frozen_runs(orn).items():
        rv = p['rv']
        arrays[name] = s.to_numpy().astype(np.int32)
        arrays[f'{name}_kc_ids'] = s.index.get_level_values(KC_ID).to_numpy()
        arrays[f'{name}_pn_sha'] = sims_sha(rv.pn.pn_sims)
        if name == 'bouton':
            arrays[f'{name}_bouton_sha'] = sims_sha(rv.pn.bouton_sims)
        if name == 'tuned':
            arrays['tuned_thr'] = np.asarray(p['fixed_thr'], dtype=np.float64)
            arrays['tuned_wAPLKC'] = np.asarray(p['wAPLKC'], dtype=np.float64)
    bi = osm.build_info()
    meta = dict(olfsysm_commit=bi['git_hash'], git_dirty=bi['git_dirty'],
        cpp_dirty=bi['cpp_dirty'], source_sha256=bi['source_sha256'],
        build_host=bi['build_host'], odors=[str(c) for c in orn.columns]
    )
    arrays['meta'] = np.array(json.dumps(meta))
    FROZEN.parent.mkdir(exist_ok=True)
    np.savez_compressed(FROZEN, **arrays)
    print(f'wrote {FROZEN}: {meta}')


# ---------------------------------------------------------------------------------------
# (what follows needs the option, so is not used by `freeze`)

def own_rates(s: pd.DataFrame, w: pd.DataFrame, p: dict) -> pd.DataFrame:
    """The model's own odor-evoked PN rates (glomeruli x the output's odors)."""
    from pyolfsysm.mb_model import pn_evoked_rates
    rates = pn_evoked_rates(p['mp'], p['rv'], wPNKC=w, odors=s.columns)
    assert rates.columns.equals(s.columns)
    assert len(rates) == np.asarray(p['rv'].pn.pn_sims[0]).shape[0]
    return rates


def bounds(s: pd.DataFrame, w: pd.DataFrame, p: dict):
    from pyolfsysm.mb_model import pn_rate_bounds
    return pn_rate_bounds(p['mp'], p['rv'], wPNKC=w, odors=s.columns)


def steps(mp):
    """(start, onset, offset) steps of olfsysm's rate reading."""
    dt = mp.time_dt
    return tuple(int((x - mp.time_pre_start) / dt)
        for x in (mp.time_start, mp.time_stim_start, mp.time_stim_end)
    )


def numpy_rates(mp, rv, index, columns) -> pd.DataFrame:
    """The same reading as olfsysm's `pn_evoked_rates`, in numpy (the steps the odor is on,
    minus the steps from `mp.time_start` to odor onset)."""
    start, on, off = steps(mp)
    pn = np.stack([np.asarray(x) for x in rv.pn.pn_sims])
    r = pn[:, :, on:off].mean(axis=-1) - pn[:, :, start:on].mean(axis=-1)
    return pd.DataFrame(r.T, index=index, columns=columns)


def numpy_rate(mp, trace) -> float:
    start, on, off = steps(mp)
    return float(trace[on:off].mean() - trace[start:on].mean())


def numpy_unit_orn(mp) -> np.ndarray:
    """sim_ORN_layer's response to a unit delta (its smoothing, then its low-pass)."""
    n = int((mp.time_end - mp.time_pre_start) / mp.time_dt)
    _, on, off = steps(mp)
    stim = np.zeros(n)
    stim[on:off] = 1.0
    w = 0.02 / mp.time_dt
    a = 2.0 / (w + 1.0) if w > 1.0 else w
    for t in range(1, n):
        stim[t] = a * stim[t] + (1 - a) * stim[t - 1]
    k = np.zeros(n)
    mul = mp.time_dt / mp.orn.taum
    for t in range(1, n):
        k[t] = k[t - 1] * (1.0 - mul) + stim[t] * mul
    return k


def numpy_shifted_trace(mp, rv, model_pn, i: int, g: int, shift: float, unit_orn
    ) -> np.ndarray:
    """A numpy port of sim_PN_layer for glomerulus g and odor i, with the ORN drive above
    spontaneous shifted by `shift` times the unit ORN response from odor onset on, and the
    model's trace (`model_pn`, glomeruli x steps) before onset."""
    orn_t = np.asarray(rv.orn.sims[i])[g]
    inhA = np.asarray(rv.ln.ln_inhA.sims[i]).ravel()
    inhB = np.asarray(rv.ln.ln_inhB.sims[i]).ravel()
    spont_orn = np.asarray(mp.orn.data.spont).ravel()
    pn = mp.pn
    spont = spont_orn[g] * pn.inhsc / (spont_orn.sum() + pn.inhadd)
    _, on, _ = steps(mp)
    out = np.array(model_pn[g], dtype=float)
    inh = pn.inhsc / (pn.inhadd + 0.25 * inhA[on - 1] + 0.75 * inhB[on - 1])
    c = mp.time_dt / pn.taum
    for t in range(on, len(out)):
        d = orn_t[t - 1] - spont_orn[g] + shift * unit_orn[t - 1]
        dpn = -out[t - 1] + spont + 200.0 * np.tanh((d + pn.offset) * pn.tanhsc / 200.0 *
            inh
        )
        inh = pn.inhsc / (pn.inhadd + 0.25 * inhA[t] + 0.75 * inhB[t])
        out[t] = max(out[t - 1] + dpn * c, 0.0)
    return out


@pytest.fixture(scope='module')
def orn():
    return megamat_orn_deltas()


@pytest.fixture(scope='module')
def frozen():
    with np.load(FROZEN, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope='module')
def runs_unset(orn):
    return frozen_runs(orn)


def check_against_frozen(frozen, runs, label):
    meta = json.loads(str(frozen['meta']))
    assert meta['git_dirty'] == 0 and meta['cpp_dirty'] == 0
    for name, (s, _, p) in runs.items():
        rv = p['rv']
        assert np.array_equal(s.index.get_level_values(KC_ID).to_numpy(),
            frozen[f'{name}_kc_ids']
        )
        n = int((s.to_numpy() != frozen[name]).sum())
        print(f'{label} {name}: {n}/{s.size} spike counts differ from '
            f'{meta["olfsysm_commit"][:10]} ({int(s.to_numpy().sum())} spikes)'
        )
        assert n == 0
        assert np.array_equal(sims_sha(rv.pn.pn_sims), frozen[f'{name}_pn_sha'])
        if name == 'bouton':
            assert np.array_equal(sims_sha(rv.pn.bouton_sims),
                frozen[f'{name}_bouton_sha']
            )
        if name == 'tuned':
            assert np.array_equal(np.asarray(p['fixed_thr'], dtype=np.float64),
                frozen['tuned_thr']
            )
            assert np.array_equal(np.asarray(p['wAPLKC'], dtype=np.float64),
                frozen['tuned_wAPLKC']
            )


def test_unset_equals_frozen(frozen, runs_unset):
    """(a) Unset, every run gives the frozen spike counts and PN (and bouton) traces
    exactly, and the tuned run the frozen threshold and APL weight; no option in the
    params, and olfsysm's matrix is empty."""
    check_against_frozen(frozen, runs_unset, 'unset')
    for name, (_, _, p) in runs_unset.items():
        assert not any(k.startswith('pn_rates') for k in p)
        assert np.asarray(p['rv'].pn.preset_rates).size == 0


def test_all_nan_is_unset(orn, frozen, runs_unset):
    """(a) A table of NaN (every glomerulus and odor kept) changes nothing, bit for bit."""
    runs = dict()
    for name, (s0, w0, _) in runs_unset.items():
        nan = own_rates(s0, w0, runs_unset[name][2]) * np.nan
        _, s, w, p = fit_mb_model(run_orn(orn, name), pn_rates=nan,
            return_olfsysm_vars=True, **RUN_KWS[name]
        )
        assert p['pn_rates_n_set'] == 0
        runs[name] = (s, w, p)
    check_against_frozen(frozen, runs, 'all NaN')


@pytest.mark.parametrize('reading', ['olfsysm', 'numpy'])
def test_own_rates_identity(orn, runs_unset, reading):
    """(b) Every entry set to the model's own rates: outputs and traces are the unset
    run's, bit for bit (the entries are within olfsysm's tolerance of the model's rates,
    so the model's traces are kept), at fixed parameters, tuned and in the bouton
    model."""
    for name, (s0, w0, p0) in runs_unset.items():
        rates = own_rates(s0, w0, p0)
        if reading == 'numpy':
            ref = rates
            rates = numpy_rates(p0['mp'], p0['rv'], ref.index, ref.columns)
            diff = np.abs(rates.to_numpy() - ref.to_numpy())
            print(f'{name}: the numpy reading differs from olfsysm\'s by up to '
                f'{diff.max():.2g} Hz'
            )
            assert diff.max() < 1e-11
        _, s, _, p = fit_mb_model(run_orn(orn, name), pn_rates=rates,
            return_olfsysm_vars=True, **RUN_KWS[name]
        )
        assert p['pn_rates_n_set'] == rates.size
        assert s.equals(s0), name
        assert np.array_equal(sims_sha(p['rv'].pn.pn_sims),
            sims_sha(p0['rv'].pn.pn_sims)
        )
        if name == 'bouton':
            assert np.array_equal(sims_sha(p['rv'].pn.bouton_sims),
                sims_sha(p0['rv'].pn.bouton_sims)
            )
        if name == 'tuned':
            assert np.array_equal(np.asarray(p['fixed_thr']), np.asarray(p0['fixed_thr']))
            assert p['wAPLKC'] == p0['wAPLKC']


def test_set_trace_is_shifted_pn_stage(orn, runs_unset):
    """(c) A set PN's trace is the model's PN stage for that odor at a shifted ORN drive:
    a numpy port of sim_PN_layer, its shift solved on its own (scipy's brentq), gives the
    same trace; the port at no shift is the model's trace."""
    from scipy.optimize import brentq
    s0, w0, p0 = runs_unset['perkc']
    mp0, rv0 = p0['mp'], p0['rv']
    rates = own_rates(s0, w0, p0)
    lo, hi = bounds(s0, w0, p0)
    unit = numpy_unit_orn(mp0)
    # a glomerulus w/ ORN input that responds to some odors and not others: the odor it
    # responds to least set to 60 Hz, the one it responds to most to half its rate, and a
    # glomerulus without ORN input (not in the megamat ORN data) to 30 Hz
    g = int(np.argmax(rates.max(axis=1).to_numpy() - rates.min(axis=1).to_numpy()))
    weak, strong = int(np.argmin(rates.iloc[g])), int(np.argmax(rates.iloc[g]))
    g0 = [i for i, x in enumerate(rates.index) if x not in set(orn.index)][0]
    cases = [(g, weak, 60.0), (g, strong, 0.5 * rates.iat[g, strong]), (g0, weak, 30.0)]
    table = rates * np.nan
    for gi, oi, r in cases:
        table.iat[gi, oi] = r
    _, s, _, p = fit_mb_model(orn, pn_rates=table, return_olfsysm_vars=True,
        **RUN_KWS['perkc']
    )
    rv = p['rv']
    for gi, oi, r in cases:
        model_pn = np.asarray(rv0.pn.pn_sims[oi])
        assert np.abs(numpy_shifted_trace(mp0, rv0, model_pn, oi, gi, 0.0, unit) -
            model_pn[gi]).max() < 1e-9
        f = lambda x: numpy_rate(mp0, numpy_shifted_trace(mp0, rv0, model_pn, oi, gi, x,
            unit)) - r
        x = brentq(f, -1e4, 1e4, xtol=1e-12)
        expected = numpy_shifted_trace(mp0, rv0, model_pn, oi, gi, x, unit)
        got = np.asarray(rv.pn.pn_sims[oi])[gi]
        print(f'{rates.index[gi]}, {rates.columns[oi]}: {rates.iat[gi, oi]:.3f} -> {r} Hz '
            f'(ORN drive shift {x:.4g} Hz); numpy port off by '
            f'{np.abs(got - expected).max():.2g} Hz'
        )
        assert np.abs(got - expected).max() < 1e-6
        assert abs(numpy_rate(mp0, got) - r) < 1e-8
        assert lo.iat[gi, oi] < r < hi.iat[gi, oi]


def test_one_glomerulus(orn, runs_unset):
    """(c) One glomerulus set, for every odor: only its PN trace changes, from odor onset
    on, raised at every step by a higher rate, to the given rate; the others' traces are
    the model's bit for bit. With APL's output at 0, only KCs it reaches change their
    spike counts. In the bouton model with APL's output to the boutons at 0, only its
    boutons' rates change."""
    s0, w0, p0 = runs_unset['perkc']
    mp0, rv0 = p0['mp'], p0['rv']
    rates = own_rates(s0, w0, p0)
    _, on, _ = steps(mp0)
    g = int(np.argmax(rates.max(axis=1).to_numpy()))
    glom = rates.index[g]
    table = rates * np.nan
    table.iloc[g] = rates.iloc[g] + 15.0
    for apl in (dict(), dict(wAPLKC=0.0)):
        kws = {**RUN_KWS['perkc'], **apl}
        _, sb, _, pb = fit_mb_model(orn, return_olfsysm_vars=True, **kws)
        _, s, _, p = fit_mb_model(orn, pn_rates=table, return_olfsysm_vars=True, **kws)
        assert p['pn_rates_n_set'] == rates.shape[1]
        got = own_rates(s, w0, p)
        assert (np.abs(got.iloc[g] - table.iloc[g]) <= TOL).all()
        others = np.arange(len(rates)) != g
        assert np.array_equal(got.to_numpy()[others], own_rates(sb, w0, pb).to_numpy(
            )[others]
        )
        for i in range(rates.shape[1]):
            a = np.asarray(pb['rv'].pn.pn_sims[i])
            b = np.asarray(p['rv'].pn.pn_sims[i])
            assert np.array_equal(a[others], b[others])
            assert np.array_equal(a[g, :on], b[g, :on])
            assert (b[g, on:] >= a[g, on:]).all() and (b[g, on:] > a[g, on:]).any()
        if apl:
            reached = w0[glom].to_numpy() > 0
            changed = (s.to_numpy() != sb.to_numpy()).any(axis=1)
            print(f'{glom} +15 Hz, APL output at 0: {changed.sum()} KCs change, of '
                f'{reached.sum()} it reaches'
            )
            assert changed.any() and not changed[~reached].any()
            assert (s.to_numpy()[reached] >= sb.to_numpy()[reached]).all()

    # the bouton model, APL's output to the boutons at 0
    o = run_orn(orn, 'bouton')
    kws = {**RUN_KWS['bouton'], 'wAPLPN': 0.0}
    _, sb, wb, pb = fit_mb_model(o, return_olfsysm_vars=True, **kws)
    rates_b = own_rates(sb, wb, pb)
    table = rates_b * np.nan
    table.loc[glom] = rates_b.loc[glom] + 15.0
    _, s, _, p = fit_mb_model(o, pn_rates=table, return_olfsysm_vars=True, **kws)
    mine = (wb.columns.get_level_values('glomerulus') == glom)
    for i in range(o.shape[1]):
        a = np.asarray(pb['rv'].pn.bouton_sims[i])
        b = np.asarray(p['rv'].pn.bouton_sims[i])
        assert a.shape[0] == len(mine)
        # (olfsysm's first step has no bouton rate: NaN)
        assert np.array_equal(a[~mine], b[~mine], equal_nan=True)
        assert np.array_equal(a[mine, :on], b[mine, :on], equal_nan=True)
        assert (b[mine, on:] >= a[mine, on:]).all() and (b[mine, on:] > a[mine, on:]).any()
    assert not s.equals(sb)


def test_nan_mask(orn, runs_unset):
    """(d) Entries of a random mask set to rates drawn from the model's own pooled
    rates (kept inside each entry's bounds), the others NaN: the NaN entries keep the
    model's traces bit for bit, the set ones get their rates; the bounds bracket the
    model's own rates, the lower one about minus the spontaneous rate."""
    s0, w0, p0 = runs_unset['perkc']
    mp0, rv0 = p0['mp'], p0['rv']
    rates = own_rates(s0, w0, p0)
    lo, hi = bounds(s0, w0, p0)
    assert (lo.to_numpy() < rates.to_numpy()).all()
    assert (rates.to_numpy() < hi.to_numpy()).all()
    start, on, _ = steps(mp0)
    spont = np.asarray(rv0.pn.pn_sims[0])[:, start:on].mean(axis=1)
    print(f'lower bound + spontaneous rate: {(lo.to_numpy() + spont[:, None]).min():.3g}'
        f' to {(lo.to_numpy() + spont[:, None]).max():.3g} Hz; upper bounds '
        f'{hi.to_numpy().min():.4g} to {hi.to_numpy().max():.4g} Hz'
    )
    assert np.abs(lo.to_numpy() + spont[:, None]).max() < 1.0

    rng = np.random.default_rng(0)
    mask = rng.random(rates.shape) < 0.4
    pooled = rng.permutation(rates.to_numpy().ravel()).reshape(rates.shape)
    eps = 1e-6
    pooled = np.clip(pooled, lo.to_numpy() + eps, hi.to_numpy() - eps)
    table = pd.DataFrame(np.where(mask, pooled, np.nan), index=rates.index,
        columns=rates.columns
    )
    # (as the queue's tables may come: a subset of glomeruli, odors in another order and
    # w/o the 'panel' level)
    table = table.loc[table.notna().any(axis=1)].iloc[::-1, ::-1]
    _, s, _, p = fit_mb_model(orn, pn_rates=table, return_olfsysm_vars=True,
        **RUN_KWS['perkc']
    )
    assert p['pn_rates_n_set'] == int(mask.sum())
    got = own_rates(s, w0, p)
    for i in range(rates.shape[1]):
        a = np.asarray(rv0.pn.pn_sims[i])
        b = np.asarray(p['rv'].pn.pn_sims[i])
        assert np.array_equal(a[~mask[:, i]], b[~mask[:, i]])
        assert np.array_equal(a[:, :on], b[:, :on])
    err = np.abs(got.to_numpy() - pooled)[mask]
    print(f'{mask.sum()} rates set, off by up to {err.max():.2g} Hz')
    assert err.max() <= TOL
    assert not s.equals(s0)

    # the bounds themselves are reached
    edge = rates * np.nan
    edge.iat[0, 0] = lo.iat[0, 0] + eps
    edge.iat[0, 1] = hi.iat[0, 1] - eps
    _, s, _, p = fit_mb_model(orn, pn_rates=edge, return_olfsysm_vars=True,
        **RUN_KWS['perkc']
    )
    got = own_rates(s, w0, p)
    assert abs(got.iat[0, 0] - edge.iat[0, 0]) <= TOL
    assert abs(got.iat[0, 1] - edge.iat[0, 1]) <= TOL


def test_tuning_and_simulator_see_rates(orn, runs_unset):
    """(e) Tuning with given rates: its threshold / APL weight differ from the unset
    tuning's, and fixing them (fit_mb_model, and make_fixed_param_kc_simulator's
    `simulate` at its initial and APL-off parameters) with the same rates gives the tuned
    outputs, and outputs that differ from those without the rates."""
    s0, w0, p0 = runs_unset['tuned']
    rates = own_rates(s0, w0, p0)
    table = rates * np.nan
    top = rates.max(axis=1).sort_values().index[-3:]
    table.loc[top] = rates.loc[top] + 20.0
    r, s, _, p = fit_mb_model(orn, pn_rates=table, **RUN_KWS['tuned'])
    assert p['pn_rates_n_set'] == table.notna().to_numpy().sum()
    print(f"tuned threshold {np.mean(p0['fixed_thr']):.3f} -> {np.mean(p['fixed_thr']):.3f}"
        f", wAPLKC {p0['wAPLKC']:.4g} -> {p['wAPLKC']:.4g} with the rates"
    )
    assert not (np.array_equal(np.asarray(p['fixed_thr']), np.asarray(p0['fixed_thr']))
        and p['wAPLKC'] == p0['wAPLKC']
    )
    fixed = dict(fixed_thr=p['fixed_thr'], wAPLKC=p['wAPLKC'])
    r2, s2, _, _ = fit_mb_model(orn, pn_rates=table, **fixed, **PERKC_KWS)
    assert s2.equals(s) and r2.equals(r)
    _, s3, _, _ = fit_mb_model(orn, **fixed, **PERKC_KWS)
    assert not s3.equals(s)

    initial = dict(fixed_thr=float(np.mean(p['fixed_thr'])), wAPLKC_scale=p['wAPLKC'])
    simulate, params0 = make_fixed_param_kc_simulator(orn, initial_params=initial,
        pn_rates=table, **PERKC_KWS
    )
    for params in (params0, {**params0, 'wAPLKC_scale': 0.0}):
        rs, ss = simulate(params)
        rf, sf, _, _ = fit_mb_model(orn, pn_rates=table, **PERKC_KWS,
            **fixed_params_to_fit_mb_model_kws(params)
        )
        assert ss.equals(sf) and rs.equals(rf)


def test_refused(orn, runs_unset):
    """(f) Bad tables and settings are refused: in pyolfsysm (not a DataFrame, glomeruli
    or odors not the model's, duplicates, a MultiIndex, non-numeric or infinite values,
    tune_on_hallem, a rate outside the bounds, w/ the glomerulus and odor named) and in
    olfsysm's own checks (shape, infinite values, PN noise, an unreachable rate)."""
    s0, w0, p0 = runs_unset['perkc']
    rates = own_rates(s0, w0, p0)
    lo, _ = bounds(s0, w0, p0)
    kws = RUN_KWS['perkc']
    o = orn.iloc[:, :3]
    good = rates.iloc[:, :3]

    bad_gloms = good.copy()
    bad_gloms.index = list(good.index[:-1]) + ['not_a_glomerulus']
    dup = pd.concat([good, good.iloc[:1]])
    unknown_odor = good.rename(columns={good.columns[0]: 'nope @ -3'})
    strings = good.astype(object)
    strings.iat[0, 0] = 'x'
    inf = good.copy()
    inf.iat[0, 0] = np.inf
    multi = good.copy()
    multi.index = pd.MultiIndex.from_arrays([good.index, range(len(good))],
        names=['glomerulus', 'n']
    )
    for bad in (good.to_numpy(), good.iloc[:, 0], bad_gloms, dup, unknown_odor,
            good.loc[:, [good.columns[0], good.columns[0]]], strings, inf, -inf, multi,
            good.iloc[:0], good.iloc[:, :0]):
        with pytest.raises(ValueError):
            fit_mb_model(o, pn_rates=bad, **kws)
    with pytest.raises(NotImplementedError):
        fit_mb_model(o, pn_rates=good, tune_on_hallem=True, **PERKC_KWS)

    below = good * np.nan
    below.iat[2, 1] = lo.iat[2, 1] - 1.0
    with pytest.raises(ValueError, match='outside the odor-evoked rates') as info:
        fit_mb_model(o, pn_rates=below, **kws)
    print(info.value)
    assert repr(good.index[2]) in str(info.value)
    assert repr(o.columns[1]) in str(info.value)
    above = good * np.nan
    above.iat[0, 0] = 1e4
    with pytest.raises(ValueError, match='outside the odor-evoked rates'):
        fit_mb_model(o, pn_rates=above, **kws)

    # olfsysm's own checks, on the matrix directly
    _, _, _, p = fit_mb_model(o, return_olfsysm_vars=True, **kws)
    mp, rv = p['mp'], p['rv']
    n_gloms, n_odors = np.asarray(mp.orn.data.delta).shape
    for shape in ((n_gloms - 1, n_odors), (n_gloms, n_odors + 1), (1, 1)):
        rv.pn.preset_rates = np.full(shape, np.nan)
        with pytest.raises(ValueError, match='rows|cols'):
            osm.run_PN_sims(mp, rv)
    m = np.full((n_gloms, n_odors), np.nan)
    m[0, 0] = np.inf
    rv.pn.preset_rates = m
    with pytest.raises(ValueError, match='no_inf'):
        osm.run_PN_sims(mp, rv)
    m[0, 0] = np.nan
    rv.pn.preset_rates = m
    mp.pn.noise.sd = 1.0
    with pytest.raises(ValueError, match='noise'):
        osm.run_PN_sims(mp, rv)
    mp.pn.noise.sd = 0.0
    m[1, 1] = -1e4
    rv.pn.preset_rates = m
    with pytest.raises(ValueError, match=r'preset_rates\(1, 1\)'):
        osm.run_PN_sims(mp, rv)
    # (and empty again: the model's own traces)
    rv.pn.preset_rates = np.zeros((0, 0))
    osm.run_PN_sims(mp, rv)
    assert np.array_equal(np.asarray(rv.pn.pn_sims[1]), np.asarray(p0['rv'].pn.pn_sims[1]))


def test_model_id():
    """Model IDs / LR cache keys do not change without the option (absent or None), and
    tell tables apart."""
    kws = dict(weight_divisor=20)
    t1 = pd.DataFrame({'eb @ -3': [10.0, np.nan]}, index=['DL5', 'DM1'])
    t2 = t1 + 1.0
    assert format_model_params({**kws, 'pn_rates': None}) == format_model_params(kws)
    assert onestep_lr_cache_key({**kws, 'pn_rates': None}) == onestep_lr_cache_key(kws)
    ids = {format_model_params(x) for x in (kws, {**kws, 'pn_rates': t1},
        {**kws, 'pn_rates': t2}
    )}
    assert len(ids) == 3
    keys = {onestep_lr_cache_key(x) for x in (kws, {**kws, 'pn_rates': t1},
        {**kws, 'pn_rates': t2}
    )}
    assert len(keys) == 3


if __name__ == '__main__':
    if sys.argv[1:] == ['--freeze']:
        freeze()
    else:
        raise SystemExit('usage: python test/test_pn_preset_rates.py --freeze')
