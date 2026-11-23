"""Rebuilding a fitted mushroom-body model from its saved pieces, and running it on new ORN input.

A fitted model of the KC-remapping experiment queue (model_experiment_queue) is
`fit_mb_model` with fixed parameters (the somatic threshold and the APL / KC-KC weight
scales, as absolute values) on hard-coded weights: the claw x bouton PN->KC wiring
(`_wPNKC`, a glomerulus gain already applied) and the unscaled APL weights (`_wAPLKC`,
breadth-weighted, `_wKCAPL`, `_wAPLPN`, `_wPNAPL`). A rebuild archive holds exactly those
pieces (`write_archive` / `load_archive`; layout in `write_archive`), so that `simulate_at`
can run the model as fitted on any ORN input without the queue's code or data, and
`make_simulator` gives the fixed-parameter simulator that `pyolfsysm.tuning`'s
`bisect_factor` / `hold_rates` (or `tune_threshold` here) re-tune.

Moved from the KC-remapping experiment queue (model_experiment_queue's
experiments/E10_joint_fit.py `breadth_weights`, E27_glomerulus_gain.py `gained`),
2026-11-16; the queue keeps its names as re-exports.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from pyolfsysm.tuning import APL_OFF, Simulator, bisect_factor, n_responses


# the archive's format, written into its JSON
ARCHIVE_FORMAT = 'pyolfsysm.rebuild/1'
ARCHIVE_JSON = 'rebuild.json'
# the fitted parameters (absolute values; `make_fixed_param_kc_simulator`'s keys)
PARAM_KEYS: Tuple[str, ...] = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale',
    'wPNAPL_scale', 'wKCKC_scale', 'wMBC1_scale'
)
# the unscaled APL weights (`fit_mb_model`'s `_wAPLKC`, `_wKCAPL`, `_wAPLPN`, `_wPNAPL`)
WEIGHT_KEYS: Tuple[str, ...] = ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL')
# the conditions of an archive's saved outputs, and their files' prefixes
CONDITIONS: Dict[str, str] = {'on': 'on', 'off': 'acute_off'}
# keys of the archive's JSON that write_archive sets (`meta` may not set them)
_RESERVED_KEYS = ('format', 'params', 'model_kws', 'apl_off', 'response_criterion', 'files',
    'written'
)


def breadth_weights(wAPLKC0: pd.Series, breadth: pd.Series, k: float,
    kc_level: str = 'kc_id') -> pd.Series:
    """Returns APL->KC weights scaled by each KC's breadth.

    Each KC's weight is multiplied by ((breadth + 1) / mean(breadth + 1)) ** k, the mean over
    the KCs of `breadth`; KCs of `wAPLKC0` without a breadth keep their weight.

    Args:
        wAPLKC0: unscaled APL->KC weights (e.g. per claw), with a `kc_level` index level.
        breadth: per-KC breadth (e.g. each KC's total spike count with APL off), indexed by
            KC id.
        k: the breadth exponent.
        kc_level: the index level of `wAPLKC0` holding the KC ids.
    """
    bp1 = breadth + 1.0
    f = ((bp1 / bp1.mean()) ** k).reindex(wAPLKC0.index.get_level_values(kc_level))
    return wAPLKC0 * f.fillna(1.0).to_numpy()


def glomerulus_gain(wPNKC: pd.DataFrame, glomeruli: Sequence[str], gain: float
    ) -> pd.DataFrame:
    """Returns `wPNKC` (as float) with the columns of `glomeruli` multiplied by `gain`.

    Args:
        wPNKC: PN->KC wiring (e.g. claw x bouton), with a 'glomerulus' column level.
        glomeruli: the glomeruli whose PN->claw weights are scaled.
        gain: the factor.
    """
    col = wPNKC.columns.get_level_values('glomerulus').isin(glomeruli)
    out = wPNKC.astype(float).copy()
    out.loc[:, col] = out.loc[:, col] * gain
    return out


def scaled_weights(weights: Mapping[str, pd.Series], params: Mapping[str, float]
    ) -> Dict[str, pd.Series]:
    """Returns the absolute APL weights, each unscaled weight times its fitted scale.

    Args:
        weights: {'wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL'} unscaled weights (the last two
            only for a model with PN<>APL weights).
        params: the fitted parameters (`PARAM_KEYS`; '<name>_scale' per weight).
    """
    return {k: weights[k] * float(params[f'{k}_scale']) for k in WEIGHT_KEYS if k in weights}


def apl_off_overrides(params: Mapping[str, float]) -> Dict[str, float]:
    """Returns the acute APL-off condition's overrides of a model's parameters.

    `pyolfsysm.tuning.APL_OFF` (the APL->KC and APL->bouton scales at 0) restricted to the
    scales the model has (no APL->bouton scale without PN<>APL weights); the KC->APL and
    bouton->APL scales keep their values. `simulate(dict(params, **apl_off_overrides(params)))`
    runs `make_simulator`'s simulator with APL off.
    """
    return {k: v for k, v in APL_OFF.items() if k in params}


def model_kwargs(params: Mapping[str, float], wiring: pd.DataFrame,
    weights: Mapping[str, pd.Series], model_kws: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
    """Returns the `fit_mb_model` kwargs of a fitted model at fixed parameters.

    Args:
        params: the fitted parameters (`PARAM_KEYS`, absolute; 'wKCKC_scale' only for a
            model with KC-KC inhibition, 'wMBC1_scale' only for one with MB-C1 cells, whose
            weights come from `connectome_MBC1_cell_weights` with the MB-C1 kwargs of
            `model_kws`, 'wAPLPN_scale' / 'wPNAPL_scale' only for one with PN<>APL
            weights).
        wiring: the PN->KC wiring as fitted (`_wPNKC`; claw x bouton, gain applied).
        weights: the unscaled APL weights as fitted (`WEIGHT_KEYS`).
        model_kws: the model's other (scalar) `fit_mb_model` kwargs, e.g.
            one_row_per_claw, prat_boutons, claw_thr, kc_kc_tau.
    """
    from pyolfsysm.mb_model import fixed_params_to_fit_mb_model_kws, optimizable_model_params
    kws = dict(model_kws or {})
    clash = set(kws) & (set(optimizable_model_params.values()) | {f'_{k}' for k in
        ('wPNKC',) + WEIGHT_KEYS})
    if clash:
        raise ValueError(f'model_kws must not hold {sorted(clash)} (they come from params, '
            'wiring and weights)'
        )
    missing = {'wAPLKC', 'wKCAPL'} - set(weights)
    if missing:
        raise ValueError(f'weights lack {sorted(missing)}')
    if ('wAPLPN' in weights) != ('wPNAPL' in weights):
        raise ValueError('weights must hold both wAPLPN and wPNAPL, or neither')
    kws.update(fixed_params_to_fit_mb_model_kws({k: float(v) for k, v in params.items()}))
    kws.update(_wPNKC=wiring, _wAPLKC=weights['wAPLKC'], _wKCAPL=weights['wKCAPL'])
    if 'wAPLPN' in weights:
        kws.update(_wAPLPN=weights['wAPLPN'], _wPNAPL=weights['wPNAPL'])
    return kws


def simulate_at(params: Mapping[str, float], wiring: pd.DataFrame,
    weights: Mapping[str, pd.Series], orn_deltas: pd.DataFrame, *,
    model_kws: Optional[Mapping[str, Any]] = None, apl_off: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Returns `fit_mb_model`'s outputs for a fitted model at its parameters.

    One `fit_mb_model` call (the ORN / PN layers and the KCs) on `orn_deltas` with the
    parameters fixed and the wiring and APL weights given.

    Args:
        params, wiring, weights, model_kws: as `model_kwargs`.
        orn_deltas: ORN input, glomeruli x odors (spike deltas, the lab's units; glomeruli
            of the wiring that it lacks get no input).
        apl_off: run with APL's output blocked (the APL->KC and APL->bouton scales at 0,
            `apl_off_overrides`), as the fits' acute APL-off condition.

    Returns (responses, spike_counts, wPNKC, param_dict), as `fit_mb_model`.
    """
    from pyolfsysm.mb_model import fit_mb_model
    p = dict(params)
    if apl_off:
        p.update(apl_off_overrides(p))
    return fit_mb_model(orn_deltas, **model_kwargs(p, wiring, weights, model_kws))


def make_simulator(params: Mapping[str, float], wiring: pd.DataFrame,
    weights: Mapping[str, pd.Series], orn_deltas: pd.DataFrame, *,
    model_kws: Optional[Mapping[str, Any]] = None) -> Tuple[Simulator, Dict[str, float]]:
    """Returns a fixed-parameter simulator of a fitted model on `orn_deltas`.

    `make_fixed_param_kc_simulator` built at the fitted parameters (one `fit_mb_model`
    call): `simulate(p)` re-runs only the KCs with any parameters of `p` changed and
    returns (responses, spike_counts), the form `pyolfsysm.tuning.bisect_factor` /
    `hold_rates` take; `simulate(dict(params, **apl_off_overrides(params)))` is the APL-off
    condition (for `hold_rates`, pass `off_params=apl_off_overrides(params)` and `tied_on`
    without 'wPNAPL_scale' for a model without PN<>APL weights).

    Args: as `simulate_at`.

    Returns (simulate, params): the simulator and its parameters (the fitted ones).
    """
    from pyolfsysm.mb_model import make_fixed_param_kc_simulator, optimizable_model_params
    p = {k: float(v) for k, v in params.items()}
    kws = model_kwargs(p, wiring, weights, model_kws)
    # (the parameters go in as initial_params, not as kwargs)
    for k in p:
        kws.pop(optimizable_model_params[k])
    return make_fixed_param_kc_simulator(orn_deltas, initial_params=p, **kws)


def tune_threshold(simulate: Simulator, params: Mapping[str, float], *,
    target_rate: Optional[float] = None, target_n: Optional[int] = None,
    apl_off: bool = False, bracket: Tuple[float, float] = (0.1, 10.0), step: float = 1.1,
    label: str = 'threshold') -> Dict[str, Any]:
    """Returns the somatic threshold re-tuned so the response rate hits a target.

    `pyolfsysm.tuning.bisect_factor` on 'fixed_thr' (log scale, from factor 1), every other
    parameter at `params`' values.

    Args:
        simulate: from `make_simulator`.
        params: the parameters the threshold factor multiplies.
        target_rate, target_n: the response rate (fraction of KC x odor entries) or the
            number of responses (two or more spikes) to hit; one of the two.
        apl_off: tune with APL off (`apl_off_overrides(params)` over `params`).
        bracket, step: the factors searched and the first step.
        label: printed with each step.

    Returns dict with 'params' (absolute, `params` with the re-tuned threshold; APL
    scales as in `params` even with `apl_off`), 'factor', 'n_responses', 'target_n',
    'rate', 'at_bound', 'history', 'outputs' ((responses, spike_counts) at the factor).
    """
    if (target_rate is None) == (target_n is None):
        raise ValueError('pass one of target_rate and target_n')
    p = {k: float(v) for k, v in params.items()}
    off = apl_off_overrides(p) if apl_off else {}
    cache: Dict[float, tuple] = {}

    def run(f: float) -> tuple:
        if f not in cache:
            cache[f] = simulate(dict(p, fixed_thr=p['fixed_thr'] * f, **off))
        return cache[f]

    size = int(run(1.0)[0].size)
    if target_n is None:
        target_n = int(round(float(target_rate) * size))
    f, out, hist, at_bound = bisect_factor(run, int(target_n), size, bracket, step, label)
    n = n_responses(out[0])
    return dict(params=dict(p, fixed_thr=p['fixed_thr'] * f), factor=float(f),
        n_responses=n, target_n=int(target_n), rate=n / size, at_bound=bool(at_bound),
        history=hist, outputs=out
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _json_scalar(k: str, v: Any) -> Any:
    if isinstance(v, (bool, str)) or v is None:
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    raise TypeError(f'model kwarg {k}={v!r}: only scalars (bool, int, float, str, None) are '
        'archived; arrays go in as wiring / weights'
    )


def _write_frame(df: pd.DataFrame, path: Path) -> dict:
    df.to_parquet(path)
    return dict(file=path.name, sha256=_sha256(path), shape=list(df.shape))


def write_archive(path: Union[str, Path], *, params: Mapping[str, float],
    wiring: pd.DataFrame, weights: Mapping[str, pd.Series],
    model_kws: Mapping[str, Any], orn_deltas: Optional[pd.DataFrame] = None,
    outputs: Optional[Mapping[str, Tuple[pd.DataFrame, pd.DataFrame]]] = None,
    breadth: Optional[pd.Series] = None, meta: Optional[Mapping[str, Any]] = None
    ) -> Path:
    """Writes a rebuild archive of a fitted model to the folder `path`.

    Layout (one folder per fitted model):
    - rebuild.json: 'format' (`ARCHIVE_FORMAT`), 'params' (`PARAM_KEYS` the model has,
      absolute), 'model_kws' (the scalar `fit_mb_model` kwargs), 'apl_off' (the APL-off
      condition's overrides, `apl_off_overrides`), 'response_criterion' (a KC responds to an odor
      with at least 'n_spikes_for_response' spikes in the simulation), 'files' ({piece:
      {'file', 'sha256', 'shape', and 'series_name' for a Series}}), 'written' (when, and
      the olfsysm build), and the keys of `meta` (e.g. the wiring's and the gain's
      description, the fit it comes from).
    - wPNKC.parquet: the PN->KC wiring as fitted (claw x bouton, any gain applied).
    - wAPLKC.parquet, wKCAPL.parquet, wAPLPN.parquet, wPNAPL.parquet: the unscaled APL
      weights as fitted (each a one-column frame named after the weight; absolute weight =
      the column times params['<name>_scale']).
    - orn_deltas.parquet (optional): the ORN input of the fit (glomeruli x odors).
    - breadth.parquet (optional): per-KC breadth behind the APL->KC weights (one column).
    - expected_responses_on.parquet, expected_spike_counts_on.parquet,
      expected_responses_acute_off.parquet, expected_spike_counts_acute_off.parquet
      (optional, with `outputs`): the fit's outputs, KCs x odors, which `check_archive`
      compares a rebuild with (named so that they do not match the experiment folders'
      `<variant>_<condition>_spike_counts.parquet`).

    Args:
        path: the archive's folder (created; files in it are overwritten).
        params, wiring, weights, model_kws: as `model_kwargs`.
        orn_deltas: the fit's ORN input.
        outputs: the fit's outputs, {'on', 'off'}: (responses, spike_counts).
        breadth: the per-KC breadth.
        meta: further JSON-serializable keys for rebuild.json (not `_RESERVED_KEYS`).

    Returns the path of rebuild.json.
    """
    from pyolfsysm.mb_model import olfsysm_build_info
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    clash = set(meta) & set(_RESERVED_KEYS)
    if clash:
        raise ValueError(f'meta must not set {sorted(clash)}')
    p = {k: float(v) for k, v in params.items()}
    unknown = set(p) - set(PARAM_KEYS)
    if unknown:
        raise ValueError(f'unknown parameters {sorted(unknown)}')
    mkws = {k: _json_scalar(k, v) for k, v in sorted(model_kws.items())}
    # (fails here, not at load time, on a clash with params / wiring / weights)
    model_kwargs(p, wiring, weights, mkws)

    files: Dict[str, dict] = {'wPNKC': _write_frame(wiring, d / 'wPNKC.parquet')}
    for k in WEIGHT_KEYS:
        if k in weights:
            s = weights[k]
            files[k] = dict(_write_frame(s.to_frame(name=k), d / f'{k}.parquet'),
                series_name=None if s.name is None else str(s.name))
    if orn_deltas is not None:
        files['orn_deltas'] = _write_frame(orn_deltas, d / 'orn_deltas.parquet')
    if breadth is not None:
        files['breadth'] = dict(_write_frame(breadth.to_frame(name='breadth'),
            d / 'breadth.parquet'), series_name=None if breadth.name is None else
            str(breadth.name))
    if outputs is not None:
        if set(outputs) != set(CONDITIONS):
            raise ValueError(f'outputs must have the conditions {sorted(CONDITIONS)}')
        for c, prefix in CONDITIONS.items():
            r, s = outputs[c]
            for kind, df in (('responses', r), ('spike_counts', s)):
                name = f'expected_{kind}_{prefix}'
                files[name] = _write_frame(df, d / f'{name}.parquet')
    bi = olfsysm_build_info()
    rec = dict(format=ARCHIVE_FORMAT, params=p, model_kws=mkws, apl_off=apl_off_overrides(p),
        response_criterion=dict(n_spikes_for_response=mkws.get('n_spikes_for_response'),
            meaning='a KC responds to an odor with at least n_spikes_for_response spikes in '
            'the simulation (None: fit_mb_model\'s default)'),
        files=files,
        written=dict(time=datetime.now(timezone.utc).isoformat(timespec='seconds'),
            olfsysm_build={k: bi.get(k) for k in ('git_hash', 'git_dirty', 'cpp_dirty',
                'source_sha256', 'build_time')}),
        **meta
    )
    out = d / ARCHIVE_JSON
    out.write_text(json.dumps(rec, indent=1, default=str) + '\n')
    return out


def _archive_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    return p.parent if p.name == ARCHIVE_JSON else p


def load_archive(path: Union[str, Path], verify: bool = True) -> Dict[str, Any]:
    """Returns a rebuild archive's pieces.

    Args:
        path: the archive's folder (or its rebuild.json).
        verify: check each file's sha256 against rebuild.json (raises ValueError if one
            differs).

    Returns dict with 'params', 'wiring', 'weights', 'model_kws', 'orn_deltas' (or None),
    'outputs' ({'on', 'off'}: (responses, spike_counts), or None), 'breadth' (or None),
    'meta' (rebuild.json's contents), 'path' (the folder).
    """
    d = _archive_dir(path)
    meta = json.loads((d / ARCHIVE_JSON).read_text())
    if meta.get('format') != ARCHIVE_FORMAT:
        raise ValueError(f'{d}: format {meta.get("format")!r}, not {ARCHIVE_FORMAT!r}')
    files = meta['files']

    def read(name: str) -> Optional[pd.DataFrame]:
        if name not in files:
            return None
        f = d / files[name]['file']
        if verify:
            h = _sha256(f)
            if h != files[name]['sha256']:
                raise ValueError(f'{f}: sha256 {h} differs from {ARCHIVE_JSON}\'s')
        return pd.read_parquet(f)

    def series(name: str) -> Optional[pd.Series]:
        df = read(name)
        return None if df is None else df.iloc[:, 0].rename(files[name].get('series_name'))

    outputs = None
    if all(f'expected_{kind}_{p}' in files for p in CONDITIONS.values()
            for kind in ('responses', 'spike_counts')):
        outputs = {c: (read(f'expected_responses_{p}'), read(f'expected_spike_counts_{p}'))
            for c, p in CONDITIONS.items()}
    return dict(params={k: float(v) for k, v in meta['params'].items()}, wiring=read('wPNKC'),
        weights={k: series(k) for k in WEIGHT_KEYS if k in files},
        model_kws=dict(meta['model_kws']), orn_deltas=read('orn_deltas'), outputs=outputs,
        breadth=series('breadth'), meta=meta, path=d
    )


def _as_archive(archive: Union[str, Path, Mapping[str, Any]]) -> Mapping[str, Any]:
    return load_archive(archive) if isinstance(archive, (str, Path)) else archive


def simulate_archive(archive: Union[str, Path, Mapping[str, Any]],
    orn_deltas: Optional[pd.DataFrame] = None, *, apl_off: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """`simulate_at` with an archive's pieces (`load_archive`'s dict, or its path), on
    `orn_deltas` (default the archive's own input)."""
    a = _as_archive(archive)
    orn = a['orn_deltas'] if orn_deltas is None else orn_deltas
    if orn is None:
        raise ValueError('the archive has no ORN input: pass orn_deltas')
    return simulate_at(a['params'], a['wiring'], a['weights'], orn,
        model_kws=a['model_kws'], apl_off=apl_off)


def _n_diffs(a: pd.DataFrame, b: pd.DataFrame) -> int:
    if not (a.index.equals(b.index) and a.columns.equals(b.columns)):
        return -1
    return int((a.to_numpy(dtype=float) != b.to_numpy(dtype=float)).sum())


def check_archive(archive: Union[str, Path, Mapping[str, Any]],
    simulator: bool = True, record: bool = False) -> Dict[str, Any]:
    """Returns how far a rebuild from an archive is from the fit's saved outputs.

    Runs `simulate_at` on the archive's own ORN input with APL on and with APL off (and,
    with `simulator`, `make_simulator`'s simulator at the same two points), and counts the
    differing spike counts and responses against the archive's saved outputs.

    Args:
        archive: `load_archive`'s dict, or the archive's path.
        simulator: also check the fixed-parameter simulator.
        record: append the result to rebuild.json's 'checks' list.

    Returns dict with, per route ('fit_mb_model', 'simulator') and condition ('on',
    'off'), 'spike_count_diffs' and 'response_diffs' (-1 where the index or columns
    differ), 'n_entries' (KCs x odors), 'ok' (all zero), 'time' and 'olfsysm_build'.
    """
    from pyolfsysm.mb_model import olfsysm_build_info
    a = _as_archive(archive)
    if a['outputs'] is None or a['orn_deltas'] is None:
        raise ValueError('the archive has no saved outputs or ORN input to check against')
    routes: Dict[str, Dict[str, tuple]] = {'fit_mb_model': {}}
    for c in CONDITIONS:
        r, s, _, _ = simulate_archive(a, apl_off=(c == 'off'))
        routes['fit_mb_model'][c] = (r, s)
    if simulator:
        sim, p = make_simulator(a['params'], a['wiring'], a['weights'], a['orn_deltas'],
            model_kws=a['model_kws'])
        routes['simulator'] = {'on': sim(p), 'off': sim(dict(p, **apl_off_overrides(p)))}
        del sim
    res: Dict[str, Any] = {}
    ok = True
    for route, outs in routes.items():
        res[route] = {}
        for c, (r, s) in outs.items():
            rs, ss = a['outputs'][c]
            nd = dict(spike_count_diffs=_n_diffs(s, ss), response_diffs=_n_diffs(r, rs),
                n_entries=int(ss.size))
            res[route][c] = nd
            ok = ok and nd['spike_count_diffs'] == 0 and nd['response_diffs'] == 0
    bi = olfsysm_build_info()
    res.update(ok=ok, time=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        olfsysm_build={k: bi.get(k) for k in ('git_hash', 'git_dirty', 'cpp_dirty',
            'source_sha256')})
    if record:
        if 'path' not in a:
            raise ValueError('record=True needs an archive loaded from a path')
        f = Path(a['path']) / ARCHIVE_JSON
        meta = json.loads(f.read_text())
        meta.setdefault('checks', []).append(res)
        f.write_text(json.dumps(meta, indent=1, default=str) + '\n')
    return res
