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
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from pyolfsysm.tuning import APL_OFF, Simulator


# the archive's format, written into its JSON
ARCHIVE_FORMAT = 'pyolfsysm.rebuild/1'
# the fitted parameters (absolute values; `make_fixed_param_kc_simulator`'s keys)
PARAM_KEYS: Tuple[str, ...] = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale',
    'wPNAPL_scale', 'wKCKC_scale'
)
# the unscaled APL weights (`fit_mb_model`'s `_wAPLKC`, `_wKCAPL`, `_wAPLPN`, `_wPNAPL`)
WEIGHT_KEYS: Tuple[str, ...] = ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL')


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
    raise NotImplementedError


def glomerulus_gain(wPNKC: pd.DataFrame, glomeruli: Sequence[str], gain: float
    ) -> pd.DataFrame:
    """Returns `wPNKC` (as float) with the columns of `glomeruli` multiplied by `gain`.

    Args:
        wPNKC: PN->KC wiring (e.g. claw x bouton), with a 'glomerulus' column level.
        glomeruli: the glomeruli whose PN->claw weights are scaled.
        gain: the factor.
    """
    raise NotImplementedError


def scaled_weights(weights: Mapping[str, pd.Series], params: Mapping[str, float]
    ) -> Dict[str, pd.Series]:
    """Returns the absolute APL weights, each unscaled weight times its fitted scale.

    Args:
        weights: {'wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL'} unscaled weights.
        params: the fitted parameters (`PARAM_KEYS`; '<name>_scale' per weight).
    """
    raise NotImplementedError


def model_kwargs(params: Mapping[str, float], wiring: pd.DataFrame,
    weights: Mapping[str, pd.Series], model_kws: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
    """Returns the `fit_mb_model` kwargs of a fitted model at fixed parameters.

    Args:
        params: the fitted parameters (`PARAM_KEYS`, absolute; 'wKCKC_scale' only for a
            model with KC-KC inhibition).
        wiring: the PN->KC wiring as fitted (`_wPNKC`; claw x bouton, gain applied).
        weights: the unscaled APL weights as fitted (`WEIGHT_KEYS`).
        model_kws: the model's other (scalar) `fit_mb_model` kwargs, e.g.
            one_row_per_claw, prat_boutons, claw_thr, kc_kc_tau.
    """
    raise NotImplementedError


def simulate_at(params: Mapping[str, float], wiring: pd.DataFrame,
    weights: Mapping[str, pd.Series], orn_deltas: pd.DataFrame, *,
    model_kws: Optional[Mapping[str, Any]] = None, apl_off: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Returns `fit_mb_model`'s outputs for a fitted model at its parameters.

    One `fit_mb_model` call (the ORN / PN layers and the KCs) on `orn_deltas` with the
    parameters fixed and the wiring and APL weights given.

    Args:
        params, wiring, weights, model_kws: as `model_kwargs`.
        orn_deltas: ORN input, glomeruli x odors (spike deltas, the lab's units).
        apl_off: run with APL's output blocked (the APL->KC and APL->bouton scales at 0,
            `pyolfsysm.tuning.APL_OFF`), as the fits' acute APL-off condition.

    Returns (responses, spike_counts, wPNKC, param_dict), as `fit_mb_model`.
    """
    raise NotImplementedError


def make_simulator(params: Mapping[str, float], wiring: pd.DataFrame,
    weights: Mapping[str, pd.Series], orn_deltas: pd.DataFrame, *,
    model_kws: Optional[Mapping[str, Any]] = None) -> Tuple[Simulator, Dict[str, float]]:
    """Returns a fixed-parameter simulator of a fitted model on `orn_deltas`.

    `make_fixed_param_kc_simulator` built at the fitted parameters: `simulate(p)` re-runs
    only the KCs with any parameters of `p` changed and returns (responses, spike_counts),
    the form `pyolfsysm.tuning.bisect_factor` / `hold_rates` take.

    Args: as `simulate_at`.

    Returns (simulate, params): the simulator and its parameters (the fitted ones).
    """
    raise NotImplementedError


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
        apl_off: tune with APL off (`APL_OFF` over `params`).
        bracket, step: the factors searched and the first step.
        label: printed with each step.

    Returns dict with 'params' (absolute, `params` with the re-tuned threshold; APL
    scales as in `params` even with `apl_off`), 'factor', 'n_responses', 'target_n',
    'rate', 'at_bound', 'history', 'outputs' ((responses, spike_counts) at the factor).
    """
    raise NotImplementedError


def write_archive(path: Union[str, Path], *, params: Mapping[str, float],
    wiring: pd.DataFrame, weights: Mapping[str, pd.Series],
    model_kws: Mapping[str, Any], orn_deltas: Optional[pd.DataFrame] = None,
    outputs: Optional[Mapping[str, Tuple[pd.DataFrame, pd.DataFrame]]] = None,
    breadth: Optional[pd.Series] = None, meta: Optional[Mapping[str, Any]] = None
    ) -> Path:
    """Writes a rebuild archive of a fitted model to the folder `path`.

    Files: rebuild.json (format, params, model_kws, `meta`, and each file's sha256),
    wPNKC.parquet, wAPLKC.parquet, wKCAPL.parquet, wAPLPN.parquet, wPNAPL.parquet, and
    optionally orn_deltas.parquet (the fit's input), breadth.parquet, and the fit's outputs
    (on_responses.parquet, on_spike_counts.parquet, acute_off_responses.parquet,
    acute_off_spike_counts.parquet), which `check_archive` compares a rebuild with.

    Returns the path of rebuild.json.
    """
    raise NotImplementedError


def load_archive(path: Union[str, Path]) -> Dict[str, Any]:
    """Returns a rebuild archive's pieces.

    Args:
        path: the archive's folder (or its rebuild.json).

    Returns dict with 'params', 'wiring', 'weights', 'model_kws', 'orn_deltas' (or None),
    'outputs' ({'on', 'off'}: (responses, spike_counts), or None), 'breadth' (or None),
    'meta' (rebuild.json's contents), 'path'.
    """
    raise NotImplementedError


def simulate_archive(archive: Union[str, Path, Mapping[str, Any]],
    orn_deltas: Optional[pd.DataFrame] = None, *, apl_off: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """`simulate_at` with an archive's pieces (`load_archive`'s dict, or its path), on
    `orn_deltas` (default the archive's own input)."""
    raise NotImplementedError


def check_archive(archive: Union[str, Path, Mapping[str, Any]],
    simulator: bool = True) -> Dict[str, Any]:
    """Returns how far a rebuild from an archive is from the fit's saved outputs.

    Runs `simulate_at` on the archive's own ORN input with APL on and with APL off (and,
    with `simulator`, `make_simulator`'s simulator at the same two points), and counts the
    differing spike counts and responses against the archive's saved outputs.

    Returns dict with, per route ('fit_mb_model', 'simulator') and condition ('on',
    'off'), 'spike_count_diffs' and 'response_diffs' (-1 where the index or columns
    differ), and 'ok' (all zero).
    """
    raise NotImplementedError
