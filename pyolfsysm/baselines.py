"""The baseline models olfdb's baseline-model worker runs, defined in one place.

The bouton base is the KC-remapping experiment queue's (model_experiment_queue) "bouton
base": E13_bouton_coincidence's `bouton_base` (= E3's `BOUTON`, E4's `CLAWS` plus
boutons), the hemibrain claw+bouton model with connectome APL weights, KC and PN
(bouton) input to APL, a response of 2 or more spikes, tuned with APL on to a 5 %
response rate. Its variants replace the connectome's identities (`fit_mb_model`'s
`pn2kc_connections='uniform-glomeruli-boutons'`, `uniform_apl_weights`,
`shuffle_connectome` / `shuffle_seed`), each also with the wiring restricted to
the glomeruli of the ORN input (`input_glomeruli_only`; names ending '_input_glomeruli').

`run_baseline` runs one as E13 did (E3's `tune_and_knockout`): tuned with APL on, then
acutely without APL output (the tuned threshold kept, APL->KC and APL->bouton weights 0,
KC->APL 0 as `fit_mb_model` derives it from APL->KC, bouton->APL at its tuned scale).
"""
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd


# E13_bouton_coincidence's `bouton_base` (provenance.json "base", plus its
# `target_sparsity`; `pn2kc_connections` at its default, 'hemibrain'). E13 also passed
# make_plots=False, silent=True, which change no outputs (`run_baseline` passes them).
BOUTON_BASE_KWS: Dict[str, Any] = dict(
    one_row_per_claw=True,
    prat_claws=True,
    prat_boutons=True,
    use_connectome_APL_weights=True,
    pn_claw_to_apl=True,
    n_spikes_for_response=2,
    target_sparsity=0.05,
)

# name -> `fit_mb_model` kwargs
BASELINE_VARIANTS: Dict[str, Dict[str, Any]] = {
    'bouton_base': dict(BOUTON_BASE_KWS),
    # each claw (7 per KC) from a random bouton of a uniformly drawn glomerulus, and
    # every KC<>APL and PN<>APL weight equal (random claws have no connectome claw
    # identity, so prat_claws=False, as 'uniform-boutons' requires)
    'uniform_glomeruli_claws_uniform_apl': dict(BOUTON_BASE_KWS, prat_claws=False,
        pn2kc_connections='uniform-glomeruli-boutons', n_claws=7, seed=0,
        uniform_apl_weights=True
    ),
    # the connectome's PN->KC, KC<>APL and PN<>APL connections each permuted (one seed)
    'shuffled_connectome_pairs_kept_seed0': dict(BOUTON_BASE_KWS,
        shuffle_connectome='pairs_kept', shuffle_seed=0
    ),
    'shuffled_connectome_pairs_broken_seed0': dict(BOUTON_BASE_KWS,
        shuffle_connectome='pairs_broken', shuffle_seed=0
    ),
}
# each with the wiring restricted to the glomeruli of the ORN input it is run on
# (`fit_mb_model`'s `input_glomeruli_only`), as '<name>_input_glomeruli'
BASELINE_VARIANTS.update({f'{name}_input_glomeruli': dict(kws, input_glomeruli_only=True)
    for name, kws in list(BASELINE_VARIANTS.items())
})


def baseline_kws(name: str, **overrides) -> Dict[str, Any]:
    """Returns the `fit_mb_model` kwargs of baseline `name`, updated with `overrides`.
    """
    if name not in BASELINE_VARIANTS:
        raise ValueError(f'unknown baseline {name!r} (one of {list(BASELINE_VARIANTS)})')
    return dict(BASELINE_VARIANTS[name], **overrides)


def run_baseline(name: str, orn_deltas: pd.DataFrame, **overrides) -> Dict[str, Any]:
    """Runs baseline `name` with APL on (tuned) and acutely with APL output off.

    Args:
        name: a key of `BASELINE_VARIANTS`
        orn_deltas: ORN input (glomeruli x odors), as for `fit_mb_model`
        **overrides: update the baseline's `fit_mb_model` kwargs (e.g. `seed`,
            `shuffle_seed`, `target_sparsity`, `plot_dir`)

    Returns dict with 'kws' (the APL-on call's kwargs, without make_plots / silent),
    'off_kws' (the APL-off call's), 'responses_on', 'spike_counts_on', 'responses_off',
    'spike_counts_off' (KCs x odors), 'wPNKC', 'params' (the APL-on `param_dict`) and
    'params_off'. `save_baseline_outputs` writes the matrices as E13 did.
    """
    from pyolfsysm.mb_model import fit_mb_model

    kws = baseline_kws(name, **overrides)
    common = dict(make_plots=False, silent=True)
    resp_on, spk_on, wPNKC, params = fit_mb_model(orn_deltas, **kws, **common)

    off_kws = {k: v for k, v in kws.items() if k != 'target_sparsity'}
    off_kws.update(fixed_thr=params['fixed_thr'], wAPLKC=0.0)
    if kws.get('prat_boutons'):
        off_kws.update(wAPLPN=0.0,
            wPNAPL=float(params.get('wPNAPL_scale', params.get('wPNAPL', 0.0)))
        )
    resp_off, spk_off, _, params_off = fit_mb_model(orn_deltas, **off_kws, **common)
    return dict(kws=kws, off_kws=off_kws, responses_on=resp_on, spike_counts_on=spk_on,
        responses_off=resp_off, spike_counts_off=spk_off, wPNKC=wPNKC, params=params,
        params_off=params_off
    )


def save_baseline_outputs(result: Dict[str, Any], out_dir: Union[str, Path], tag: str
    ) -> None:
    """Writes `<tag>_{on,acute_off}_{responses,spike_counts}.parquet` (as E13 did).

    Args:
        result: from `run_baseline`
        out_dir: an existing directory
        tag: file name prefix (E13: 'bouton_base')
    """
    out_dir = Path(out_dir)
    for cond, suffix in (('on', 'on'), ('acute_off', 'off')):
        result[f'responses_{suffix}'].to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
        result[f'spike_counts_{suffix}'].to_parquet(
            out_dir / f'{tag}_{cond}_spike_counts.parquet'
        )
