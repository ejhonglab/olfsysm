"""The Python side of the olfsysm mushroom body model.

Wraps the compiled `olfsysm` extension (in the same distribution) to fit and run the
model on ORN data and connectome wiring, and to save, load and plot its outputs:
- `pyolfsysm.mb_model`: the model (`fit_mb_model`, `fit_and_plot_mb_model`,
  `model_mb_responses`, ...), its inputs (`megamat_orn_deltas`, `connectome_wPNKC`, ...) and
  the provenance of its outputs (`model_provenance`).
- `pyolfsysm.wpnkc_shuffles`: shuffled PN->KC wiring.
- `pyolfsysm.util`: loaders of the KC and ORN data the model is compared with or fed, and
  `data_root`, the data shipped with this package (`pyolfsysm/data`).
- `pyolfsysm.readout`: the model's KC output read as calcium imaging sees it (readouts of the
  spike counts and their calibration, spikes through the indicator kernel, measurement noise,
  a per-KC session z-score, odor-odor distances and response fractions of an output).
- `pyolfsysm.kc_stats`: KC population statistics (response concentration, odors-per-KC
  classes, cosine similarities split by responders and classes).
- `pyolfsysm.effects`: the effect of a manipulation (e.g. blocking APL) on odor-odor
  distances, pair by pair, against a measured one.
- `pyolfsysm.tuning`: re-tuning a fixed-parameter model's threshold and APL scales to hold its
  response rates, and running a population of simulators.

Moved from al_analysis (its `mb_model`, `wpnkc_shuffles` and part of `al_util`), 2026-11;
`readout`, `kc_stats`, `effects` and `tuning` from the KC-remapping experiment queue
(model_experiment_queue's kcq.harness and experiments), 2026-11-12.
Importing this package does not import the submodules (`mb_model` is large).
"""
