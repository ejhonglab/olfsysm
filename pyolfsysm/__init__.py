"""The Python side of the olfsysm mushroom body model.

Wraps the compiled `olfsysm` extension (in the same distribution) to fit and run the
model on ORN data and connectome wiring, and to save, load and plot its outputs:
- `pyolfsysm.mb_model`: the model (`fit_mb_model`, `fit_and_plot_mb_model`,
  `model_mb_responses`, ...), its inputs (`megamat_orn_deltas`, `connectome_wPNKC`, ...) and
  the provenance of its outputs (`model_provenance`).
- `pyolfsysm.wpnkc_shuffles`: shuffled PN->KC wiring.
- `pyolfsysm.util`: loaders of the KC and ORN data the model is compared with or fed, and
  `data_root`, the data shipped with this package (`pyolfsysm/data`).

Moved from al_analysis (its `mb_model`, `wpnkc_shuffles` and part of `al_util`), 2026-11.
Importing this package does not import the submodules (`mb_model` is large).
"""
