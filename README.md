A model of the fly olfactory system (ORNs, PNs, the mushroom body's Kenyon cells and APL),
in C++ with R and Python bindings (the `olfsysm` module). The Python package `pyolfsysm`
(in the same distribution) runs the model on ORN data and connectome wiring, tunes it,
and saves, loads and plots its outputs, with the data it needs.

## Organization
The main model is implemented in C++ as a static library (.a). This implementation is in `libolfsysm/src/olfsysm.cpp`,
with API declarations in `libolfsysm/api/olfsysm.hpp`. The model is a port of Ann Kennedy's model, with a few modifications.

R bindings to the implementation library are in `bindings/R/olfsysm/`, which is structured as an R package.
The R interface is implemented in `bindings/R/olfsysm/R/olfsysm.R`.
The functions there closely mirror the API functions documented in `libolfsysm/api/olfsysm.hpp`.

Python bindings are in `bindings/python/` (the compiled `olfsysm` module). `pyolfsysm/` is
the model's Python side (moved from [al_analysis](https://github.com/ejhonglab/al_analysis),
with its history, in 2026-11): `pyolfsysm.mb_model` (`fit_mb_model`, `fit_and_plot_mb_model`,
`model_mb_responses`, the inputs: `megamat_orn_deltas`, `connectome_wPNKC`, ...),
`pyolfsysm.wpnkc_shuffles` (shuffled PN->KC wiring), `pyolfsysm.util` (loaders of the KC
and ORN data the model is compared with), `pyolfsysm.readout` (the model's KC output read as
calcium imaging sees it: compressive readouts, spikes through the indicator kernel,
measurement noise, a per-KC session z-score), `pyolfsysm.kc_stats` (KC population
statistics: response concentration, odors-per-KC classes), `pyolfsysm.effects` (the effect
of a manipulation, e.g. blocking APL, on odor-odor distances, against the data's) and
`pyolfsysm.tuning` (re-tuning threshold and APL scales to hold response rates, populations
of simulators; the last four from the KC-remapping experiment queue,
[model_experiment_queue](https://github.com/ejhonglab/model_experiment_queue), 2026-11-12),
with the data they read in `pyolfsysm/data`
(installed with the package; see its README). `scripts/` has scripts that run the model,
and `test/` the tests of both the C++ code and pyolfsysm.

## Installation
### R bindings
First install the R packages [xptr](https://cran.r-project.org/web/packages/xptr/index.html)
and [Rcpp](https://cran.r-project.org/web/packages/Rcpp/index.html).

Once these are installed, `make` from the base directory will build everything and install the R package.
If not run with `sudo`, a local R install directory must already exist.

Compilation will fail with `g++` version 5.5.0, but works with `g++` 7.4.0.
If using Ubuntu 16.04, you may need to follow [these instructions](https://gist.github.com/jlblancoc/99521194aba975286c80f93e47966dc5)
to update your `g++`.

### Python bindings and pyolfsysm
pyolfsysm needs [hong2p](https://github.com/ejhonglab/hong2p) and
[natmix](https://github.com/ejhonglab/natmix), which are not on PyPI, so install them
first (installing al_analysis with its `model` extra, `pip install -e '.[model]'` in an
al_analysis checkout, does all of this, from GitHub). In a venv (`./venv`):
```
sudo apt install build-essential python3.14-dev
python3.14 -m venv venv
./venv/bin/pip install 'hong2p[all] @ git+https://github.com/ejhonglab/hong2p'
./venv/bin/pip install 'natmix @ git+https://github.com/ejhonglab/natmix'
./venv/bin/pip install -e .
```
This installs the `olfsysm` extension and `pyolfsysm` (with its other dependencies, and
the `plot-dynamics` and `step_model_pn_apl` commands). Python 3.8 (with pandas 1.5) is
also supported.

To see compilation errors, which can be helpful when modify the library code / bindings:
```
pip install -v .
```

Every `pip install` recompiles the whole extension (no reuse of cached objects, so a
header-only edit or a changed flag can never leave stale code in the build) and bakes the
build's provenance into the module:
```python
import olfsysm
olfsysm.build_info()     # dict: git_hash, git_dirty (1 = built with uncommitted changes to
                         # tracked files, 0 = clean, -1 = not built from a git checkout),
                         # git_untracked, git_diff (of a dirty build), source_sha256 (over the
                         # compiled sources and headers), build_time, build_flags, compiler,
                         # build_host
olfsysm.__git_hash__     # the same values are also module attributes (__git_dirty__, ...)
```
`build_info()` also has `cpp_dirty` and `cpp_diff`: the same as `git_dirty` and `git_diff`,
for the compiled C++ sources and headers only (listed in `pyolfsysm/build_sources.py`).

pyolfsysm refuses to run the model (`fit_mb_model`) unless the installed extension is a
clean build of its checkout's C++ sources: built with no uncommitted changes to them
(`cpp_dirty` 0), and with the same `source_sha256` as the C++ sources in the checkout it was
installed from now, which must have no uncommitted C++ changes either
(`pyolfsysm.mb_model.olfsysm_clean_problems()` says what is wrong). Commits that change only
Python files (pyolfsysm, scripts, tests) therefore need no rebuild; their commit, dirty
state and diff are recorded at run time instead. During development, set
`AL_ANALYSIS_ALLOW_DIRTY_OLFSYSM=1` to run anyway (the dirty diff is then recorded in the
output's `model_provenance.json`); entry points that set
`pyolfsysm.mb_model.STRICT_CLEAN_OLFSYSM = True` ignore it. Commit the C++ change, then
`pip install -e .`, before producing outputs that need to be reproducible.

`pyolfsysm.mb_model.model_provenance()` collects this build information, the commit, dirty
state and diff of pyolfsysm (this repository) and hong2p (and of al_analysis, when it ran
the model), the CPU, Python and package versions, and content hashes of the model inputs;
`fit_and_plot_mb_model` writes it as `model_provenance.json` next to each output's
parameter files.

Most recently tested using:
- pip `22.3.1`
- `pybind11==2.10.4`
- Python `3.8.12`
- Ubuntu 20.04

## Example Run
### R
```R
library(olfsysm)

mp <- mk_modelparams()
load_hc_data(mp, "hc_data.csv")
rv <- mk_runvars(mp)

run_ORN_LN_sims(mp, rv)
run_PN_sims(mp, rv)
run_KC_sims(mp, rv, T)

kc_resp <- get_rvar(rv, "kc.responses")
```

### Python
```python3
import olfsysm as osm
mp = osm.ModelParams()
osm.load_hc_data(mp, "hc_data.csv")
rv = osm.RunVars(mp)

osm.run_ORN_LN_sims(mp, rv)
osm.run_PN_sims(mp, rv)
osm.run_KC_sims(mp, rv, True)

kc_resp = rv.kc.responses
```



### Support for writing directly to numpy `.npy` files (optional)

1. Make sure the `cnpy` submodule (under the `libolfsysm/include` directory) is present.
   If you still have not cloned `olfsysm`, or are setting it up from scratch, you can:
   ```
   git clone --recurse-submodules https://github.com/ejhonglab/olfsysm
   ```
   If you have already cloned `olfsysm`, you can try:
   ```
   git submodule update --init --recursive
   ```

2. Follow `cnpy` [README instructions](https://github.com/rogersce/cnpy) to compile and
   install that library. You will need `sudo make install` for the `make install`
   command.

3. Rebuild `olfsysm` (e.g. `pip install -v .`, from root directory, to install Python
   bindings).


## Development

Tests of the C++ model and of pyolfsysm, in `test/` (they were in al_analysis until
2026-11). From the repository root, with the venv that has this package installed
editable (e.g. `../al_analysis/venv`); `QUICK=1` skips most model cases:
```
QUICK=1 ./venv/bin/python -m pytest
```
The characterization tests (`test/test_characterize_model.py`) compare outputs with ones
frozen in `test/test_data/characterization/model/` (regenerate with
`CHARACTERIZE_REGEN=1`). `tox` (`pip install tox tox-uv`) runs the tests under Python 3.8 as
well (`tox -e py38`; see `pyproject.toml`).
