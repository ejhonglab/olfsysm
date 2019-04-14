## Organization
The main model is implemented in C++ as a static library (.a). This implementation is in `libolfsysm/src/olfsysm.cpp`,
with API declarations in `libolfsysm/api/olfsysm.hpp`. The model is a port of Ann Kennedy's model, with a few modifications.

R bindings to the implementation library are in `bindings/R/olfsysm/`, which is structured as an R package.
The R interface is implemented in `bindings/R/olfsysm/R/olfsysm.R`.
The functions there closely mirror the API functions documented in `libolfsysm/api/olfsysm.hpp`.

## Installation
### R bindings
First install the R packages [xptr](https://cran.r-project.org/web/packages/xptr/index.html)
and [Rcpp](https://cran.r-project.org/web/packages/Rcpp/index.html).

Once these are installed, `make` from the base directory will build everything and install the R package.
If not run with `sudo`, a local R install directory must already exist.

Compilation will fail with `g++` version 5.5.0, but works with `g++` 7.4.0.
If using Ubuntu 16.04, you may need to follow [these instructions](https://gist.github.com/jlblancoc/99521194aba975286c80f93e47966dc5)
to update your `g++`.

### Python bindings
```
pip install pybind11
python setup.py install
```

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
