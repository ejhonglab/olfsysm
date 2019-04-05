## Organization
The main model is implemented in C++ as a static library (.a). This implementation is in `libolfsysm/src/olfsysm.cpp`,
with API declarations in `libolfsysm/api/olfsysm.hpp`. The model is a port of Ann Kennedy's model, with a few modifications.

R bindings to the implementation library are in `bindings/R/olfsysm/`, which is structured as an R package.
The R interface is implemented in `bindings/R/olfsysm/R/olfsysm.R`.
The functions there closely mirror the API functions documented in `libolfsysm/api/olfsysm.hpp`.

## Installation
First install the R packages [xptr](https://cran.r-project.org/web/packages/xptr/index.html)
and [Rcpp](https://cran.r-project.org/web/packages/Rcpp/index.html).

Once these are installed, `make` from the base directory will build everything and install the R package.
If not run with `sudo`, a local R install directory must already exist.

## Example Run (R Code)
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
