library(xptr)

mk_modelparams <- function() {
    .Call(C_mk_modelparams);
}
mk_runvars <- function(mp) {
    if (!is_xpt(mp)) stop("mp must be externalptr");
    .Call(C_mk_runvars, mp);
}

access_mparam <- function(mp, param, val) {
    if (!is_xpt(mp)) stop("mp must be externalptr");
    if (!is.character(param)) stop("param must be string");
    set = !is.null(val);
    .Call(C_access_mparam, mp, param, val, set);
}
set_mparam <- function(mp, param, val) {
    access_mparam(mp, param, val);
    invisible();
}
get_mparam <- function(mp, param) {
    access_mparam(mp, param, NULL);
} 
access_rvar <- function(rv, var, val) {
    if (!is_xpt(rv)) stop("rv must be externalptr");
    if (!is.character(var)) stop("var must be string");
    set = !is.null(val);
    .Call(C_access_rvar, rv, var, val, set);
}
set_rvar <- function(rv, var, val) {
    access_rvar(rv, var, val);
    invisible();
}
get_rvar <- function(rv, var) {
    access_rvar(rv, var, NULL);
}

set_log_dest <- function(rv, dest) {
    if (!is_xpt(rv)) stop("rv must be externalptr");
    if (!is.character(dest)) stop("dest must be string");
    .Call(C_set_log_destf, rv, dest);
    invisible();
}

mprv_funccall <- function(mp, rv, func) {
    if (!is_xpt(mp)) stop("mp must be externalptr");
    if (!is_xpt(rv)) stop("rv must be externalptr");
    .Call(func, mp, rv);
    invisible();
}

load_hc_data <- function(mp, fp) {
    if (!is_xpt(mp)) stop("mp must be externalptr");
    if (!is.character(fp)) stop("fp must be string");
    .Call(C_load_hc_data, mp, fp);
    invisible();
}

build_wPNKC <- function(mp, rv) { mprv_funccall(mp, rv, C_build_wPNKC); }
fit_sparseness <- function(mp, rv) { mprv_funccall(mp, rv, C_fit_sparseness); }
run_ORN_LN_sims <- function(mp, rv) { mprv_funccall(mp, rv, C_run_ORN_LN_sims); }
run_PN_sims <- function(mp, rv) { mprv_funccall(mp, rv, C_run_PN_sims); }

run_KC_sims <- function(mp, rv, regen=TRUE) {
    if (!is_xpt(mp)) stop("mp must be externalptr");
    if (!is_xpt(rv)) stop("rv must be externalptr");
    if (!is.logical(regen)) stop("regen must be logical");
    .Call(C_run_KC_sims, mp, rv, regen);
    invisible();
}
