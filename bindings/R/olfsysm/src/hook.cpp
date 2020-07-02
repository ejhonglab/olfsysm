#include "olfsysm.hpp"
#include <string>
#include <vector>
#include <exception>

#include <RcppCommon.h>

namespace Rcpp {
    template<> ::Matrix as(SEXP);
    template<> SEXP wrap(::Matrix const&);

    template<> SEXP wrap(std::vector<::Matrix> const&);

    template<> Eigen::VectorXd as(SEXP);
    template<> SEXP wrap(Eigen::VectorXd const&);
}

#include <Rcpp.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#define DEFFROM_AS(t, v, s) t v = Rcpp::as<t>(s)
#define TRYFWD(code) \
    try { \
        code \
    } \
    catch (std::exception const& ex) { \
        forward_exception_to_r(ex); \
    } \
    catch(...) { \
        ::Rf_error("c++ exception"); \
    }

template<class A, class B> B copymtxbyel(A const& mtx) {
    B ret(mtx.rows(), mtx.cols());
    for (int i = 0; i < mtx.rows(); i++) {
        for (int j = 0; j < mtx.cols(); j++) {
            ret(i,j) = mtx(i,j);
        }
    }
    return ret;
}
template<> Matrix Rcpp::as(SEXP s) {
    return copymtxbyel<Rcpp::NumericMatrix, ::Matrix>
        (Rcpp::as<Rcpp::NumericMatrix>(s));
}
template<> SEXP Rcpp::wrap(::Matrix const& mtx) {
    return Rcpp::wrap(copymtxbyel<::Matrix, Rcpp::NumericMatrix>(mtx));
}
template<> SEXP Rcpp::wrap(std::vector<::Matrix> const& v) {
    Rcpp::List ret;
    for (::Matrix const& m : v) {
        ret.push_back(Rcpp::wrap<::Matrix>(m));
    }
    return Rcpp::wrap<Rcpp::List>(ret);
}
template<> Eigen::VectorXd Rcpp::as(SEXP s) {
    Rcpp::NumericVector src = Rcpp::as<Rcpp::NumericVector>(s);
    Eigen::VectorXd dst;
    dst.resize(src.size());
    for (unsigned i = 0; i < src.size(); i++) {
        dst[i] = src[i];
    }
    return dst;
}
template<> SEXP Rcpp::wrap(Eigen::VectorXd const& src) {
    Rcpp::NumericVector dst(src.size());
    for (unsigned i = 0; i < src.size(); i++) {
        dst[i] = src[i];
    }
    return Rcpp::wrap<Rcpp::NumericVector>(dst);
}

extern "C" SEXP mk_modelparams() { TRYFWD (
    Rcpp::XPtr<ModelParams> p(new ModelParams(DEFAULT_PARAMS), true);
    return Rcpp::wrap(p);
)}
extern "C" SEXP mk_runvars(SEXP mp_) { TRYFWD (
    DEFFROM_AS(Rcpp::XPtr<ModelParams>, mp, mp_);
    Rcpp::XPtr<RunVars> rv(new RunVars(*mp), true);
    return Rcpp::wrap(rv);
)}

#ifdef ACCESS
    #error ACCESS macro already defined!
#endif
#define ACCESS(n, p) \
    if (name == n) { \
        if (set) { \
            p = Rcpp::as<decltype(p)>(val); \
        } \
        return Rcpp::wrap<decltype(p)>(p); \
    }
extern "C" SEXP access_mparam(
        SEXP mp_,
        SEXP name_,
        SEXP val, SEXP set_) { TRYFWD (
    DEFFROM_AS(Rcpp::XPtr<ModelParams>, mp, mp_);
    DEFFROM_AS(std::string, name, name_);
    DEFFROM_AS(bool, set, set_);

    ACCESS("time.pre_start",           mp->time.pre_start);
    ACCESS("time.start",               mp->time.start);
    ACCESS("time.end",                 mp->time.end);
    ACCESS("time.stim.start",          mp->time.stim.start);
    ACCESS("time.stim.end",            mp->time.stim.end);
    ACCESS("time.dt",                  mp->time.dt);
    ACCESS("orn.taum",                 mp->orn.taum);
    ACCESS("orn.n_physical_gloms",     mp->orn.n_physical_gloms);
    ACCESS("orn.data.spont",           mp->orn.data.spont);
    ACCESS("orn.data.delta",           mp->orn.data.delta);
    ACCESS("ln.taum",                  mp->ln.taum);
    ACCESS("ln.tauGA",                 mp->ln.tauGA);
    ACCESS("ln.tauGB",                 mp->ln.tauGB);
    ACCESS("ln.thr",                   mp->ln.thr);
    ACCESS("ln.inhsc",                 mp->ln.inhsc);
    ACCESS("ln.inhadd",                mp->ln.inhadd);
    ACCESS("pn.taum",                  mp->pn.taum);
    ACCESS("pn.offset",                mp->pn.offset);
    ACCESS("pn.tanhsc",                mp->pn.tanhsc);
    ACCESS("pn.inhsc",                 mp->pn.inhsc);
    ACCESS("pn.inhadd",                mp->pn.inhadd);
    ACCESS("pn.noise.mean",            mp->pn.noise.mean);
    ACCESS("pn.noise.sd",              mp->pn.noise.sd);
    ACCESS("ffapl.taum",               mp->ffapl.taum);
    ACCESS("ffapl.w",                  mp->ffapl.w);
    ACCESS("ffapl.coef",               mp->ffapl.coef);
    ACCESS("ffapl.gini.a",             mp->ffapl.gini.a);
    ACCESS("ffapl.gini.source",        mp->ffapl.gini.source);
    ACCESS("ffapl.lts.m",              mp->ffapl.lts.m);
    ACCESS("kc.N",                     mp->kc.N);
    ACCESS("kc.nclaws",                mp->kc.nclaws);
    ACCESS("kc.uniform_pns",           mp->kc.uniform_pns);
    ACCESS("kc.cxn_distrib",           mp->kc.cxn_distrib);
    ACCESS("kc.preset_wPNKC",          mp->kc.preset_wPNKC);
    ACCESS("kc.seed",                  mp->kc.seed);
    ACCESS("kc.currents",              mp->kc.currents);
    ACCESS("kc.enable_apl",            mp->kc.enable_apl);
    ACCESS("kc.ignore_ffapl",          mp->kc.ignore_ffapl);
    ACCESS("kc.fixed_thr",             mp->kc.fixed_thr);
    ACCESS("kc.use_fixed_thr",         mp->kc.use_fixed_thr);
    ACCESS("kc.use_homeostatic_thrs",  mp->kc.use_homeostatic_thrs);
    ACCESS("kc.thr_type",              mp->kc.thr_type);
    ACCESS("kc.sp_target",             mp->kc.sp_target);
    ACCESS("kc.sp_acc",                mp->kc.sp_acc);
    ACCESS("kc.sp_lr_coeff",           mp->kc.sp_lr_coeff);
    ACCESS("kc.max_iters",             mp->kc.max_iters);
    ACCESS("kc.tune_from",             mp->kc.tune_from);
    ACCESS("kc.apltune_subsample",     mp->kc.apltune_subsample);
    ACCESS("kc.taum",                  mp->kc.taum);
    ACCESS("kc.apl_taum",              mp->kc.apl_taum);
    ACCESS("kc.tau_apl2kc",            mp->kc.tau_apl2kc);
    ACCESS("kc.tau_r",                 mp->kc.tau_r);
    ACCESS("kc.ves_p",                 mp->kc.ves_p);
    ACCESS("kc.save_vm_sims",          mp->kc.save_vm_sims);
    ACCESS("kc.save_spike_recordings", mp->kc.save_spike_recordings);
    ACCESS("kc.save_nves_sims",        mp->kc.save_nves_sims);
    ACCESS("kc.save_inh_sims",         mp->kc.save_inh_sims);
    ACCESS("kc.save_Is_sims",          mp->kc.save_Is_sims);
    ACCESS("sim_only",                 mp->sim_only);

    Rcpp::stop(std::string("invalid model parameter: ") + name);
    return R_NilValue;
)}

extern "C" SEXP access_rvar(
        SEXP rv_,
        SEXP name_,
        SEXP val, SEXP set_) { TRYFWD (
    DEFFROM_AS(Rcpp::XPtr<RunVars>, rv, rv_);
    DEFFROM_AS(std::string, name, name_);
    DEFFROM_AS(bool, set, set_);

    ACCESS("orn.sims",            rv->orn.sims);
    ACCESS("ln.inhA.sims",        rv->ln.inhA.sims);
    ACCESS("ln.inhB.sims",        rv->ln.inhB.sims);
    ACCESS("pn.sims",             rv->pn.sims);
    ACCESS("ffapl.vm_sims",       rv->ffapl.vm_sims);
    ACCESS("ffapl.coef_sims",     rv->ffapl.coef_sims);
    ACCESS("kc.wPNKC",            rv->kc.wPNKC);
    ACCESS("kc.wAPLKC",           rv->kc.wAPLKC);
    ACCESS("kc.wKCAPL",           rv->kc.wKCAPL);
    ACCESS("kc.thr",              rv->kc.thr);
    ACCESS("kc.responses",        rv->kc.responses);
    ACCESS("kc.spike_counts",     rv->kc.spike_counts);
    ACCESS("kc.vm_sims",          rv->kc.vm_sims);
    ACCESS("kc.spike_recordings", rv->kc.spike_recordings);
    ACCESS("kc.nves_sims",        rv->kc.nves_sims);
    ACCESS("kc.inh_sims",         rv->kc.inh_sims);
    ACCESS("kc.Is_sims",          rv->kc.Is_sims);
    ACCESS("kc.tuning_iters",     rv->kc.tuning_iters);

    Rcpp::stop(std::string("invalid run variable: ") + name);
    return R_NilValue;
)}

extern "C" SEXP set_log_destf(
        SEXP rv_,
        SEXP path_) { TRYFWD (
    DEFFROM_AS(Rcpp::XPtr<RunVars>, rv, rv_);
    DEFFROM_AS(std::string, path, path_);
    rv->log.redirect(path);
    return R_NilValue;
)}

extern "C" SEXP EXPORT_load_hc_data(
        SEXP mp_,
        SEXP path_) { TRYFWD (
    DEFFROM_AS(Rcpp::XPtr<ModelParams>, mp, mp_);
    DEFFROM_AS(std::string, fpath, path_);
    load_hc_data(*mp, fpath);
    return R_NilValue;
)}

#define MP_RV_FUNC(wrapped) \
extern "C" SEXP EXPORT_##wrapped(SEXP mp_, SEXP rv_) { TRYFWD ( \
    DEFFROM_AS(Rcpp::XPtr<ModelParams>, mp, mp_); \
    DEFFROM_AS(Rcpp::XPtr<RunVars>, rv, rv_); \
    wrapped(*mp, *rv); \
    return R_NilValue; \
)}

MP_RV_FUNC(build_wPNKC);
MP_RV_FUNC(fit_sparseness);
MP_RV_FUNC(run_ORN_LN_sims);
MP_RV_FUNC(run_PN_sims);
MP_RV_FUNC(run_FFAPL_sims);

extern "C" SEXP EXPORT_run_KC_sims(SEXP mp_, SEXP rv_, SEXP regen_) { TRYFWD (
    DEFFROM_AS(Rcpp::XPtr<ModelParams>, mp, mp_);
    DEFFROM_AS(Rcpp::XPtr<RunVars>, rv, rv_);
    DEFFROM_AS(bool, regen, regen_);
    run_KC_sims(*mp, *rv, regen);
    return R_NilValue;
)}

extern "C" const R_CallMethodDef CallEntries[13] = {
    {"mk_modelparams", (DL_FUNC) &mk_modelparams, 0},
    {"mk_runvars", (DL_FUNC) &mk_runvars, 1},
    {"access_mparam", (DL_FUNC) &access_mparam, 4},
    {"access_rvar", (DL_FUNC) &access_rvar, 4},
    {"set_log_destf", (DL_FUNC) &set_log_destf, 2},
    {"load_hc_data", (DL_FUNC) &EXPORT_load_hc_data, 2},
    {"build_wPNKC", (DL_FUNC) &EXPORT_build_wPNKC, 2},
    {"fit_sparseness", (DL_FUNC) &EXPORT_fit_sparseness, 2},
    {"run_ORN_LN_sims", (DL_FUNC) &EXPORT_run_ORN_LN_sims, 2},
    {"run_PN_sims", (DL_FUNC) &EXPORT_run_PN_sims, 2},
    {"run_FFAPL_sims", (DL_FUNC) &EXPORT_run_FFAPL_sims, 2},
    {"run_KC_sims", (DL_FUNC) &EXPORT_run_KC_sims, 3},
    {NULL, NULL, 0}
};
extern "C" void R_init_olfsysm(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
