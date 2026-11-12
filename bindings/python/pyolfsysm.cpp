#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <memory>

#include "olfsysm.hpp"

namespace py = pybind11;

// defined in the generated bindings/python/olfsysm_build_info.cpp (see setup.py)
namespace olfsysm_build_info {
extern const char* const git_hash;
extern const int git_dirty;
extern const int git_untracked;
extern const char* const git_diff;
extern const int cpp_dirty;
extern const char* const cpp_diff;
extern const char* const source_sha256;
extern const char* const build_time;
extern const char* const build_flags;
extern const char* const compiler;
extern const char* const build_host;
}

PYBIND11_MODULE(olfsysm, m) {
	/* TODO fill this in from his other docs */
    m.doc() = R"pbdoc(
        Olfactory system model based on Ann Kennedy's model.
    )pbdoc";

	py::class_<ModelParams>(m, "ModelParams")
        .def_readwrite("orn", &ModelParams::orn)
        .def_readwrite("ln", &ModelParams::ln)
        .def_readwrite("pn", &ModelParams::pn)
        .def_readwrite("kc", &ModelParams::kc)
        .def_readwrite("sim_only", &ModelParams::sim_only)
        .def(py::init([](){
            return std::unique_ptr<ModelParams>(new ModelParams(DEFAULT_PARAMS));
        }))
        .def_property("time_pre_start",
                [](ModelParams const *t){ return t->time.pre_start; },
                [](ModelParams *t, double v){ t->time.pre_start = v; })
        .def_property("time_start",
                [](ModelParams const *t){ return t->time.start; },
                [](ModelParams *t, double v){ t->time.start = v; })
        .def_property("time_end",
                [](ModelParams const *t){ return t->time.end; },
                [](ModelParams *t, double v){ t->time.end = v; })
        .def_property("time_dt",
                [](ModelParams const *t){ return t->time.dt; },
                [](ModelParams *t, double v){ t->time.dt = v; })
        .def_property("time_stim_start",
                [](ModelParams const *t){ return t->time.stim.start; },
                [](ModelParams *t, double v){ t->time.stim.start = v; })
        .def_property("time_stim_end",
                [](ModelParams const *t){ return t->time.stim.end; },
                [](ModelParams *t, double v){ t->time.stim.end = v; });

    py::class_<ModelParams::ORN>(m, "MPORN")
        .def_readwrite("taum", &ModelParams::ORN::taum)
        .def_readwrite("n_physical_gloms", &ModelParams::ORN::n_physical_gloms)
        .def_readwrite("data", &ModelParams::ORN::data);
    py::class_<ModelParams::ORN::Data>(m, "MPORNData")
        .def_readwrite("spont", &ModelParams::ORN::Data::spont)
        .def_readwrite("delta", &ModelParams::ORN::Data::delta);

    py::class_<ModelParams::LN>(m, "MPLN")
        .def_readwrite("taum", &ModelParams::LN::taum)
        .def_readwrite("tauGA", &ModelParams::LN::tauGA)
        .def_readwrite("tauGB", &ModelParams::LN::tauGB)
        .def_readwrite("thr", &ModelParams::LN::thr)
        .def_readwrite("inhsc", &ModelParams::LN::inhsc)
        .def_readwrite("inhadd", &ModelParams::LN::inhadd);

    py::class_<ModelParams::PN>(m, "MPPN")
        .def_readwrite("taum", &ModelParams::PN::taum)
        .def_readwrite("offset", &ModelParams::PN::offset)
        .def_readwrite("tanhsc", &ModelParams::PN::tanhsc)
        .def_readwrite("inhsc", &ModelParams::PN::inhsc)
        .def_readwrite("inhadd", &ModelParams::PN::inhadd)
        .def_readwrite("noise", &ModelParams::PN::noise)
        .def_readwrite("n_total_boutons", &ModelParams::PN::n_total_boutons)
        .def_readwrite("preset_wAPLPN", &ModelParams::PN::preset_wAPLPN)
        .def_readwrite("preset_wPNAPL", &ModelParams::PN::preset_wPNAPL)
        .def_readwrite("wAPLPN_pretune_scale", &ModelParams::PN::wAPLPN_pretune_scale)
        .def_readwrite("wPNAPL_pretune_scale", &ModelParams::PN::wPNAPL_pretune_scale)
        .def_readwrite("bouton_ves_p", &ModelParams::PN::bouton_ves_p)
        .def_readwrite("bouton_tau_r", &ModelParams::PN::bouton_tau_r)
        .def_readwrite("bouton_inh_divisive", &ModelParams::PN::bouton_inh_divisive);

    py::class_<ModelParams::PN::Noise>(m, "MPPNNoise")
        .def_readwrite("mean", &ModelParams::PN::Noise::mean)
        .def_readwrite("sd", &ModelParams::PN::Noise::sd);

    py::class_<ModelParams::FFAPL>(m, "MPFFAPL")
        .def_readwrite("taum", &ModelParams::FFAPL::taum)
        .def_readwrite("w", &ModelParams::FFAPL::w)
        .def_readwrite("zero", &ModelParams::FFAPL::coef)
        .def_readwrite("nneg", &ModelParams::FFAPL::coef)
        .def_readwrite("gini", &ModelParams::FFAPL::gini)
        .def_readwrite("lts", &ModelParams::FFAPL::lts);
    py::class_<ModelParams::FFAPL::Gini>(m, "MPFFAPLGini")
        .def_readwrite("a", &ModelParams::FFAPL::Gini::a)
        .def_readwrite("source", &ModelParams::FFAPL::Gini::source);
    py::class_<ModelParams::FFAPL::LTS>(m, "MPFFAPLLTS")
        .def_readwrite("m", &ModelParams::FFAPL::LTS::m);

    py::class_<ModelParams::KC>(m, "MPKC")
        .def_readwrite("N", &ModelParams::KC::N)
        .def_readwrite("nclaws", &ModelParams::KC::nclaws)
        .def_readwrite("uniform_pns", &ModelParams::KC::uniform_pns)
        .def_readwrite("cxn_distrib", &ModelParams::KC::cxn_distrib)
        .def_readwrite("pn_drop_prop", &ModelParams::KC::pn_drop_prop)
        .def_readwrite("preset_wPNKC", &ModelParams::KC::preset_wPNKC)
        .def_readwrite("seed", &ModelParams::KC::seed)
        .def_readwrite("currents", &ModelParams::KC::currents)
        .def_readwrite("tune_apl_weights", &ModelParams::KC::tune_apl_weights)
        .def_readwrite("preset_wAPLKC", &ModelParams::KC::preset_wAPLKC)
        .def_readwrite("preset_wKCAPL", &ModelParams::KC::preset_wKCAPL)
        .def_readwrite("normalize_apl_weights", &ModelParams::KC::normalize_apl_weights)
        .def_readwrite("apl_weights_kc_mean", &ModelParams::KC::apl_weights_kc_mean)
        .def_readwrite("wAPLKC_pretune_scale", &ModelParams::KC::wAPLKC_pretune_scale)
        .def_readwrite("wKCAPL_pretune_scale", &ModelParams::KC::wKCAPL_pretune_scale)
        .def_readwrite("pn_claw_to_apl", &ModelParams::KC::pn_claw_to_apl)
        .def_readwrite("claw_dynamics", &ModelParams::KC::claw_dynamics)
        .def_readwrite("claw_tau", &ModelParams::KC::claw_tau)
        .def_readwrite("n_apl_units", &ModelParams::KC::n_apl_units)
        .def_readwrite("use_kc_kc_inh", &ModelParams::KC::use_kc_kc_inh)
        .def_readwrite("kc_kc_tau", &ModelParams::KC::kc_kc_tau)
        .def_readwrite("claw_sat", &ModelParams::KC::claw_sat)
        .def_readwrite("claw_sat_hard", &ModelParams::KC::claw_sat_hard)
        .def_readwrite("claw_thr", &ModelParams::KC::claw_thr)
        .def_readwrite("claw_thr_width", &ModelParams::KC::claw_thr_width)
        .def_readwrite("claw_thr_subgain", &ModelParams::KC::claw_thr_subgain)
        .def_readwrite("claw_pow", &ModelParams::KC::claw_pow)
        .def_readwrite("claw_pow_ref", &ModelParams::KC::claw_pow_ref)
        .def_readwrite("ext_inh_scale", &ModelParams::KC::ext_inh_scale)
        .def_readwrite("apl_inh_divisive", &ModelParams::KC::apl_inh_divisive)
        .def_readwrite("refractory_period", &ModelParams::KC::refractory_period)
        .def_readwrite("adapt_inc", &ModelParams::KC::adapt_inc)
        .def_readwrite("adapt_tau", &ModelParams::KC::adapt_tau)
        .def_readwrite("ignore_ffapl", &ModelParams::KC::ignore_ffapl)
        .def_readwrite("fixed_thr", &ModelParams::KC::fixed_thr)
        .def_readwrite("n_claws_active_to_spike", &ModelParams::KC::n_claws_active_to_spike)
        .def_readwrite("add_fixed_thr_to_spont", &ModelParams::KC::add_fixed_thr_to_spont)
        .def_readwrite("use_fixed_thr", &ModelParams::KC::use_fixed_thr)
        .def_readwrite("use_vector_thr", &ModelParams::KC::use_vector_thr)
        .def_readwrite("use_homeostatic_thrs", &ModelParams::KC::use_homeostatic_thrs)
        .def_readwrite("thr_type", &ModelParams::KC::thr_type)
        .def_readwrite("sp_target", &ModelParams::KC::sp_target)
        .def_readwrite("sp_factor_pre_apl", &ModelParams::KC::sp_factor_pre_apl)
        .def_readwrite("thr_sp_acc", &ModelParams::KC::thr_sp_acc)
        .def_readwrite("sp_acc", &ModelParams::KC::sp_acc)
        .def_readwrite("thr_sp_lr_coeff", &ModelParams::KC::thr_sp_lr_coeff)
        .def_readwrite("sp_lr_coeff", &ModelParams::KC::sp_lr_coeff)
        .def_readwrite("linear_lr_falloff", &ModelParams::KC::linear_lr_falloff)
        .def_readwrite("binary_search_on_overshoot", &ModelParams::KC::binary_search_on_overshoot)
        .def_readwrite("n_spikes_for_response", &ModelParams::KC::n_spikes_for_response)
        .def_readwrite("hardcode_initial_sp", &ModelParams::KC::hardcode_initial_sp)
        .def_readwrite("max_iters", &ModelParams::KC::max_iters)
        .def_readwrite("tune_from", &ModelParams::KC::tune_from)
        .def_readwrite("apltune_subsample", &ModelParams::KC::apltune_subsample)
        .def_readwrite("taum", &ModelParams::KC::taum)
        .def_readwrite("apl_taum", &ModelParams::KC::apl_taum)
        .def_readwrite("tau_apl2kc", &ModelParams::KC::tau_apl2kc)
        .def_readwrite("apl_spike_ref_dt", &ModelParams::KC::apl_spike_ref_dt)
        .def_readwrite("tau_r", &ModelParams::KC::tau_r)
        .def_readwrite("ves_p", &ModelParams::KC::ves_p)
        // TODO add one for save_bouton_sims?
        .def_readwrite("save_vm_sims", &ModelParams::KC::save_vm_sims)
        .def_readwrite("save_spike_recordings", &ModelParams::KC::save_spike_recordings)
        .def_readwrite("save_nves_sims", &ModelParams::KC::save_nves_sims)
        .def_readwrite("save_inh_sims", &ModelParams::KC::save_inh_sims)
        .def_readwrite("save_Is_sims", &ModelParams::KC::save_Is_sims)
        .def_readwrite("save_claw_sims", &ModelParams::KC::save_claw_sims)
        .def_readwrite("save_apl_unit_sims", &ModelParams::KC::save_apl_unit_sims)
        .def_readwrite("kc_ids", &ModelParams::KC::kc_ids)
        .def_readwrite("wPNKC_one_row_per_claw", &ModelParams::KC::wPNKC_one_row_per_claw)
        .def_readwrite("allow_net_inh_per_claw", &ModelParams::KC::allow_net_inh_per_claw)
        ;


	/* TODO convert all values in DEFAULT_PARAMS to default kwargs on a python
       constructor */

	py::class_<RunVars>(m, "RunVars")
        // TODO readonly for all initial structs?
        .def_readwrite("orn", &RunVars::orn)
        .def_readwrite("ln", &RunVars::ln)
        .def_readwrite("pn", &RunVars::pn)
        .def_readwrite("ffapl", &RunVars::ffapl)
        .def_readwrite("kc", &RunVars::kc)
        //
        .def_readonly("log", &RunVars::log)
        .def_readwrite("verbose", &RunVars::verbose)
        .def_readonly("ready", &RunVars::ready)
        .def(py::init<ModelParams const&>());

    // TODO (delete. don't think this is an issue, but should prob add a test to
    // establish behavior) w/o explicit disable are there problems w/ things writing to
    // same file sequentially if not? (haven't tested yet.  might want to test multiple
    // sequential calls [seems like it appends just fine] and also what happens if file
    // either never exists in first place, or if python makes the file but then
    // moves/deletes it (does C++ just keep appending to old path? doesn't seem anything
    // blows up.) (if never need explicit, could remove disable from interface here
    // again)
    py::class_<Logger>(m, "RVLogger")
        .def("redirect", py::overload_cast<const std::string &>(&Logger::redirect))
        .def("tee", &Logger::tee)
        .def("tee_off", &Logger::tee_off)
        .def("disable", &Logger::disable)
        .def_readwrite("tee_color", &Logger::tee_color);

    // TODO readonly for all sims here (and in ln/pn/etc)?
    py::class_<RunVars::ORN>(m, "RVORN")
        .def_readonly("sims", &RunVars::ORN::sims);

    py::class_<RunVars::LN>(m, "RVLN")
        .def_readwrite("ln_inhA", &RunVars::LN::inhA)
        .def_readwrite("ln_inhB", &RunVars::LN::inhB);
    py::class_<RunVars::LN::InhA>(m, "RVLNInhA")
        .def_readonly("sims", &RunVars::LN::InhA::sims);
    py::class_<RunVars::LN::InhB>(m, "RVLNInhB")
        .def_readonly("sims", &RunVars::LN::InhB::sims);

    // TODO why pn_sims and not sims here? any reason? change for consistency w/ orn
    // handling?
    py::class_<RunVars::PN>(m, "RVPN")
        .def_readwrite("pn_to_Btns", &RunVars::PN::pn_to_Btns)
        .def_readwrite("Btn_to_pn", &RunVars::PN::Btn_to_pn)
        .def_readwrite("wAPLPN", &RunVars::PN::wAPLPN)
        .def_readwrite("wPNAPL", &RunVars::PN::wPNAPL)
        .def_readwrite("wAPLPN_scale", &RunVars::PN::wAPLPN_scale)
        .def_readwrite("wPNAPL_scale", &RunVars::PN::wPNAPL_scale)
        .def_readonly("wAPLPN_unscaled", &RunVars::PN::wAPLPN_unscaled)
        .def_readonly("wPNAPL_unscaled", &RunVars::PN::wPNAPL_unscaled)
        .def_readwrite("wAPLPN_units", &RunVars::PN::wAPLPN_units)
        .def_readwrite("wPNAPL_units", &RunVars::PN::wPNAPL_units)
        .def_readonly("pn_sims", &RunVars::PN::sims)
        .def_readonly("bouton_sims", &RunVars::PN::bouton_sims)
        .def_readwrite("bouton_ext_inh", &RunVars::PN::bouton_ext_inh);

    py::class_<RunVars::FFAPL>(m, "RVFFAPL")
        .def_readonly("vm_sims", &RunVars::FFAPL::vm_sims)
        .def_readonly("coef_sims", &RunVars::FFAPL::coef_sims);

    py::class_<RunVars::KC>(m, "RVKC")
        .def_readwrite("wPNKC", &RunVars::KC::wPNKC)
        .def_readwrite("wAPLKC", &RunVars::KC::wAPLKC)
        .def_readwrite("wKCAPL", &RunVars::KC::wKCAPL)
        .def_readwrite("wAPLKC_scale", &RunVars::KC::wAPLKC_scale)
        .def_readwrite("wKCAPL_scale", &RunVars::KC::wKCAPL_scale)
        .def_readonly("wAPLKC_unscaled", &RunVars::KC::wAPLKC_unscaled)
        .def_readonly("wKCAPL_unscaled", &RunVars::KC::wKCAPL_unscaled)
        .def_readwrite("wAPLKC_units", &RunVars::KC::wAPLKC_units)
        .def_readwrite("wKCAPL_units", &RunVars::KC::wKCAPL_units)
        .def_readwrite("apl_unit_coupling", &RunVars::KC::apl_unit_coupling)
        .def_readwrite("wKCKC", &RunVars::KC::wKCKC)
        .def_readwrite("ext_inh", &RunVars::KC::ext_inh)
        .def_readonly("pks", &RunVars::KC::pks)
        .def_readwrite("spont_in", &RunVars::KC::spont_in)
        .def_readwrite("thr", &RunVars::KC::thr)
        .def_readonly("responses", &RunVars::KC::responses)
        .def_readonly("spike_counts", &RunVars::KC::spike_counts)
        .def_readonly("vm_sims", &RunVars::KC::vm_sims)
        .def_readonly("spike_recordings", &RunVars::KC::spike_recordings)
        .def_readonly("nves_sims", &RunVars::KC::nves_sims)
        .def_readonly("inh_sims", &RunVars::KC::inh_sims)
        .def_readonly("apl_unit_inh_sims", &RunVars::KC::apl_unit_inh_sims)
        .def_readonly("Is_sims", &RunVars::KC::Is_sims)
        .def_readonly("Is_from_kcs", &RunVars::KC::Is_from_kcs)
        .def_readonly("Is_from_pns", &RunVars::KC::Is_from_pns)
        .def_readonly("claw_sims", &RunVars::KC::claw_sims)
        .def_readwrite("thr_tuning_iters", &RunVars::KC::thr_tuning_iters)
        .def_readwrite("tuning_iters", &RunVars::KC::tuning_iters)
        .def_readwrite("claw_to_kc", &RunVars::KC::claw_to_kc)
        .def_readwrite("kc_to_claws", &RunVars::KC::kc_to_claws)
        .def_readonly("tuning_successful", &RunVars::KC::tuning_successful)
        .def_readonly("thr_sp_lr_coeff_to_tune_in_one_iter", &RunVars::KC::thr_sp_lr_coeff_to_tune_in_one_iter)
        .def_readonly("sp_lr_coeff_to_tune_in_one_iter", &RunVars::KC::sp_lr_coeff_to_tune_in_one_iter)
        ;

    m.def("load_hc_data", &load_hc_data, R"pbdoc(
        Load HC data from file.
    )pbdoc");

    m.def("build_wPNKC", &build_wPNKC, R"pbdoc(
        Choose between the above functions appropriately.
    )pbdoc");

    m.def("fit_sparseness", &fit_sparseness, R"pbdoc(
        Set KC spike thresholds, and tune APL<->KC weights until reaching the
        desired sparsity specific for KC.
    )pbdoc");

    m.def("sim_ORN_layer", &sim_ORN_layer, R"pbdoc(
        Model ORN response for one odor.
    )pbdoc");

    m.def("sim_LN_layer", &sim_LN_layer, R"pbdoc(
        Model LN response for one odor.
    )pbdoc");

    m.def("sim_PN_layer", &sim_PN_layer, R"pbdoc(
        Model PN response for one odor.
    )pbdoc");

    m.def("sim_KC_layer", &sim_KC_layer, R"pbdoc(
        Model KC response for one odor.
    )pbdoc");

    m.def("run_ORN_LN_sims", &run_ORN_LN_sims, R"pbdoc(
        Run ORN and LN sims for all odors.
    )pbdoc");

    m.def("run_PN_sims", &run_PN_sims, R"pbdoc(
        Run PN sims for all odors.
    )pbdoc");

    m.def("run_FFAPL_sims", &run_FFAPL_sims, R"pbdoc(
        Run FFAPL sims for all oors.
    )pbdoc");

    m.def("run_KC_sims", &run_KC_sims, R"pbdoc(
        Regenerate PN->KC connectivity, re-tune thresholds and APL, and run KC sims
        for all odors.
        Connectivity regeneration can be turned off by passing regen=false.
    )pbdoc");

    m.def("remove_all_pretime", &remove_all_pretime, R"pbdoc(
        Frees up memory for all dynamics saved, by deleting all values between
        time.pre_start and time.start (all before odor onset at time.stim.start).
    )pbdoc");

#ifdef VERSION_INFO
    const char* version = VERSION_INFO;
#else
    const char* version = "dev";
#endif
    m.attr("__version__") = version;

    // build provenance (see setup.py): which commit / sources this extension was compiled
    // from, and whether the checkout had uncommitted changes to tracked files
    m.attr("__git_hash__") = olfsysm_build_info::git_hash;
    m.attr("__git_dirty__") = olfsysm_build_info::git_dirty;
    m.attr("__git_untracked__") = olfsysm_build_info::git_untracked;
    m.attr("__git_diff__") = olfsysm_build_info::git_diff;
    m.attr("__cpp_dirty__") = olfsysm_build_info::cpp_dirty;
    m.attr("__cpp_diff__") = olfsysm_build_info::cpp_diff;
    m.attr("__source_sha256__") = olfsysm_build_info::source_sha256;
    m.attr("__build_time__") = olfsysm_build_info::build_time;
    m.attr("__build_flags__") = olfsysm_build_info::build_flags;
    m.attr("__compiler__") = olfsysm_build_info::compiler;
    m.attr("__build_host__") = olfsysm_build_info::build_host;
    m.def("build_info", [version]() {
        py::dict d;
        d["version"] = version;
        d["git_hash"] = olfsysm_build_info::git_hash;
        // -1: not built from a git checkout (unknown); 0: clean; 1: tracked files modified
        d["git_dirty"] = olfsysm_build_info::git_dirty;
        d["git_untracked"] = olfsysm_build_info::git_untracked;
        d["git_diff"] = olfsysm_build_info::git_diff;
        // the same for the compiled C++ sources and headers only (pyolfsysm/build_sources.py),
        // which the extension's code depends on (changes to the Python package do not count)
        d["cpp_dirty"] = olfsysm_build_info::cpp_dirty;
        d["cpp_diff"] = olfsysm_build_info::cpp_diff;
        d["source_sha256"] = olfsysm_build_info::source_sha256;
        d["build_time"] = olfsysm_build_info::build_time;
        d["build_flags"] = olfsysm_build_info::build_flags;
        d["compiler"] = olfsysm_build_info::compiler;
        d["build_host"] = olfsysm_build_info::build_host;
        return d;
    }, R"pbdoc(
        Build provenance baked in at compile time (setup.py): the git commit the extension
        was built from, whether tracked files were modified then (git_dirty: 1 = dirty,
        0 = clean, -1 = not built from a git checkout), the number of untracked files, the
        diff of a dirty build, whether the compiled C++ sources / headers themselves differed
        from that commit (cpp_dirty, same values) and their diff (cpp_diff), a SHA-256 over
        the compiled sources and headers, and the build time, flags, compiler and host.
    )pbdoc");
}

