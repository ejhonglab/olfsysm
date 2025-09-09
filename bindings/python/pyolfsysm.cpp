#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <memory>

/*
#include <string>
#include <vector>
#include <exception>
*/
#include "olfsysm.hpp"

namespace py = pybind11;

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
        .def_readwrite("noise", &ModelParams::PN::noise);
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
        .def_readwrite("ignore_ffapl", &ModelParams::KC::ignore_ffapl)
        .def_readwrite("fixed_thr", &ModelParams::KC::fixed_thr)
        .def_readwrite("add_fixed_thr_to_spont", &ModelParams::KC::add_fixed_thr_to_spont)
        .def_readwrite("use_fixed_thr", &ModelParams::KC::use_fixed_thr)
        .def_readwrite("use_vector_thr", &ModelParams::KC::use_vector_thr)
        .def_readwrite("use_homeostatic_thrs", &ModelParams::KC::use_homeostatic_thrs)
        .def_readwrite("thr_type", &ModelParams::KC::thr_type)
        .def_readwrite("sp_target", &ModelParams::KC::sp_target)
        .def_readwrite("sp_factor_pre_APL", &ModelParams::KC::sp_factor_pre_APL)
        .def_readwrite("sp_acc", &ModelParams::KC::sp_acc)
        .def_readwrite("sp_lr_coeff", &ModelParams::KC::sp_lr_coeff)
        .def_readwrite("sp_lr_coeff_cl", &ModelParams::KC::sp_lr_coeff_cl)
        .def_readwrite("max_iters", &ModelParams::KC::max_iters)
        .def_readwrite("tune_from", &ModelParams::KC::tune_from)
        .def_readwrite("apltune_subsample", &ModelParams::KC::apltune_subsample)
        .def_readwrite("taum", &ModelParams::KC::taum)
        .def_readwrite("apl_taum", &ModelParams::KC::apl_taum)
        .def_readwrite("tau_apl2kc", &ModelParams::KC::tau_apl2kc)
        .def_readwrite("tau_r", &ModelParams::KC::tau_r)
        .def_readwrite("ves_p", &ModelParams::KC::ves_p)
        .def_readwrite("save_vm_sims", &ModelParams::KC::save_vm_sims)
        .def_readwrite("save_spike_recordings", &ModelParams::KC::save_spike_recordings)
        .def_readwrite("save_nves_sims", &ModelParams::KC::save_nves_sims)
        .def_readwrite("save_inh_sims", &ModelParams::KC::save_inh_sims)
        .def_readwrite("save_Is_sims", &ModelParams::KC::save_Is_sims)
        // expose the new kc_ids vector
        .def_readwrite("kc_ids", &ModelParams::KC::kc_ids)
        // expose the flag that controls one-row-per-claw behavior
        .def_readwrite("wPNKC_one_row_per_claw", &ModelParams::KC::wPNKC_one_row_per_claw);



	/* TODO convert all values in DEFAULT_PARAMS to default kwargs on a python
       constructor */

	py::class_<RunVars>(m, "RunVars")
        .def_readwrite("orn", &RunVars::orn)
        .def_readwrite("ln", &RunVars::ln)
        .def_readwrite("pn", &RunVars::pn)
        .def_readwrite("ffapl", &RunVars::ffapl)
        .def_readwrite("kc", &RunVars::kc)
        .def_readonly("log", &RunVars::log)
        .def(py::init<ModelParams const&>());

    // TODO also expose 'disable'? cause problems w/ things writing to same file
    // sequentially if not?
    py::class_<Logger>(m, "RVLogger")
        .def("redirect", py::overload_cast<const std::string &>(&Logger::redirect));

    py::class_<RunVars::ORN>(m, "RVORN")
        .def_readwrite("sims", &RunVars::ORN::sims);

    py::class_<RunVars::LN>(m, "RVLN")
        .def_readwrite("ln_inhA", &RunVars::LN::inhA)
        .def_readwrite("ln_inhB", &RunVars::LN::inhB);
    py::class_<RunVars::LN::InhA>(m, "RVLNInhA")
        .def_readwrite("sims", &RunVars::LN::InhA::sims);
    py::class_<RunVars::LN::InhB>(m, "RVLNInhB")
        .def_readwrite("sims", &RunVars::LN::InhB::sims);

    // TODO why pn_sims and not sims here? any reason? change for consistency?
    py::class_<RunVars::PN>(m, "RVPN")
        .def_readwrite("pn_sims", &RunVars::PN::sims);

    py::class_<RunVars::FFAPL>(m, "RVFFAPL")
        .def_readwrite("vm_sims", &RunVars::FFAPL::vm_sims)
        .def_readwrite("coef_sims", &RunVars::FFAPL::coef_sims);

    py::class_<RunVars::KC>(m, "RVKC")
        .def_readwrite("wPNKC", &RunVars::KC::wPNKC)
        .def_readwrite("wAPLKC", &RunVars::KC::wAPLKC)
        .def_readwrite("wKCAPL", &RunVars::KC::wKCAPL)
        .def_readwrite("wAPLKC_scale", &RunVars::KC::wAPLKC_scale)
        .def_readwrite("wKCAPL_scale", &RunVars::KC::wKCAPL_scale)
        .def_readwrite("pks", &RunVars::KC::pks)
        .def_readwrite("spont_in", &RunVars::KC::spont_in)
        .def_readwrite("thr", &RunVars::KC::thr)
        .def_readwrite("responses", &RunVars::KC::responses)
        .def_readwrite("spike_counts", &RunVars::KC::spike_counts)
        .def_readwrite("vm_sims", &RunVars::KC::vm_sims)
        .def_readwrite("spike_recordings", &RunVars::KC::spike_recordings)
        .def_readwrite("nves_sims", &RunVars::KC::nves_sims)
        .def_readwrite("inh_sims", &RunVars::KC::inh_sims)
        .def_readwrite("Is_sims", &RunVars::KC::Is_sims)
        .def_readwrite("tuning_iters", &RunVars::KC::tuning_iters)
        .def_readwrite("claw_to_kc", &RunVars::KC::claw_to_kc)
        .def_readwrite("kc_to_claws", &RunVars::KC::kc_to_claws)
        .def_readwrite("claw_compartments", &RunVars::KC::claw_compartments)
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

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
