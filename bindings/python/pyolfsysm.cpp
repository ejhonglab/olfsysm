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
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: python_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

	py::class_<ModelParams>(m, "ModelParams")
        .def_readwrite("orn", &ModelParams::orn)
        .def_readwrite("ln", &ModelParams::ln)
        .def_readwrite("kc", &ModelParams::kc)
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

    py::class_<ModelParams::KC>(m, "MPKC")
        .def_readwrite("N", &ModelParams::KC::N)
        .def_readwrite("nclaws", &ModelParams::KC::nclaws)
        .def_readwrite("uniform_pns", &ModelParams::KC::uniform_pns)
        .def_readwrite("cxn_distrib", &ModelParams::KC::cxn_distrib)
        .def_readwrite("enable_apl", &ModelParams::KC::enable_apl)
        .def_readwrite("fixed_thr", &ModelParams::KC::fixed_thr)
        .def_readwrite("use_fixed_thr", &ModelParams::KC::use_fixed_thr)
        .def_readwrite("sp_target", &ModelParams::KC::sp_target)
        .def_readwrite("sp_acc", &ModelParams::KC::sp_acc)
        .def_readwrite("sp_lr_coeff", &ModelParams::KC::sp_lr_coeff)
        .def_readwrite("max_iters", &ModelParams::KC::max_iters)
        .def_readwrite("tune_from", &ModelParams::KC::tune_from)
        .def_readwrite("taum", &ModelParams::KC::taum)
        .def_readwrite("apl_taum", &ModelParams::KC::apl_taum)
        .def_readwrite("tau_apl2kc", &ModelParams::KC::tau_apl2kc);

	/* TODO convert all values in DEFAULT_PARAMS to default kwargs on a python
       constsructor */

	py::class_<RunVars>(m, "RunVars")
        .def_readwrite("orn", &RunVars::orn)
        .def_readwrite("ln", &RunVars::ln)
        .def_readwrite("pn", &RunVars::pn)
        .def_readwrite("kc", &RunVars::kc)
        .def(py::init<ModelParams const&>());

    py::class_<RunVars::ORN>(m, "RVORN")
        .def_readwrite("sims", &RunVars::ORN::sims);

    py::class_<RunVars::LN>(m, "RVLN")
        .def_readwrite("ln_inhA", &RunVars::LN::inhA)
        .def_readwrite("ln_inhB", &RunVars::LN::inhB);
    py::class_<RunVars::LN::InhA>(m, "RVLNInhA")
        .def_readwrite("sims", &RunVars::LN::InhA::sims);
    py::class_<RunVars::LN::InhB>(m, "RVLNInhB")
        .def_readwrite("sims", &RunVars::LN::InhB::sims);

    py::class_<RunVars::PN>(m, "RVPN")
        .def_readwrite("pn_sims", &RunVars::PN::sims);

    py::class_<RunVars::KC>(m, "RVKC")
        .def_readwrite("wPNKC", &RunVars::KC::wPNKC)
        .def_readwrite("wAPLKC", &RunVars::KC::wAPLKC)
        .def_readwrite("wKCAPL", &RunVars::KC::wKCAPL)
        .def_readwrite("thr", &RunVars::KC::thr)
        .def_readwrite("responses", &RunVars::KC::responses)
        .def_readwrite("tuning_iters", &RunVars::KC::tuning_iters);

    m.def("load_hc_data", &load_hc_data, R"pbdoc(
        Load HC data from file.
    )pbdoc");

    m.def("build_wPNKC", &build_wPNKC, R"pbdoc(
        Choose between the above functions appropriately.
    )pbdoc");

    m.def("fit_sparseness", &fit_sparseness, R"pbdoc(
        Set KC spike thresholds, and tune APL<->KC weights until reaching the
        desired sparsity.
    )pbdoc");

    m.def("sim_ORN_layer", &sim_ORN_layer, R"pbdoc(
        Model ORN response for one odor.
    )pbdoc");

    m.def("sim_LN_layer", &sim_LN_layer, R"pbdoc(
        Model LN response for one odor.
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