#include <pybind11/pybind11.h>

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

	py::class_<ModelParams>(m, "ModelParams");
   	/*.def(py::init<const std::string &>())*/

	/* TODO convert all values in DEFAULT_PARAMS to default kwargs on a python
       constsructor */

	py::class_<RunVars>(m, "RunVars");

    m.def("load_hc_data", &load_hc_data, R"pbdoc(
        Load HC data from file.
    )pbdoc");

	/*
    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");
	*/

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
