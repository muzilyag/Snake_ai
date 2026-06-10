#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "vec_env.h"

namespace py = pybind11;

PYBIND11_MODULE(snake_cpp, m) 
{
    py::class_<VecSnakeEngine>(m, "VecSnakeEngine")
        .def(py::init<int, int, int, int, int, int, std::vector<float>, 
                      py::array_t<float>, py::array_t<float>, py::array_t<float>, 
                      py::array_t<int>, py::array_t<int>, py::array_t<int>, 
                      py::array_t<int>, py::array_t<int>>())
        .def("reset_all", &VecSnakeEngine::reset_all)
        .def("step", &VecSnakeEngine::step);
}