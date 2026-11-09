
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <typeinfo>
#include <iostream>
#include <algorithm>

#define PRINT(x) std::cout << __FILE__ << " " << __LINE__ << " " << x << std::endl;

#include "mkn/gpu.hpp"

void init(float* a, float* b) {
  mkn::gpu::DLauncher()([a = a, b = b] __device__() {
    a[threadIdx.x] = threadIdx.x + 1;
    b[threadIdx.x] = threadIdx.x + 2;
  }).sync();
}

namespace py = pybind11;

struct FunctionSupport {
  FunctionSupport() {
    mkn::gpu::alloc_managed(a, 32);
    mkn::gpu::alloc_managed(b, 32);
    mkn::gpu::alloc_managed(c, 32);
    std::fill_n(c, 32, 0.f);
    init(a, b);
    print();
  }
  ~FunctionSupport() {
    mkn::gpu::destroy(a);
    mkn::gpu::destroy(b);
    mkn::gpu::destroy(c);
  }

  void print() {
    PRINT(a[0]);
    PRINT(b[0]);
    PRINT(c[0]);
  }

  py::array_t<float> make(auto p) {
    return {{32}, {sizeof(float)}, p, py::capsule(p, [](void* f) { /* noop */ })};
  }

  py::array_t<float> A() { return make(a); }
  py::array_t<float> B() { return make(b); }
  py::array_t<float> C() { return make(c); }

  float* a = nullptr;
  float* b = nullptr;
  float* c = nullptr;
};

PYBIND11_MODULE(poc_pyb, m) {
  py::class_<FunctionSupport, py::smart_holder>(m, "FunctionSupport")
      .def(py::init<>())
      .def("print", &FunctionSupport::print)
      .def_readwrite("a", &FunctionSupport::a)
      // the arrays view this object's buffers, keep it alive while they exist
      .def("A", &FunctionSupport::A, py::keep_alive<0, 1>())
      .def_readwrite("b", &FunctionSupport::b)
      .def("B", &FunctionSupport::B, py::keep_alive<0, 1>())
      .def_readwrite("c", &FunctionSupport::c)
      .def("C", &FunctionSupport::C, py::keep_alive<0, 1>());
}
