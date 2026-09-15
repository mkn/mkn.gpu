
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <typeinfo>
#include <iostream>

#define PRINT(x) std::cout << __FILE__ << " " << __LINE__ << " " << x << std::endl;

#include "mkn/gpu.hpp"
#include "mkn/gpu/tuple.hpp"

void init(float* a, float* b) {
  mkn::gpu::DLauncher()([a = a, b = b] __device__() {
    a[threadIdx.x] = threadIdx.x + 1;
    b[threadIdx.x] = threadIdx.x + 2;
  });
}

namespace py = pybind11;

struct FunctionSupport {
  FunctionSupport() {
    mkn::gpu::alloc_managed(a, 32);
    mkn::gpu::alloc_managed(b, 32);
    mkn::gpu::alloc_managed(c, 32);
    print();
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
      .def("A", &FunctionSupport::A)
      .def_readwrite("b", &FunctionSupport::b)
      .def("B", &FunctionSupport::B)
      .def_readwrite("a", &FunctionSupport::a)
      .def("C", &FunctionSupport::C);

  using Span_t = mkn::gpu::Span<float>;
  py::class_<Span_t, py::smart_holder>(m, "Span_s")
      .def("__getitem__", [](Span_t& self, unsigned index) { return self[index]; });
}
