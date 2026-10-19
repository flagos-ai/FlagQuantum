#include <Python.h>

namespace {

PyObject* parallel_build_available(PyObject*, PyObject*) {
#ifdef FQ_NATIVE_CPU_PARALLEL
  Py_RETURN_TRUE;
#else
  Py_RETURN_FALSE;
#endif
}

PyMethodDef methods[] = {
    {
        "parallel_build_available",
        parallel_build_available,
        METH_NOARGS,
        "Return whether the extension was built with intra-op parallelism.",
    },
    {nullptr, nullptr, 0, nullptr},
};

}  // namespace

extern "C" PyObject* PyInit__C() {
  static PyModuleDef module = {
      PyModuleDef_HEAD_INIT,
      "_C",
      nullptr,
      -1,
      methods,
  };
  return PyModule_Create(&module);
}
