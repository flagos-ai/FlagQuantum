// FlagQuantum's private CPython bridge to the pinned Stim measurement sampler.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

#include "stim/circuit/circuit.h"
#include "stim/simulators/frame_simulator_util.h"
#include "stim/simulators/tableau_simulator.h"
#include "stim/util_bot/probability_util.h"

namespace {

constexpr char capsule_name[] = "flagquantum.embedded_stim_sampler";
constexpr size_t width = stim::MAX_BITWORD_WIDTH;

struct Sampler {
    stim::Circuit circuit;
    stim::simd_bits<width> reference;
    std::mt19937_64 rng;

    Sampler(const char *text, std::mt19937_64 generator)
        : circuit(text),
          reference(stim::TableauSimulator<width>::reference_sample_circuit(circuit)),
          rng(std::move(generator)) {}
};

void destroy_sampler(PyObject *capsule) {
    delete static_cast<Sampler *>(PyCapsule_GetPointer(capsule, capsule_name));
}

PyObject *prepare(PyObject *, PyObject *args) {
    const char *text;
    PyObject *seed;
    if (!PyArg_ParseTuple(args, "sO", &text, &seed)) {
        return nullptr;
    }
    try {
        auto rng = seed == Py_None ? stim::externally_seeded_rng() : std::mt19937_64{};
        if (seed != Py_None) {
            const auto value = PyLong_AsUnsignedLongLong(seed);
            if (PyErr_Occurred()) {
                return nullptr;
            }
            rng.seed(value ^ stim::INTENTIONAL_VERSION_SEED_INCOMPATIBILITY);
        }
        auto sampler = std::make_unique<Sampler>(text, std::move(rng));
        PyObject *capsule = PyCapsule_New(sampler.get(), capsule_name, destroy_sampler);
        if (capsule != nullptr) {
            sampler.release();
        }
        return capsule;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject *sample(PyObject *, PyObject *args) {
    PyObject *capsule;
    Py_ssize_t shots;
    int packed;
    if (!PyArg_ParseTuple(args, "Onp", &capsule, &shots, &packed)) {
        return nullptr;
    }
    auto *sampler = static_cast<Sampler *>(PyCapsule_GetPointer(capsule, capsule_name));
    if (sampler == nullptr) {
        return nullptr;
    }
    try {
        const size_t columns = sampler->circuit.count_measurements();
        const size_t stride = packed ? (columns + 7) / 8 : columns;
        if (shots <= 0 || stride == 0 ||
            size_t(shots) > size_t(std::numeric_limits<Py_ssize_t>::max()) / stride) {
            throw std::invalid_argument("Invalid embedded sampler output dimensions.");
        }
        auto records = stim::sample_batch_measurements(
            sampler->circuit, sampler->reference, size_t(shots), sampler->rng, false);
        const size_t bytes = size_t(shots) * stride;
        PyObject *buffer = PyByteArray_FromStringAndSize(nullptr, Py_ssize_t(bytes));
        if (buffer == nullptr) {
            return nullptr;
        }
        auto *output = reinterpret_cast<unsigned char *>(PyByteArray_AsString(buffer));
        std::memset(output, 0, bytes);
        for (size_t column = 0; column < columns; ++column) {
            const auto record = records[column];
            for (size_t shot = 0; shot < size_t(shots); ++shot) {
                if (packed) {
                    output[shot * stride + column / 8] |=
                        static_cast<unsigned char>(bool(record[shot]) << (column % 8));
                } else {
                    output[shot * stride + column] = bool(record[shot]);
                }
            }
        }
        return buffer;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject *write_b8(PyObject *, PyObject *args) {
    PyObject *capsule;
    Py_ssize_t shots;
    const char *path;
    if (!PyArg_ParseTuple(args, "Ons", &capsule, &shots, &path)) {
        return nullptr;
    }
    auto *sampler = static_cast<Sampler *>(PyCapsule_GetPointer(capsule, capsule_name));
    if (sampler == nullptr) {
        return nullptr;
    }
    try {
        if (shots <= 0) {
            throw std::invalid_argument("Invalid embedded sampler shot count.");
        }
        std::unique_ptr<FILE, decltype(&fclose)> file(fopen(path, "wb"), fclose);
        if (file == nullptr) {
            throw std::runtime_error("Could not open the staged b8 output.");
        }
        stim::sample_batch_measurements_writing_results_to_disk(
            sampler->circuit, sampler->reference, size_t(shots), file.get(),
            stim::SampleFormat::SAMPLE_FORMAT_B8, sampler->rng);
        const bool write_failed = ferror(file.get()) != 0;
        if (fclose(file.release()) != 0 || write_failed) {
            throw std::runtime_error("Could not complete the staged b8 output.");
        }
        Py_RETURN_NONE;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {"prepare", prepare, METH_VARARGS, nullptr},
    {"sample", sample, METH_VARARGS, nullptr},
    {"write_b8", write_b8, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_embedded_stim", nullptr, -1, methods,
    nullptr, nullptr, nullptr, nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__embedded_stim() {
    PyObject *result = PyModule_Create(&module);
    if (result != nullptr &&
        PyModule_AddStringConstant(result, "upstream_commit",
                                   "e2fc1eca7fd21684d433aa5f10f4504ea4860d07") != 0) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}
