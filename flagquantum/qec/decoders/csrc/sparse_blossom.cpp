// Copyright 2026 FlagQuantum Contributors
// SPDX-License-Identifier: Apache-2.0

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "pymatching/sparse_blossom/matcher/mwpm.h"

namespace {

constexpr char capsule_name[] = "flagquantum.qec.sparse_blossom";
constexpr pm::signed_weight_int max_half_weight = (1 << 24) - 1;
constexpr size_t observable_capacity = sizeof(pm::obs_int) * 8;

struct Edge {
    size_t first;
    size_t second;
    double weight;
    std::vector<size_t> observables;
};

struct Matcher {
    pm::Mwpm solver;
    size_t detectors;
    size_t observables;
};

void dispose_matcher(PyObject* capsule) {
    delete static_cast<Matcher*>(PyCapsule_GetPointer(capsule, capsule_name));
}

std::unique_ptr<Matcher> make_matcher(
    size_t detectors, size_t observables, const std::vector<Edge>& edges) {
    // Use PyMatching 2.4's weight discretization. Even weights make collisions
    // occur at integer times.
    double maximum = 0;
    bool integral = true;
    for (const auto& edge : edges) {
        maximum = std::max(maximum, edge.weight);
        integral &= std::round(edge.weight) == edge.weight;
    }
    double scale = integral ? 1.0 : max_half_weight / maximum;
    pm::MatchingGraph graph(detectors, observables, 2 * scale);
    for (const auto& edge : edges) {
        auto weight = 2 * static_cast<pm::signed_weight_int>(std::round(scale * edge.weight));
        if (edge.second != detectors) {
            graph.add_edge(edge.first, edge.second, weight, edge.observables);
        }
    }
    // Match the upstream insertion order, including for tied paths.
    for (const auto& edge : edges) {
        if (edge.second == detectors) {
            auto weight = 2 * static_cast<pm::signed_weight_int>(std::round(scale * edge.weight));
            graph.add_boundary_edge(edge.first, weight, edge.observables);
        }
    }
    return std::make_unique<Matcher>(
        Matcher{pm::Mwpm(pm::GraphFlooder(std::move(graph))), detectors, observables});
}

PyObject* create(PyObject*, PyObject* arguments) {
    Py_ssize_t detectors;
    Py_ssize_t observables;
    PyObject* records;
    if (!PyArg_ParseTuple(arguments, "nnO!", &detectors, &observables, &PyTuple_Type, &records)) {
        return nullptr;
    }
    if (detectors < 1 || observables < 1 || observables > static_cast<Py_ssize_t>(observable_capacity)) {
        PyErr_SetString(PyExc_ValueError, "Expected at least one detector and 1 to 64 observables.");
        return nullptr;
    }

    try {
        std::vector<Edge> edges;
        const auto edge_count = PyTuple_Size(records);
        edges.reserve(edge_count);
        for (Py_ssize_t index = 0; index < edge_count; ++index) {
            Py_ssize_t first;
            Py_ssize_t second;
            double weight;
            PyObject* labels;
            if (!PyArg_ParseTuple(PyTuple_GetItem(records, index), "nndO!",
                                  &first, &second, &weight, &PyTuple_Type, &labels)) {
                return nullptr;
            }
            if (first < 0 || first >= second || second > detectors ||
                !std::isfinite(weight) || weight < 0 || weight > max_half_weight) {
                PyErr_SetString(PyExc_ValueError, "Invalid detector edge or weight.");
                return nullptr;
            }
            Edge edge{static_cast<size_t>(first), static_cast<size_t>(second), weight, {}};
            for (Py_ssize_t label = 0; label < PyTuple_Size(labels); ++label) {
                const auto id = PyLong_AsSsize_t(PyTuple_GetItem(labels, label));
                if (PyErr_Occurred()) {
                    return nullptr;
                }
                if (id < 0 || id >= observables) {
                    PyErr_SetString(PyExc_ValueError, "Logical observable index is outside the graph.");
                    return nullptr;
                }
                edge.observables.push_back(static_cast<size_t>(id));
            }
            edges.push_back(std::move(edge));
        }
        auto matcher = make_matcher(detectors, observables, edges);
        PyObject* capsule = PyCapsule_New(matcher.get(), capsule_name, dispose_matcher);
        if (capsule != nullptr) {
            matcher.release();
        }
        return capsule;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

pm::MatchingResult decode_row(pm::Mwpm& solver, const std::vector<size_t>& events) {
    solver.flooder.queue.cur_time = 0;
    for (size_t detector : events) {
        solver.create_detection_event(&solver.flooder.graph.nodes[detector]);
    }
    for (auto event = solver.flooder.run_until_next_mwpm_notification();
         event.event_type != pm::NO_EVENT; event = solver.flooder.run_until_next_mwpm_notification()) {
        solver.process_event(event);
    }
    if (solver.node_arena.allocated.size() != solver.node_arena.available.size()) {
        solver.reset();
        // Upstream reset explicitly destroys these arenas. Reconstruct them
        // before another decode or the solver's eventual destruction.
        std::construct_at(&solver.node_arena);
        std::construct_at(&solver.flooder.region_arena);
        throw std::invalid_argument("No perfect matching exists for this syndrome.");
    }
    pm::MatchingResult result;
    for (size_t detector : events) {
        const auto& node = solver.flooder.graph.nodes[detector];
        if (node.region_that_arrived) {
            result += solver.shatter_blossom_and_extract_matches(node.region_that_arrived_top);
        }
    }
    return result;
}

PyObject* decode_batch_impl(
    PyObject* arguments, bool packed_shots, bool packed_predictions, bool with_weights) {
    PyObject* capsule;
    PyObject* input;
    if (with_weights) {
        int input_packed;
        int output_packed;
        if (!PyArg_ParseTuple(arguments, "OO!pp", &capsule, &PyBytes_Type, &input,
                              &input_packed, &output_packed)) {
            return nullptr;
        }
        packed_shots = input_packed != 0;
        packed_predictions = output_packed != 0;
    } else if (!PyArg_ParseTuple(arguments, "OO!", &capsule, &PyBytes_Type, &input)) {
        return nullptr;
    }
    auto* matcher = static_cast<Matcher*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (matcher == nullptr) {
        return nullptr;
    }
    char* data;
    Py_ssize_t size;
    if (PyBytes_AsStringAndSize(input, &data, &size) < 0) {
        return nullptr;
    }
    const size_t input_width = packed_shots ? (matcher->detectors + 7) / 8 : matcher->detectors;
    const size_t output_width = packed_predictions ? (matcher->observables + 7) / 8 : matcher->observables;
    if (static_cast<size_t>(size) % input_width != 0) {
        PyErr_SetString(PyExc_ValueError, "Syndrome bytes do not match the graph width.");
        return nullptr;
    }
    const size_t shots = static_cast<size_t>(size) / input_width;
    if (shots > PY_SSIZE_T_MAX / output_width ||
        (with_weights && shots > PY_SSIZE_T_MAX / sizeof(double))) {
        return PyErr_NoMemory();
    }
    PyObject* output = PyBytes_FromStringAndSize(nullptr, shots * output_width);
    if (output == nullptr) {
        return nullptr;
    }
    PyObject* weights = nullptr;
    if (with_weights) {
        weights = PyBytes_FromStringAndSize(nullptr, shots * sizeof(double));
        if (weights == nullptr) {
            Py_DECREF(output);
            return nullptr;
        }
    }
    char* predictions = PyBytes_AsString(output);
    char* weight_data = with_weights ? PyBytes_AsString(weights) : nullptr;
    try {
        std::vector<size_t> events;
        events.reserve(matcher->detectors);
        for (size_t shot = 0; shot < shots; ++shot) {
            events.clear();
            const auto* row = reinterpret_cast<const unsigned char*>(data);
            const size_t remainder = matcher->detectors % 8;
            if (packed_shots && remainder && (row[input_width - 1] >> remainder)) {
                throw std::invalid_argument("Unused packed detector padding bits must be zero.");
            }
            for (size_t detector = 0; detector < matcher->detectors; ++detector) {
                bool fired = packed_shots ? ((row[detector / 8] >> (detector % 8)) & 1) : row[detector] != 0;
                if (fired) {
                    events.push_back(detector);
                }
            }
            data += input_width;
            auto result = decode_row(matcher->solver, events);
            auto mask = result.obs_mask;
            if (with_weights) {
                // Match upstream's rescaling of the quantized matching cost.
                double weight = static_cast<double>(result.weight) /
                                matcher->solver.flooder.graph.normalising_constant;
                std::memcpy(weight_data + shot * sizeof(double), &weight, sizeof(double));
            }
            if (packed_predictions) {
                // Extract bytes explicitly to preserve little-endian bit order on every host.
                for (size_t byte = 0; byte < output_width; ++byte) {
                    *predictions++ = static_cast<char>((mask >> (8 * byte)) & 0xff);
                }
            } else {
                for (size_t observable = 0; observable < matcher->observables; ++observable) {
                    *predictions++ = (mask >> observable) & 1;
                }
            }
        }
        if (!with_weights) {
            return output;
        }
        PyObject* pair = PyTuple_Pack(2, output, weights);
        Py_DECREF(output);
        Py_DECREF(weights);
        return pair;
    } catch (const std::bad_alloc&) {
        Py_XDECREF(weights);
        Py_DECREF(output);
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        Py_XDECREF(weights);
        Py_DECREF(output);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* decode_batch(PyObject*, PyObject* arguments) {
    return decode_batch_impl(arguments, false, false, false);
}

PyObject* decode_packed_batch(PyObject*, PyObject* arguments) {
    return decode_batch_impl(arguments, true, true, false);
}

PyObject* decode_batch_with_weights(PyObject*, PyObject* arguments) {
    return decode_batch_impl(arguments, false, false, true);
}

PyMethodDef methods[] = {
    {"create", create, METH_VARARGS, "Compile a graph for sparse blossom matching."},
    {"decode_batch", decode_batch, METH_VARARGS, "Decode dense rows of detector bits."},
    {"decode_packed_batch", decode_packed_batch, METH_VARARGS, "Decode packed rows of detector bits."},
    {"decode_batch_with_weights", decode_batch_with_weights, METH_VARARGS, "Decode rows and return rescaled matching weights."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_sparse_blossom", nullptr, -1, methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__sparse_blossom() {
    return PyModule_Create(&module);
}
