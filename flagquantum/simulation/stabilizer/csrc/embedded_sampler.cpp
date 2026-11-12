// FlagQuantum's private CPython bridge to the pinned Stim measurement sampler.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cstring>
#include <cerrno>
#include <system_error>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "stim/circuit/circuit.h"
#include "stim/io/measure_record_writer.h"
#include "stim/io/measure_record_reader.h"
#include "stim/mem/simd_bit_table.h"
#include "stim/simulators/frame_simulator_util.h"
#include "stim/simulators/error_analyzer.h"
#include "stim/simulators/measurements_to_detection_events.h"
#include "stim/simulators/tableau_simulator.h"
#include "stim/util_bot/probability_util.h"

namespace {

constexpr char capsule_name[] = "flagquantum.embedded_stim_sampler";
constexpr size_t width = stim::MAX_BITWORD_WIDTH;
static_assert(width <= 1024, "the reviewed shot-data estimate assumes SIMD width <= 1024");

size_t checked_add(size_t left, size_t right) {
    if (right > std::numeric_limits<size_t>::max() - left) {
        throw std::invalid_argument("Shot data size overflow.");
    }
    return left + right;
}

size_t checked_mul(size_t left, size_t right) {
    if (left && right > std::numeric_limits<size_t>::max() / left) {
        throw std::invalid_argument("Shot data size overflow.");
    }
    return left * right;
}

size_t rounded_simd(size_t value) {
    return checked_mul(checked_add(value, 1023) / 1024, 1024);
}

size_t shot_principal_bytes(size_t shots, size_t columns, bool packed) {
    const size_t stride = packed ? checked_add(columns, 7) / 8 : columns;
    const size_t scratch = checked_mul(
        checked_mul(rounded_simd(columns), rounded_simd(std::max<size_t>(shots, 64))), 2) / 8;
    const size_t transposed = checked_mul(rounded_simd(columns), 2) / 8;
    return checked_add(checked_add(checked_mul(shots, stride), scratch),
                       checked_add(transposed, 65536));
}

size_t shot_output_bound(size_t shots, size_t columns, stim::SampleFormat format) {
    if (format == stim::SampleFormat::SAMPLE_FORMAT_B8) {
        return checked_mul(shots, checked_add(columns, 7) / 8);
    }
    if (format == stim::SampleFormat::SAMPLE_FORMAT_PTB64) {
        if (shots % 64) throw std::invalid_argument("ptb64 shots must be a multiple of 64.");
        return checked_mul(checked_mul(shots / 64, columns), 8);
    }
    if (format == stim::SampleFormat::SAMPLE_FORMAT_01 ||
        format == stim::SampleFormat::SAMPLE_FORMAT_R8) {
        return checked_mul(shots, checked_add(columns, 1));
    }
    const size_t digits = std::to_string(columns - 1).size();
    if (format == stim::SampleFormat::SAMPLE_FORMAT_HITS) {
        return checked_mul(shots, std::max<size_t>(1, checked_mul(columns, digits + 1)));
    }
    return checked_mul(shots, checked_add(5, checked_mul(columns, digits + 2)));
}

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

stim::SampleFormat parse_format(const char *format) {
    if (strcmp(format, "01") == 0) return stim::SampleFormat::SAMPLE_FORMAT_01;
    if (strcmp(format, "b8") == 0) return stim::SampleFormat::SAMPLE_FORMAT_B8;
    if (strcmp(format, "r8") == 0) return stim::SampleFormat::SAMPLE_FORMAT_R8;
    if (strcmp(format, "hits") == 0) return stim::SampleFormat::SAMPLE_FORMAT_HITS;
    if (strcmp(format, "dets") == 0) return stim::SampleFormat::SAMPLE_FORMAT_DETS;
    if (strcmp(format, "ptb64") == 0) return stim::SampleFormat::SAMPLE_FORMAT_PTB64;
    throw std::invalid_argument("Unknown sample file format.");
}

PyObject *write_shot_records(PyObject *, PyObject *args) {
    PyObject *data;
    Py_ssize_t shots_arg, nm_arg, nd_arg, no_arg;
    const char *path, *format;
    int packed;
    if (!PyArg_ParseTuple(args, "Onpssnnn", &data, &shots_arg, &packed, &path,
                          &format, &nm_arg, &nd_arg, &no_arg)) {
        return nullptr;
    }
    try {
        if (!PyByteArray_Check(data)) {
            throw std::invalid_argument("Shot data must be an owned bytearray.");
        }
        if (shots_arg < 0 || nm_arg < 0 || nd_arg < 0 || no_arg < 0 ||
            nm_arg > PY_SSIZE_T_MAX - nd_arg ||
            nm_arg + nd_arg > PY_SSIZE_T_MAX - no_arg) {
            throw std::invalid_argument("Invalid shot data dimensions.");
        }
        const size_t shots = size_t(shots_arg);
        const size_t nm = size_t(nm_arg), nd = size_t(nd_arg), no = size_t(no_arg);
        if ((nm == 0 && nd + no == 0) || (nm != 0 && (nd != 0 || no != 0))) {
            throw std::invalid_argument("Invalid measurement, detector and observable sections.");
        }
        const size_t columns = nm + nd + no;
        const auto output_format = parse_format(format);
        const size_t stride = packed ? (columns + 7) / 8 : columns;
        const size_t expected = checked_mul(shots, stride);
        if (expected > size_t(PY_SSIZE_T_MAX) ||
            PyByteArray_Size(data) != Py_ssize_t(expected)) {
            throw std::invalid_argument("Shot data byte length disagrees with its dimensions.");
        }
        if (shot_principal_bytes(shots, columns, packed != 0) > 64 * 1024 * 1024) {
            throw std::invalid_argument("Shot data principal payload exceeds the 64 MiB limit.");
        }
        if (shot_output_bound(shots, columns, output_format) > 1024ULL * 1024 * 1024) {
            throw std::invalid_argument("Shot data output exceeds the 1 GiB limit.");
        }
        const auto *input = reinterpret_cast<const uint8_t *>(PyByteArray_AsString(data));
        if (input == nullptr) return nullptr;
        if (packed && columns % 8) {
            const uint8_t mask = uint8_t(255 ^ ((1u << (columns % 8)) - 1));
            for (size_t shot = 0; shot < shots; ++shot) {
                if (input[shot * stride + stride - 1] & mask) {
                    throw std::invalid_argument("Shot data has nonzero packed padding bits.");
                }
            }
        } else if (!packed) {
            for (size_t index = 0; index < expected; ++index) {
                if (input[index] > 1) {
                    throw std::invalid_argument("Dense shot data contains a nonbinary value.");
                }
            }
        }

        errno = 0;
        FILE *raw_file = fopen(path, "wb");
        if (raw_file == nullptr) {
            throw std::system_error(errno ? errno : EIO, std::generic_category(),
                                    "Could not open staged shot-data output");
        }
        std::unique_ptr<FILE, decltype(&fclose)> file(raw_file, fclose);
        stim::simd_bits<width> reference(0);
        for (size_t offset = 0; offset < shots; offset += 64) {
            const size_t block_shots = std::min<size_t>(64, shots - offset);
            stim::simd_bit_table<width> table(columns, block_shots);
            for (size_t column = 0; column < columns; ++column) {
                for (size_t shot = 0; shot < block_shots; ++shot) {
                    const size_t source_row = offset + shot;
                    const bool value = packed
                        ? ((input[source_row * stride + column / 8] >> (column % 8)) & 1) != 0
                        : input[source_row * stride + column] != 0;
                    table[column][shot] = value;
                }
            }
            const char first = nm == 0 ? 'D' : 'M';
            const char second = nm == 0 ? 'L' : 'M';
            stim::write_table_data(raw_file, block_shots, columns, reference, table,
                                   output_format, first, second, nm + nd);
            if (ferror(raw_file)) {
                throw std::system_error(errno ? errno : EIO, std::generic_category(),
                                        "Failed writing staged shot data");
            }
        }
        if (fflush(raw_file) != 0 || ferror(raw_file)) {
            throw std::system_error(errno ? errno : EIO, std::generic_category(),
                                    "Failed flushing staged shot data");
        }
        FILE *closed = file.release();
        if (fclose(closed) != 0) {
            throw std::system_error(errno ? errno : EIO, std::generic_category(),
                                    "Failed closing staged shot data");
        }
        Py_RETURN_NONE;
    } catch (const std::system_error &error) {
        PyErr_SetString(PyExc_OSError, error.what());
        return nullptr;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject *write_measurements(PyObject *args, bool select_format) {
    PyObject *capsule;
    Py_ssize_t shots;
    const char *path;
    const char *format = "b8";
    const bool parsed = select_format
        ? PyArg_ParseTuple(args, "Onss", &capsule, &shots, &path, &format)
        : PyArg_ParseTuple(args, "Ons", &capsule, &shots, &path);
    if (!parsed) {
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
        const auto output_format = parse_format(format);
        if (output_format == stim::SampleFormat::SAMPLE_FORMAT_PTB64 && shots % 64) {
            throw std::invalid_argument("ptb64 shots must be a multiple of 64.");
        }
        std::unique_ptr<FILE, decltype(&fclose)> file(fopen(path, "wb"), fclose);
        if (file == nullptr) {
            throw std::runtime_error("Could not open the staged measurement output.");
        }
        stim::sample_batch_measurements_writing_results_to_disk(
            sampler->circuit, sampler->reference, size_t(shots), file.get(),
            output_format, sampler->rng);
        const bool write_failed = ferror(file.get()) != 0;
        if (fclose(file.release()) != 0 || write_failed) {
            throw std::runtime_error("Could not complete the staged measurement output.");
        }
        Py_RETURN_NONE;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}


PyObject *write_b8(PyObject *, PyObject *args) {
    return write_measurements(args, false);
}

PyObject *write_records(PyObject *, PyObject *args) {
    return write_measurements(args, true);
}


constexpr char converter_capsule_name[] = "flagquantum.embedded_stim_converter";

struct Converter {
    stim::Circuit circuit;
    stim::CircuitStats stats;
    stim::simd_bits<width> reference;

    explicit Converter(const char *text)
        : circuit(text), stats(circuit.compute_stats()), reference(0) {
        // Preserve the SDK's rejection of nondeterministic declared parities.
        stim::ErrorAnalyzer::circuit_to_detector_error_model(
            circuit, false, false, false, 0, false, false);
        reference = stim::TableauSimulator<width>::reference_sample_circuit(circuit);
    }
};

void destroy_converter(PyObject *capsule) {
    delete static_cast<Converter *>(PyCapsule_GetPointer(capsule, converter_capsule_name));
}

PyObject *prepare_converter(PyObject *, PyObject *args) {
    const char *text;
    if (!PyArg_ParseTuple(args, "s", &text)) {
        return nullptr;
    }
    try {
        auto converter = std::make_unique<Converter>(text);
        PyObject *capsule = PyCapsule_New(
            converter.get(), converter_capsule_name, destroy_converter);
        if (capsule != nullptr) {
            converter.release();
        }
        return capsule;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject *convert_records(PyObject *, PyObject *args) {
    PyObject *capsule;
    const char *raw;
    Py_ssize_t length, shots;
    if (!PyArg_ParseTuple(args, "Oy#n", &capsule, &raw, &length, &shots)) {
        return nullptr;
    }
    auto *converter = static_cast<Converter *>(
        PyCapsule_GetPointer(capsule, converter_capsule_name));
    if (converter == nullptr) {
        return nullptr;
    }
    try {
        const auto &stats = converter->stats;
        const size_t measurements = stats.num_measurements;
        const size_t detectors = stats.num_detectors;
        const size_t observables = stats.num_observables;
        const size_t limit = size_t(std::numeric_limits<Py_ssize_t>::max());
        if (shots <= 0 || measurements == 0 || detectors == 0 ||
            size_t(shots) > limit / measurements ||
            size_t(length) != size_t(shots) * measurements ||
            size_t(shots) > limit / detectors ||
            (observables != 0 && size_t(shots) > limit / observables)) {
            throw std::invalid_argument("Invalid embedded conversion dimensions.");
        }
        stim::simd_bit_table<width> records(measurements, size_t(shots));
        stim::simd_bit_table<width> sweeps(0, size_t(shots));
        stim::simd_bit_table<width> results(detectors + observables, size_t(shots));
        for (size_t shot = 0; shot < size_t(shots); ++shot) {
            for (size_t column = 0; column < measurements; ++column) {
                const unsigned char bit = raw[shot * measurements + column];
                if (bit > 1) {
                    throw std::invalid_argument("Measurement bytes must be binary.");
                }
                records[column][shot] = bit;
            }
        }
        stim::measurements_to_detection_events_helper(
            records, sweeps, results, converter->circuit.aliased_noiseless_circuit(),
            stats, converter->reference, true);
        PyObject *pair = PyTuple_New(2);
        if (pair == nullptr) {
            return nullptr;
        }
        for (size_t part = 0; part < 2; ++part) {
            const size_t columns = part == 0 ? detectors : observables;
            const size_t offset = part == 0 ? 0 : detectors;
            PyObject *buffer = PyByteArray_FromStringAndSize(
                nullptr, Py_ssize_t(size_t(shots) * columns));
            if (buffer == nullptr) {
                Py_DECREF(pair);
                return nullptr;
            }
            auto *output = PyByteArray_AsString(buffer);
            for (size_t shot = 0; shot < size_t(shots); ++shot) {
                for (size_t column = 0; column < columns; ++column) {
                    output[shot * columns + column] = bool(results[offset + column][shot]);
                }
            }
            PyTuple_SetItem(pair, Py_ssize_t(part), buffer);
        }
        return pair;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}


PyObject *convert_file(PyObject *, PyObject *args) {
    PyObject *capsule;
    const char *input_path, *input_format, *output_path, *output_format, *obs_path, *obs_format;
    int append;
    if (!PyArg_ParseTuple(args, "Osssspzs", &capsule, &input_path, &input_format,
                          &output_path, &output_format, &append, &obs_path, &obs_format)) {
        return nullptr;
    }
    auto *converter = static_cast<Converter *>(PyCapsule_GetPointer(capsule, converter_capsule_name));
    if (converter == nullptr) return nullptr;
    try {
        const auto in_format = parse_format(input_format);
        const auto out_format = parse_format(output_format);
        const auto logical_format = parse_format(obs_format);
        const auto &stats = converter->stats;
        if (out_format == stim::SampleFormat::SAMPLE_FORMAT_PTB64 ||
            (obs_path && logical_format == stim::SampleFormat::SAMPLE_FORMAT_PTB64) ||
            stats.num_measurements == 0 || stats.num_measurements > 65536 ||
            stats.num_qubits > 4096 || stats.num_detectors + stats.num_observables > 65536) {
            throw std::invalid_argument("Invalid embedded file conversion request.");
        }
        auto padded = [](size_t n) { return std::max(size_t(1024), ((n + 1023) / 1024) * 1024); };
        const size_t rows = padded(stats.num_measurements) +
            2 * padded(stats.num_detectors + stats.num_observables) + 2 * padded(stats.num_qubits) +
            2 * padded(2 * stats.num_measurements) + padded(stats.num_observables) + 4 * padded(0) + 8;
        if (rows * 128 > 64 * 1024 * 1024) {
            throw std::invalid_argument("Embedded conversion principal tables exceed 64 MiB.");
        }
        std::unique_ptr<FILE, decltype(&fclose)> input(fopen(input_path, "rb"), fclose);
        if (!input) throw std::runtime_error("Could not open the measurement snapshot.");
        std::unique_ptr<FILE, decltype(&fclose)> output(fopen(output_path, "wb"), fclose);
        std::unique_ptr<FILE, decltype(&fclose)> obs(obs_path ? fopen(obs_path, "wb") : nullptr, fclose);
        if (!output || (obs_path && !obs)) throw std::runtime_error("Could not open staged conversion output.");
        stim::stream_measurements_to_detection_events_helper<width>(
            input.get(), in_format, nullptr, stim::SampleFormat::SAMPLE_FORMAT_01,
            output.get(), out_format, converter->circuit.aliased_noiseless_circuit(),
            stats, append, converter->reference, obs.get(), logical_format);
        bool failed = ferror(input.get()) != 0 || ferror(output.get()) != 0;
        if (obs) {
            failed |= ferror(obs.get()) != 0;
            failed |= fclose(obs.release()) != 0;
        }
        failed |= fclose(input.release()) != 0;
        failed |= fclose(output.release()) != 0;
        if (failed) throw std::runtime_error("Could not complete staged conversion files.");
        Py_RETURN_NONE;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}


PyObject *prepare_detector(PyObject *self, PyObject *args) {
    const char *text;
    PyObject *seed;
    if (!PyArg_ParseTuple(args, "sO", &text, &seed)) {
        return nullptr;
    }
    try {
        stim::Circuit circuit(text);
        // Gauge rejection concerns the ideal circuit, not exact noisy DEM admission.
        stim::ErrorAnalyzer::circuit_to_detector_error_model(
            circuit.aliased_noiseless_circuit(), false, false, false, 0, false, false);
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
    return prepare(self, args);
}

PyObject *sample_detectors(PyObject *, PyObject *args) {
    PyObject *capsule;
    Py_ssize_t shots;
    if (!PyArg_ParseTuple(args, "On", &capsule, &shots)) {
        return nullptr;
    }
    auto *sampler = static_cast<Sampler *>(PyCapsule_GetPointer(capsule, capsule_name));
    if (sampler == nullptr) {
        return nullptr;
    }
    try {
        const auto stats = sampler->circuit.compute_stats();
        const size_t limit = size_t(std::numeric_limits<Py_ssize_t>::max());
        if (shots <= 0 || (stats.num_detectors && size_t(shots) > limit / stats.num_detectors) ||
            (stats.num_observables && size_t(shots) > limit / stats.num_observables)) {
            throw std::invalid_argument("Invalid embedded detector sample dimensions.");
        }
        auto records = stim::sample_batch_detection_events<width>(
            sampler->circuit, size_t(shots), sampler->rng);
        PyObject *pair = PyTuple_New(2);
        if (pair == nullptr) {
            return nullptr;
        }
        for (size_t part = 0; part < 2; ++part) {
            const size_t columns = part == 0 ? stats.num_detectors : stats.num_observables;
            const auto &table = part == 0 ? records.first : records.second;
            PyObject *buffer = PyByteArray_FromStringAndSize(nullptr, Py_ssize_t(size_t(shots) * columns));
            if (buffer == nullptr) {
                Py_DECREF(pair);
                return nullptr;
            }
            auto *output = PyByteArray_AsString(buffer);
            for (size_t shot = 0; shot < size_t(shots); ++shot) {
                for (size_t column = 0; column < columns; ++column) {
                    output[shot * columns + column] = bool(table[column][shot]);
                }
            }
            PyTuple_SetItem(pair, Py_ssize_t(part), buffer);
        }
        return pair;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject *write_detectors(PyObject *, PyObject *args) {
    PyObject *capsule;
    Py_ssize_t shots;
    const char *path, *format, *obs_path, *obs_format;
    int prepend, append;
    if (!PyArg_ParseTuple(args, "Onssppzs", &capsule, &shots, &path, &format,
                          &prepend, &append, &obs_path, &obs_format)) return nullptr;
    auto *sampler = static_cast<Sampler *>(PyCapsule_GetPointer(capsule, capsule_name));
    if (sampler == nullptr) return nullptr;
    try {
        auto output_format = parse_format(format);
        auto observable_format = parse_format(obs_format);
        if (shots <= 0 || prepend + append + (obs_path != nullptr) > 1 ||
            ((output_format == stim::SampleFormat::SAMPLE_FORMAT_PTB64 ||
              (obs_path && observable_format == stim::SampleFormat::SAMPLE_FORMAT_PTB64)) && shots % 64)) {
            throw std::invalid_argument("Invalid embedded detector file request.");
        }
        std::unique_ptr<FILE, decltype(&fclose)> file(fopen(path, "wb"), fclose);
        std::unique_ptr<FILE, decltype(&fclose)> obs(obs_path ? fopen(obs_path, "wb") : nullptr, fclose);
        if (!file || (obs_path && !obs)) throw std::runtime_error("Could not open staged detector output.");
        stim::sample_batch_detection_events_writing_results_to_disk<width>(
            sampler->circuit, size_t(shots), prepend, append, file.get(),
            output_format, sampler->rng, obs.get(), observable_format);
        bool failed = ferror(file.get()) != 0;
        if (obs) {
            failed |= ferror(obs.get()) != 0;
            failed |= fclose(obs.release()) != 0;
        }
        failed |= fclose(file.release()) != 0;
        if (failed) throw std::runtime_error("Could not complete staged detector output.");
        Py_RETURN_NONE;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject *detector_error_model(PyObject *, PyObject *args) {
    const char *text;
    if (!PyArg_ParseTuple(args, "s", &text)) {
        return nullptr;
    }
    try {
        stim::Circuit circuit(text);
        auto model = stim::ErrorAnalyzer::circuit_to_detector_error_model(
            circuit, false, false, false, 0, false, false);
        const auto result = model.str();
        if (result.size() > 16 * 1024 * 1024) {
            throw std::invalid_argument("Embedded DEM text exceeds 16 MiB.");
        }
        return PyUnicode_FromStringAndSize(result.data(), Py_ssize_t(result.size()));
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

constexpr char reader_capsule_name[] = "flagquantum.embedded_shot_reader";

struct ShotReader {
    FILE *file;
    std::unique_ptr<stim::MeasureRecordReader<width>> reader;
    size_t columns;
    bool check_padding;

    ShotReader(const char *path, const char *format, size_t nm, size_t nd, size_t no)
        : file(std::fopen(path, "rb")), columns(nm + nd + no),
          check_padding(std::strcmp(format, "b8") == 0) {
        if (!file) throw std::system_error(errno, std::generic_category(), "Cannot open shot snapshot");
        try {
            reader = stim::MeasureRecordReader<width>::make(file, parse_format(format), nm, nd, no);
        } catch (...) {
            std::fclose(file);
            file = nullptr;
            throw;
        }
    }
    ~ShotReader() { if (file) std::fclose(file); }
    void close() {
        reader.reset();
        FILE *owned = file;
        file = nullptr;
        if (owned && std::fclose(owned)) throw std::runtime_error("Cannot close shot snapshot.");
    }
    bool next(stim::simd_bits<width> &row) {
        if (!file || !reader) throw std::invalid_argument("Shot reader is closed.");
        row.clear();
        bool more = reader->start_and_read_entire_record(row);
        if (std::ferror(file)) throw std::system_error(errno, std::generic_category(), "Cannot read shot snapshot");
        if (more && check_padding && columns % 8 && (row.u8[columns / 8] >> (columns % 8)))
            throw std::invalid_argument("Nonzero b8 shot padding.");
        return more;
    }
};

void destroy_shot_reader(PyObject *capsule) {
    delete static_cast<ShotReader *>(PyCapsule_GetPointer(capsule, reader_capsule_name));
}

PyObject *open_shot_reader(PyObject *, PyObject *args) {
    const char *path, *format;
    Py_ssize_t nm, nd, no;
    if (!PyArg_ParseTuple(args, "ssnnn", &path, &format, &nm, &nd, &no)) return nullptr;
    try {
        if (nm < 0 || nd < 0 || no < 0 || nm > PY_SSIZE_T_MAX - nd || nm + nd > PY_SSIZE_T_MAX - no || nm + nd + no == 0)
            throw std::invalid_argument("Invalid shot columns.");
        // Bound codec scratch before construction and subsequent size arithmetic.
        constexpr size_t max_columns = ((64 * 1024 * 1024 - 65536) / 16384) * 1024;
        if (size_t(nm + nd + no) > max_columns)
            throw std::invalid_argument("Shot codec scratch exceeds byte limit.");
        auto owned = std::make_unique<ShotReader>(path, format, size_t(nm), size_t(nd), size_t(no));
        PyObject *capsule = PyCapsule_New(owned.get(), reader_capsule_name, destroy_shot_reader);
        if (capsule) owned.release();
        return capsule;
    } catch (const std::system_error &error) { PyErr_SetString(PyExc_OSError, error.what()); return nullptr; } catch (const std::exception &error) {
        PyErr_SetString(PyExc_ValueError, error.what()); return nullptr;
    }
}

PyObject *close_shot_reader(PyObject *, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return nullptr;
    auto *cursor = static_cast<ShotReader *>(PyCapsule_GetPointer(capsule, reader_capsule_name));
    if (!cursor) return nullptr;
    try { cursor->close(); Py_RETURN_NONE; }
    catch (const std::exception &error) { PyErr_SetString(PyExc_OSError, error.what()); return nullptr; }
}

PyObject *scan_shot_reader(PyObject *, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return nullptr;
    auto *cursor = static_cast<ShotReader *>(PyCapsule_GetPointer(capsule, reader_capsule_name));
    if (!cursor) return nullptr;
    try {
        stim::simd_bits<width> row(cursor->columns);
        size_t count = 0;
        while (cursor->next(row)) {
            if (count == size_t(PY_SSIZE_T_MAX)) throw std::invalid_argument("Shot count overflow.");
            ++count;
        }
        return PyLong_FromSize_t(count);
    } catch (const std::system_error &error) { PyErr_SetString(PyExc_OSError, error.what()); return nullptr; } catch (const std::exception &error) { PyErr_SetString(PyExc_ValueError, error.what()); return nullptr; }
}

PyObject *read_shot_reader(PyObject *, PyObject *args) {
    PyObject *capsule;
    Py_ssize_t requested, first_columns;
    int packed, separate;
    if (!PyArg_ParseTuple(args, "Onppn", &capsule, &requested, &packed, &separate, &first_columns)) return nullptr;
    auto *cursor = static_cast<ShotReader *>(PyCapsule_GetPointer(capsule, reader_capsule_name));
    if (!cursor) return nullptr;
    try {
        if (requested < 0 || first_columns < 0 || size_t(first_columns) > cursor->columns)
            throw std::invalid_argument("Invalid shot read dimensions.");
        size_t a = separate ? size_t(first_columns) : cursor->columns;
        size_t b = separate ? cursor->columns - a : 0;
        size_t astride = packed ? (a + 7) / 8 : a;
        size_t bstride = packed ? (b + 7) / 8 : b;
        // Independently enforce the reviewed payload estimate at the native boundary.
        size_t p = (cursor->columns + 7) / 8;
        size_t scratch = 128 * ((cursor->columns + 1023) / 1024) * 128 + 65536;
        constexpr size_t limit = 64 * 1024 * 1024;
        if (scratch > limit || p + astride + bstride > (limit - scratch) / 4 ||
            size_t(requested) > (limit - scratch) / (4 * (p + astride + bstride)))
            throw std::invalid_argument("Shot payload exceeds byte limit.");
        PyObject *first = PyByteArray_FromStringAndSize(nullptr, requested * astride);
        if (!first) return nullptr;
        PyObject *second = PyByteArray_FromStringAndSize(nullptr, requested * bstride);
        if (!second) { Py_DECREF(first); return nullptr; }
        auto *outa = reinterpret_cast<uint8_t *>(PyByteArray_AsString(first));
        auto *outb = reinterpret_cast<uint8_t *>(PyByteArray_AsString(second));
        std::memset(outa, 0, requested * astride);
        std::memset(outb, 0, requested * bstride);
        size_t count = 0;
        try {
            stim::simd_bits<width> row(cursor->columns);
            while (count < size_t(requested) && cursor->next(row)) {
                for (size_t c = 0; c < cursor->columns; ++c) {
                    if (!row[c]) continue;
                    size_t bit = c < a ? c : c - a;
                    size_t stride = c < a ? astride : bstride;
                    uint8_t *out = c < a ? outa : outb;
                    if (packed) out[count * stride + bit / 8] |= uint8_t(1 << (bit % 8));
                    else out[count * stride + bit] = 1;
                }
                ++count;
            }
        } catch (...) { Py_DECREF(first); Py_DECREF(second); throw; }
        PyObject *result = PyTuple_New(3);
        PyObject *rows = PyLong_FromSize_t(count);
        if (!result || !rows) { Py_XDECREF(result); Py_XDECREF(rows); Py_DECREF(first); Py_DECREF(second); return nullptr; }
        PyTuple_SetItem(result, 0, first); PyTuple_SetItem(result, 1, second); PyTuple_SetItem(result, 2, rows);
        return result;
    } catch (const std::system_error &error) { PyErr_SetString(PyExc_OSError, error.what()); return nullptr; } catch (const std::exception &error) { PyErr_SetString(PyExc_ValueError, error.what()); return nullptr; }
}

PyMethodDef methods[] = {
    {"write_shot_records", write_shot_records, METH_VARARGS, nullptr},
    {"open_shot_reader", open_shot_reader, METH_VARARGS, nullptr},
    {"scan_shot_reader", scan_shot_reader, METH_VARARGS, nullptr},
    {"read_shot_reader", read_shot_reader, METH_VARARGS, nullptr},
    {"close_shot_reader", close_shot_reader, METH_VARARGS, nullptr},
    {"prepare_detector", prepare_detector, METH_VARARGS, nullptr},
    {"sample_detectors", sample_detectors, METH_VARARGS, nullptr},
    {"write_detectors", write_detectors, METH_VARARGS, nullptr},
    {"detector_error_model", detector_error_model, METH_VARARGS, nullptr},
    {"prepare_converter", prepare_converter, METH_VARARGS, nullptr},
    {"convert_records", convert_records, METH_VARARGS, nullptr},
    {"convert_file", convert_file, METH_VARARGS, nullptr},
    {"prepare", prepare, METH_VARARGS, nullptr},
    {"sample", sample, METH_VARARGS, nullptr},
    {"write_b8", write_b8, METH_VARARGS, nullptr},
    {"write_records", write_records, METH_VARARGS, nullptr},
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
