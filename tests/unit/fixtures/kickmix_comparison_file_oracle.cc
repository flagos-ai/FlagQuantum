#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "kickmix/gen/comparators/gen_cmp.h"

int main(int argc, char **argv) {
    using namespace kickmix;
    if (argc != 8) return 2;
    size_t n = std::atoi(argv[1]);
    size_t work = std::atoi(argv[2]);
    std::string kind = argv[3];
    std::string relation = argv[4];
    bool phase = std::atoi(argv[5]) != 0;
    std::string control_kind = argv[6];
    double btol = std::atof(argv[7]);
    CircuitBuilder builder;
    auto left = builder.append_register(n, "left");
    array_z storage;
    std::vector<QubitId> quantum_right;
    std::vector<BitId> classical_right;
    stride_span_z right;
    if (kind == "b") {
        classical_right = builder.append_classical_register(n, "right");
        right = stride_span_z(classical_right);
    } else if (kind == "q") {
        quantum_right = builder.append_register(n, "right");
        right = stride_span_z(quantum_right);
    } else if (kind == "m") {
        quantum_right = builder.append_register((n + 2) / 3, "right_q");
        classical_right = builder.append_classical_register((n + 1) / 3, "right_b");
        storage = array_z::alloc_noinit(n);
        for (size_t k = 0; k < n; ++k) {
            if (k % 3 == 0) storage[k] = quantum_right[k / 3];
            else if (k % 3 == 1) storage[k] = classical_right[k / 3];
            else storage[k] = k % 2 == 0;
        }
        right = storage;
    } else {
        storage = array_z::alloc_noinit(n);
        for (size_t k = 0; k < n; ++k) storage[k] = k >= n / 2;
        storage.recompute_common_type();
        right = storage;
    }
    QubitOrMinusState out = phase ? QubitOrMinusState(XBool(true)) : builder.append_register(1, "output")[0];
    QubitOrTrue control = control_kind == "q" ? QubitOrTrue(builder.append_register(1)[0]) : QubitOrTrue(true);
    auto clean = builder.reserve_qubits(work);
    CircuitGenCtx ctx{.clean_workspace = clean};
    QubitOrBitOrBool equal = control_kind == "b" && (relation == "lt" || relation == "gt") ? QubitOrBitOrBool(builder.append_classical_register(1)[0]) : QubitOrBitOrBool(false);
    if (relation == "lt") gen_flip_if_lt(builder, ctx, left, right, out, equal, control, btol);
    else if (relation == "gt") gen_flip_if_gt(builder, ctx, left, right, out, equal, control, btol);
    else if (relation == "eq") gen_flip_if_eq(builder, ctx, left, right, out, control);
    else if (relation == "ge") gen_flip_if_ge(builder, ctx, left, right, out, control, btol);
    else if (relation == "le") gen_flip_if_le(builder, ctx, left, right, out, control, btol);
    else return 2;
    std::cout << builder.finish_circuit().str();
}
