#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include "kickmix/gen/comparators/gen_cmp_low_space.h"
#include "kickmix/gen/comparators/gen_cmp_qq.h"

using namespace kickmix;

static QubitOrBitOrBool operand(CircuitBuilder &builder, std::string_view kind) {
    if (kind == "q") return builder.append_register(1)[0];
    if (kind == "b") return builder.append_classical_register(1)[0];
    return kind == "1";
}

static void diagram(size_t clean_count) {
    CircuitBuilder builder;
    auto target = builder.append_register(10);
    auto masks = builder.append_classical_register_qcarray_result(10);
    auto clean = builder.reserve_qubits(clean_count);
    auto dirty = builder.reserve_qubits(7 - clean_count);
    CircuitGenCtx ctx{.clean_workspace = clean};
    masked_phase_by_prefix_using_dirty_workspace(builder, ctx.with_more_dirty_qubits(dirty), target, masks);
    auto circuit = builder.finish_circuit();
    std::cout << circuit.str() << "\n===DIAGRAM===\n" << circuit.text_diagram();
}

int main(int argc, char **argv) {
    if (argc == 3 && std::string_view(argv[1]) == "diagram") {
        diagram(std::atoi(argv[2]));
        return 0;
    }
    if (argc != 11) return 2;
    bool qq = std::string_view(argv[1]) == "qq";
    size_t n = std::atoi(argv[2]);
    size_t m = std::atoi(argv[3]);
    size_t work = std::atoi(argv[4]);
    bool phase = std::atoi(argv[5]) != 0;
    bool controlled = std::atoi(argv[6]) != 0;
    bool minimize = std::atoi(argv[7]) != 0;
    bool classical_rhs = std::atoi(argv[8]) != 0;
    CircuitBuilder builder;
    auto left = builder.append_register(n);
    std::vector<QubitId> right_q;
    std::vector<BitId> right_b;
    auto constants = array_z::alloc_noinit(m);
    stride_span_z right;
    if (qq) {
        right_q = builder.append_register(m);
    } else if (classical_rhs) {
        right_b = builder.append_classical_register(m);
        right = stride_span_z(right_b);
    } else {
        for (size_t k = 0; k < m; ++k) constants[k] = k % 2 == 0;
        constants.recompute_common_type();
        right = constants;
    }
    QubitOrMinusState out = phase ? QubitOrMinusState(XBool(true)) : builder.append_register(1)[0];
    QubitOrTrue control = controlled ? QubitOrTrue(builder.append_register(1)[0]) : QubitOrTrue(true);
    auto equal = operand(builder, argv[9]);
    auto inverted = operand(builder, argv[10]);
    auto clean = builder.reserve_qubits(work);
    CircuitGenCtx ctx{.clean_workspace = clean, .minimize_qubits = minimize};
    if (qq) {
        gen_flip_if_lt_qq(builder, ctx, left, right_q, out, equal, control);
    } else {
        gen_xif_less_than_3anc_2ntof(builder, ctx, left, right, out, equal, inverted, control);
    }
    std::cout << builder.finish_circuit().str();
}
