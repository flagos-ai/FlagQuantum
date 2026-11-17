#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include "kickmix/gen/gf_arithmetic/gen_gf_iadd.h"

using namespace kickmix;

int main(int argc, char **argv) {
    if (argc != 5) return 2;
    size_t n = std::atoi(argv[1]);
    std::string_view kind = argv[2];
    std::string_view control_kind = argv[3];
    int layout = std::atoi(argv[4]);
    int stride = layout == 2 ? 2 : 1;
    CircuitBuilder builder;
    auto target = builder.append_register(n * stride, "target");
    std::vector<QubitId> source_q;
    std::vector<BitId> source_b;
    auto source_mixed = array_z::alloc_noinit(n);
    stride_span_z source;
    if (kind == "q") {
        source_q = builder.append_register(n * stride, "source");
        auto view = stride_span<const QubitId>(source_q.data(), stride, n);
        source = layout == 1 ? view.reversed() : view;
    } else if (kind == "b") {
        source_b = builder.append_classical_register(n, "source");
        source = stride_span_z(source_b);
    } else {
        if (kind == "m") {
            source_q = builder.append_register((n + 3) / 4, "source_q");
            source_b = builder.append_classical_register(2, "source_b");
        }
        for (size_t k = 0; k < n; k++) {
            if (kind == "m" && k % 4 == 0) source_mixed[k] = source_q[k / 4];
            else if (kind == "m" && k % 4 == 1) source_mixed[k] = source_b[(k / 4) % 2];
            else source_mixed[k] = kind != "z" && k % 4 != 3;
        }
        source_mixed.recompute_common_type();
        source = source_mixed;
    }
    auto view = stride_span<const QubitId>(target.data(), stride, n);
    if (control_kind == "q") {
        auto control = builder.append_register(1)[0];
        gen_gf_iadd(builder, CircuitGenCtx{}, view, source, control);
    } else if (control_kind == "b" || control_kind == "alias") {
        // The locked Python binding puts a classical guard outside the generator.
        BitId bit = control_kind == "alias" ? source_b[0] : builder.append_classical_register(1)[0];
        auto guard = builder.raii_push_condition(bit);
        gen_gf_iadd(builder, CircuitGenCtx{}, view, source);
    } else if (control_kind != "false") {
        gen_gf_iadd(builder, CircuitGenCtx{}, view, source);
    }
    std::cout << builder.finish_circuit().str();
}
