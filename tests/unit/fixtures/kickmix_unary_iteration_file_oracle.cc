// Exercise the locked generator without changing its stream.
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include "kickmix/gen/iteration/gen_unary_iteration.h"

using namespace kickmix;

int main(int argc, char **argv) {
    if (argc < 4) return 2;
    const size_t width = std::strtoull(argv[1], nullptr, 10);
    const unsigned mode = std::strtoul(argv[2], nullptr, 10);
    const bool sweep = std::string_view(argv[3]) == "sweep";
    CircuitBuilder builder;
    auto address = builder.append_register(width);
    auto control = builder.append_register(1);
    auto target = sweep ? std::vector<QubitId>{} : builder.append_register(1);
    auto work = builder.reserve_qubits(std::max(size_t{1}, width));
    CircuitGenCtx ctx{.clean_workspace = work};
    const QubitOrTrue enable = mode & 1 ? QubitOrTrue(control[0]) : QubitOrTrue(true);
    std::vector<uint64_t> values;
    for (int k = 4; k < argc; k++) values.push_back(std::strtoull(argv[k], nullptr, 10));
    auto generate = [&]() {
        if (sweep) {
            builder.mut.clear();
            UnaryIterationCursor cursor(builder, ctx, address, enable);
            for (uint64_t value = 0; value < 1024; value++) cursor.move_to(value);
            cursor.close();
        } else {
            gen_unary_iteration(builder, ctx, address, enable, values,
                                [&](uint64_t, QubitId match) { builder.cx(match, target[0]); });
        }
    };
    if (mode & 2) {
        auto guard = builder.append_classical_register(1);
        auto scope = builder.raii_push_condition(guard[0]);
        generate();
    } else {
        generate();
    }
    std::cout << builder.finish_circuit().str();
}
