#include <cstdlib>
#include <iostream>
#include "kickmix/gen/adders/gen_iadd_control.h"
int main(int argc, char **argv) {
    using namespace kickmix;
    if (argc != 6 && argc != 7) return 2;
    size_t n = std::atoi(argv[1]);
    CircuitBuilder builder;
    auto target = builder.append_register(n, "target");
    auto offset = builder.append_register(n + std::atoi(argv[2]), "offset");
    // The width-256 performance workload allocates carry before enable.
    bool benchmark_layout = argc == 7;
    auto first = builder.append_register(1, benchmark_layout ? "carry" : "enable")[0];
    auto second = builder.append_register(1, benchmark_layout ? "enable" : "carry")[0];
    auto enable = benchmark_layout ? second : first;
    auto carry_q = benchmark_layout ? first : second;
    auto clean = builder.append_register((n < 2 ? 0 : n) + std::atoi(argv[3]), "clean");
    int kind = std::atoi(argv[4]);
    auto carry_b = benchmark_layout
        ? BitId{0}
        : builder.append_classical_register(1, "carry_bit")[0];
    QubitOrBitOrBool carry = kind == 1;
    if (kind == 2) carry = carry_q;
    if (kind == 3) carry = carry_b;
    if (std::atoi(argv[5])) {
        gen_isub_control(builder, CircuitGenCtx{.clean_workspace=clean}, target, offset, enable);
    } else {
        gen_iadd_control(builder, CircuitGenCtx{.clean_workspace=clean}, target, offset, carry, enable);
    }
    std::cout << builder.finish_circuit().str();
}
