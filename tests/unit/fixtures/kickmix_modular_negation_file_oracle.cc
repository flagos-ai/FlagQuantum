#include <cstdlib>
#include <iostream>
#include <vector>
#include "kickmix/gen/modular_arithmetic/gen_ineg_mod.h"

using namespace kickmix;
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    size_t n = std::strtoul(argv[1], nullptr, 10);
    int mode = std::atoi(argv[2]);
    size_t dirty_count = std::strtoul(argv[3], nullptr, 10);
    CircuitBuilder builder;
    auto target = builder.append_register(n);
    std::vector<QubitOrBitOrBool> modulus;
    if (mode == 0) {
        for (size_t i = 0; i < n; i++) modulus.push_back(i == 0 || i == n-1 || i % 3 == 0);
    } else if (mode == 1) {
        for (auto bit : builder.append_classical_register(n-1)) modulus.push_back(bit);
        modulus.push_back(true);
    } else {
        for (auto wire : builder.append_register(n-1)) modulus.push_back(wire);
        modulus.push_back(true);
    }
    auto clean = builder.reserve_qubits(n + 2);
    auto dirty = builder.reserve_qubits(dirty_count);
    auto ctx = CircuitGenCtx{.clean_workspace = clean}.with_more_dirty_qubits(dirty);
    try {
        gen_ineg_mod(builder, ctx, target, modulus);
    } catch (const std::exception &) {
        std::cout << "REJECTED\n";
        return 0;
    }
    std::cout << builder.finish_circuit().str();
}
