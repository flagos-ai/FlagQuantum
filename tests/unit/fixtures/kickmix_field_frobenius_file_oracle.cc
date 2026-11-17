#include <cstdlib>
#include <iostream>
#include "kickmix/gen/gf_arithmetic/gen_gf_ifrobenius.h"

using namespace kickmix;

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    size_t m = std::atoi(argv[1]);
    size_t power = std::strtoull(argv[2], nullptr, 10);
    bool adjoint = std::atoi(argv[3]) != 0;
    GF2Field field(m);
    CircuitBuilder builder;
    auto target = builder.append_register(m);
    if (adjoint) gen_gf_ifrobenius_adjoint(builder, CircuitGenCtx{}, field, target, power);
    else gen_gf_ifrobenius(builder, CircuitGenCtx{}, field, target, power);
    std::cout << field.modulus().str() << "\n" << builder.finish_circuit().str();
}
