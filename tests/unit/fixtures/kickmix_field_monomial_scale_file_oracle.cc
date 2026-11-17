#include <cstdlib>
#include <iostream>
#include <string_view>
#include "kickmix/gen/gf_arithmetic/gen_gf_imul_x.h"

using namespace kickmix;

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    size_t width = std::atoi(argv[1]);
    bool divide = std::atoi(argv[2]) != 0;
    GF2Field field(width);
    size_t power = std::strtoull(argv[3], nullptr, 10);
    CircuitBuilder builder;
    auto target = builder.append_register(width);
    try {
        if (divide) gen_gf_idiv_x(builder, CircuitGenCtx{}, field, target, power);
        else gen_gf_imul_x(builder, CircuitGenCtx{}, field, target, power);
    } catch (const std::invalid_argument &) {
        std::cout << "REJECTED\n";
        return 0;
    }
    std::cout << field.modulus().str() << "\n" << builder.finish_circuit().str();
}
