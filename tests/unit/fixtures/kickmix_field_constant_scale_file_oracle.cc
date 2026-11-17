#include <cstdlib>
#include <iostream>
#include <string_view>
#include "kickmix/gen/gf_arithmetic/gen_gf_imul_classical.h"

using namespace kickmix;

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    size_t width = std::atoi(argv[1]);
    bool divide = std::atoi(argv[2]) != 0;
    GF2Field field(width);
    GF2Poly factor = GF2Poly::from_str(std::string_view(argv[3]));
    CircuitBuilder builder;
    auto target = builder.append_register(width);
    std::cout << field.modulus().str() << '\n';
    try {
        if (divide) gen_gf_idiv_classical(builder, CircuitGenCtx{}, field, target, factor);
        else gen_gf_imul_classical(builder, CircuitGenCtx{}, field, target, factor);
    } catch (const std::invalid_argument &) {
        std::cout << "REJECTED\n";
        return 0;
    }
    std::cout << builder.finish_circuit().str();
}
