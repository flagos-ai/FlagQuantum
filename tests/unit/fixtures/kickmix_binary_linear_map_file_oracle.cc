#include <cstdlib>
#include <iostream>
#include <random>
#include "kickmix/gen/gf_arithmetic/gen_linear_map.h"
#include "kickmix/util/gf2_field.h"
using namespace kickmix;
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    size_t m = std::atoi(argv[1]);
    int mode = std::atoi(argv[2]);
    bool adjoint = std::atoi(argv[3]);
    GF2Matrix matrix = GF2Matrix::identity(m);
    if (mode == 1 && m > 1) {
        std::mt19937_64 rand(0x619134AB);
        for (size_t i = 0; i < 5*m; ++i) {
            size_t a = rand() % m, b = rand() % m;
            if (a != b) {
                if ((i & 3) == 0) matrix.swap_rows(a,b);
                else matrix.xor_row_into(a,b);
            }
        }
    }
    if (mode == 2) matrix = GF2Field(m).frobenius_matrix(1);
    if (mode == 3) matrix = GF2Matrix(m,m);
    if (mode == 4 && m) matrix.swap_rows(0,m-1);
    CircuitBuilder builder;
    auto target = builder.append_register(m);
    for (size_t c = 0; c < m; ++c) {
        for (size_t r = 0; r < m; ++r) std::cout << matrix.get(r,c);
        std::cout << '\n';
    }
    try {
        if (adjoint) gen_linear_map_adjoint(builder, CircuitGenCtx{}, target, matrix);
        else gen_linear_map(builder, CircuitGenCtx{}, target, matrix);
    } catch (const std::invalid_argument &) {
        std::cout << "REJECTED\n";
        return 0;
    }
    std::cout << builder.finish_circuit().str();
}
