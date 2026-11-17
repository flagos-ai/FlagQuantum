#include <cstdlib>
#include <iostream>
#include "kickmix/gen/gf_arithmetic/gen_gf_mul.h"
using namespace kickmix;
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    size_t m = std::atoi(argv[1]);
    int route = std::atoi(argv[2]);
    GF2Field field(m);
    CircuitBuilder builder;
    auto target = builder.append_register(route == 0 ? 2*m - 1 : route >= 5 ? 0 : m);
    auto lhs = builder.append_register(m);
    auto rhs = builder.append_register(m);
    auto control = route == 3 || route == 4 ? QubitOrTrue(builder.append_register(1)[0]) : QubitOrTrue(true);
    auto clean = builder.reserve_qubits(route == 3 || route == 4 ? m : 0);
    CircuitGenCtx ctx{.clean_workspace=clean};
    if (route == 0) gen_gf2_poly_mul(builder, ctx, target, lhs, rhs);
    if (route == 1 || route == 3) gen_gf_mul(builder, ctx, field, target, lhs, rhs, control);
    if (route == 2 || route == 4) gen_gf_unmul(builder, ctx, field, target, lhs, rhs, control);
    if (route == 5) {
        auto mask = builder.append_classical_register(m);
        gen_gf_phase_by_product(builder, ctx, field, mask, lhs, rhs);
    }
    if (route == 6) {
        auto mask = builder.append_classical_register(1);
        gen_gf_phase_by_product(builder, ctx, field, stride_span<const BitId>(mask.data(), 0, m), lhs, rhs);
    }
    std::cout << field.modulus().str() << "\n" << clean.size() << "\n" << builder.finish_circuit().str();
}
