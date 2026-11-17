#include <cstdlib>
#include <iostream>
#include "kickmix/gen/gf_arithmetic/gen_gf_div.h"

using namespace kickmix;

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    size_t m = std::atoi(argv[1]);
    int route = std::atoi(argv[2]);
    bool shared = std::atoi(argv[3]) != 0;
    GF2Field field(m);
    CircuitBuilder builder;
    auto lhs = builder.append_register(m);
    auto rhs = shared ? lhs : builder.append_register(m);
    auto target = builder.append_register(m);
    auto clean = builder.reserve_qubits(gf_div_workspace_size(field));
    CircuitGenCtx ctx;
    if (route < 2) ctx.clean_workspace = clean;
    if (route == 0) gen_gf_div(builder, ctx, field, target, lhs, rhs);
    if (route == 1) gen_gf_undiv(builder, ctx, field, target, lhs, rhs);
    if (route == 2) gen_gf_div(builder, ctx, field, target, lhs, rhs, clean);
    if (route == 3) gen_gf_undiv(builder, ctx, field, target, lhs, rhs, clean);
    std::cout << field.modulus().str() << "\n" << clean.size() << "\n" << builder.finish_circuit().str();
}
