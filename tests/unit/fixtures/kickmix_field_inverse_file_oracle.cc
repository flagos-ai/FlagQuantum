#include <cstdlib>
#include <iostream>
#include "kickmix/gen/gf_arithmetic/gen_gf_inverse.h"

using namespace kickmix;

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    size_t m = std::atoi(argv[1]);
    int route = std::atoi(argv[2]);
    GF2Field field(m);
    CircuitBuilder builder;
    auto input = builder.append_register(m);
    auto target = builder.append_register(m);
    auto workspace = route < 2
        ? builder.reserve_qubits(gf_inverse_workspace_size(field))
        : builder.append_register(gf_inverse_chain_size(field));
    CircuitGenCtx ctx;
    if (route < 2) ctx.clean_workspace = workspace;
    if (route == 0) gen_gf_inverse(builder, ctx, field, target, input);
    if (route == 1) gen_gf_uninverse(builder, ctx, field, target, input);
    if (route == 2) gen_gf_inverse(builder, ctx, field, target, input, workspace);
    if (route == 3) gen_gf_uninverse(builder, ctx, field, target, input, workspace);
    std::cout << field.modulus().str() << "\n" << workspace.size() << "\n" << builder.finish_circuit().str();
}
