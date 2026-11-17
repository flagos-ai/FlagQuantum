#include <cmath>
#include <iostream>
#include <string_view>
#include "kickmix/gen/modular_arithmetic/gen_iadd_mod.h"

using namespace kickmix;
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    std::string_view kind = argv[1];
    CircuitBuilder builder;
    auto target = builder.append_register(256);
    auto offset = builder.append_register(256);
    array_z modulus = kind == "approx"
        ? array_z::copy_of(FixedWidthInt("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"))
        : builder.append_classical_register_qcarray_result(256);
    if (kind != "approx") {
        modulus.front() = true;
        modulus.back() = true;
        modulus.recompute_common_type();
    }
    auto clean = builder.append_register(256);
    CircuitGenCtx ctx{.clean_workspace = clean};
    builder.mut.clear();
    if (kind == "sub") gen_isub_mod(builder, ctx, target, offset, modulus, true, INFINITY);
    else gen_iadd_mod(builder, ctx, target, offset, modulus, true, INFINITY);
    std::cout << builder.finish_circuit().str();
}
