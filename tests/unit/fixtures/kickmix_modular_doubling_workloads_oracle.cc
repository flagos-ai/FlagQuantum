#include <iostream>
#include <string_view>
#include "kickmix/gen/modular_arithmetic/gen_imul2_mod.h"

using namespace kickmix;
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    CircuitBuilder builder;
    const bool use_bits = std::string_view(argv[1]) == "bits";
    auto modulus = array_z::copy_of(FixedWidthInt("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"));
    if (use_bits) {
        std::vector<QubitOrBitOrBool> bits{true};
        for (auto bit : builder.reserve_bits(254)) bits.push_back(bit);
        bits.push_back(true);
        modulus = array_z::copy_of(bits);
    }
    auto target = builder.append_register(256);
    auto control = builder.append_register(1)[0];
    auto clean = builder.reserve_qubits(4);
    auto dirty = builder.reserve_qubits(256);
    CircuitGenCtx ctx{.clean_workspace = clean};
    ctx = ctx.with_more_dirty_qubits(dirty);
    builder.mut.clear();
    gen_imul2_mod(builder, ctx, target, modulus, control, 30);
    std::cout << builder.finish_circuit().str();
}
