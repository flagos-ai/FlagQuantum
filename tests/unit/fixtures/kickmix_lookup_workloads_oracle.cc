#include <iostream>
#include <string_view>
#include "kickmix/gen/lookup/gen_lookup.h"

using namespace kickmix;

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    CircuitBuilder builder;
    auto address = builder.append_register(8);
    auto output = builder.append_register(30);
    auto table_bits = builder.append_classical_register(30 << 8);
    auto clean = builder.reserve_qubits(10);
    CircuitGenCtx ctx{.clean_workspace = clean};
    builder.mut.clear();
    if (std::string_view(argv[1]) == "erase") {
        gen_unlookup(builder, ctx, table_bits, address, output);
    } else {
        gen_lookup(builder, ctx, table_bits, address, output, 'X', true);
    }
    std::cout << builder.finish_circuit().str();
}
