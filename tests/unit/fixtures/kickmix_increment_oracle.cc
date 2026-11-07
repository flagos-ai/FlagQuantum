#include <cstdlib>
#include <iostream>
#include "kickmix/gen/adders/gen_iadd1.h"
int main(int argc, char **argv) {
    using namespace kickmix;
    if (argc != 4) return 2;
    CircuitBuilder builder;
    auto target = builder.append_register(std::atoi(argv[1]), "target");
    auto flag = builder.append_register(1, "control")[0];
    auto clean = builder.append_register(std::atoi(argv[2]), "clean");
    gen_iadd1(builder, CircuitGenCtx{.clean_workspace=clean}, target,
              std::atoi(argv[3]) ? QubitOrTrue(flag) : QubitOrTrue(true));
    std::cout << builder.finish_circuit().str();
}
