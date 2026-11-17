#include <cstdlib>
#include <iostream>
#include <vector>
#include "kickmix/gen/qft/gen_qft_demolition_measure.h"

using namespace kickmix;
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    size_t n = std::strtoul(argv[1], nullptr, 10);
    int mode = std::atoi(argv[2]);
    bool inverse = std::atoi(argv[3]);
    CircuitBuilder builder;
    auto q = builder.reserve_qubits(mode < 2 ? n : 3*n+2);
    auto b = builder.reserve_bits(mode < 2 ? n : 2*n+3);
    std::vector<QubitId> target;
    std::vector<BitId> output;
    for (size_t i = 0; i < n; i++) {
        if (mode == 0) {
            target.push_back(q[i]);
            output.push_back(b[i]);
        } else if (mode == 1) {
            target.push_back(q[n-1-i]);
            output.push_back(b[i]);
        } else {
            target.push_back(q[3*i+1]);
            output.push_back(b[2*(n-1-i)+2]);
        }
    }
    gen_qft_demolition_measure(builder, CircuitGenCtx{}, target, output, inverse);
    std::cout << builder.finish_circuit().str();
}
