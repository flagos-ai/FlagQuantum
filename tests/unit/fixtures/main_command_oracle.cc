// SPDX-License-Identifier: Apache-2.0
// Link unchanged frozen main.cc to command probes at its declared boundary.
#include <iomanip>
#include <iostream>
#include <sstream>
#define main frozen_command_main
#include "kickmix/main.cc"
#undef main

static int forward(std::string_view name, int argc, const char **argv) {
    if (argc > 2 && std::string_view(argv[2]) == "throw") {
        throw std::invalid_argument("command failure");
    }
    std::cout << name << "\n";
    for (int index = 2; index < argc; index++) {
        for (unsigned char byte : std::string_view(argv[index])) {
            std::cout << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
        }
        std::cout << std::dec << "\n";
    }
    return 7;
}

namespace kickmix {
int main_generate_circuit(int argc, const char **argv) { return forward("circuit", argc, argv); }
int main_count_operations(int argc, const char **argv) { return forward("count_operations", argc, argv); }
int main_kmb2kmx(int argc, const char **argv) { return forward("kmb2kmx", argc, argv); }
int main_kmx2kmb(int argc, const char **argv) { return forward("kmx2kmb", argc, argv); }
int main_count_operations_sampled(int argc, const char **argv) { return forward("count_operations_sampled", argc, argv); }
int main_sample(int argc, const char **argv) { return forward("sample", argc, argv); }
int main_spin(int argc, const char **argv) { return forward("spin", argc, argv); }
int main_diagram(int argc, const char **argv) { return forward("diagram", argc, argv); }
}
int main(int argc, const char **argv) { return frozen_command_main(argc, argv); }
