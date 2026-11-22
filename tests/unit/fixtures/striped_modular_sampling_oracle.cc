#include "kickmix/util/mod_rand.h"

#include <iostream>
#include <vector>

int main() {
    size_t width, output_size, buffer_size, repeats;
    uint64_t seed;
    std::string literal;
    while (std::cin >> width >> literal >> output_size >> buffer_size >> seed >> repeats) {
        try {
            auto modulus = width ? kickmix::FixedWidthInt::from_str_hex(literal, width)
                                 : kickmix::FixedWidthInt(0);
            std::vector<uint64_t> output(output_size, UINT64_MAX);
            std::vector<uint64_t> buffer(buffer_size, UINT64_MAX);
            std::mt19937_64 rng(seed);
            for (size_t repeat = 0; repeat < repeats; repeat++) {
                kickmix::generate_bit_striped_random_values_mod(rng, output, buffer, modulus);
            }
            for (auto word : output) {
                std::cout << word << ' ';
            }
            std::cout << "| ";
            for (auto word : buffer) {
                std::cout << word << ' ';
            }
            std::cout << "| " << rng() << '\n';
        } catch (const std::invalid_argument &) {
            std::cout << "REJECTED\n";
        }
    }
}
