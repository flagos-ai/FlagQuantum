// Compare caller-owned spans without changing the frozen sampling implementation.
#include "kickmix/util/mod_rand.h"
#include <algorithm>
#include <bit>
#include <sstream>
#include <iostream>
#include <vector>

int main() {
    size_t width, output_offset, scratch_offset, repeats;
    uint64_t modulus_value, seed;
    while (std::cin >> width >> modulus_value >> seed >> output_offset >> scratch_offset >> repeats) {
        std::ostringstream literal;
        literal << std::hex << modulus_value;
        auto modulus = kickmix::FixedWidthInt::from_str_hex(literal.str(), width);
        // Detect the original memcpy's undefined overlap before invoking the source.
        std::mt19937_64 probe(seed);
        bool undefined = false;
        for (size_t repeat = 0; repeat < repeats; repeat++) {
            size_t total = 0;
            while (total < 64) {
                uint64_t less = 0, different = 0;
                for (size_t k = width; k--;) {
                    auto word = probe();
                    if (modulus[k]) { less |= ~word & ~different; different |= ~word; }
                    else { different |= word; }
                }
                auto gained = std::popcount(less);
                if (gained == 64) {
                    undefined |= output_offset < scratch_offset + width && scratch_offset < output_offset + width;
                    break;
                }
                total += gained;
            }
        }
        if (undefined) { std::cout << "UNDEFINED_COPY\n"; continue; }
        std::vector<uint64_t> words(std::max(output_offset, scratch_offset) + width + 2, UINT64_MAX);
        std::mt19937_64 rng(seed);
        for (size_t repeat = 0; repeat < repeats; repeat++) {
            kickmix::generate_bit_striped_random_values_mod(rng,
                std::span(words).subspan(output_offset, width),
                std::span(words).subspan(scratch_offset, width), modulus);
        }
        for (auto word : words) std::cout << word << ' ';
        std::cout << "| " << rng() << '\n';
    }
}
