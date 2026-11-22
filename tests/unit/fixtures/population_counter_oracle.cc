// Observe the original template and all portable packed widths unchanged.
#include "kickmix/simd/simd.h"
#include "kickmix/util/pop_counter.h"

#include <chrono>
#include <iomanip>
#include <iostream>

using namespace kickmix;

template<typename W> void print_word(W value) {
    std::cout << std::hex;
    for (size_t k = W::BIT_SIZE / 64; k--;) {
        std::cout << std::setw(16) << std::setfill('0') << value.u64(k);
    }
    std::cout << std::dec << ' ';
}

template<typename W> void run(uint64_t seed, size_t repeats, int mode, bool clear) {
    PopCounter<W> counter;
    std::mt19937_64 rng(seed);
    W mask{};
    if (mode == 4) {
        counter.bit_totals[5] = ~W{};
        counter.paired_totals[1] = ~W{};
        counter.shared_count = UINT32_MAX;
        counter.step = 255;
    }
    if (mode == 5) { counter.bit_totals[0] = W::from_u32_broadcast(0x12345678); }
    auto start = std::chrono::steady_clock::now();
    for (size_t k = 0; k < repeats; k++) {
        if (mode == 0) { mask = W{}; }
        if (mode == 1) { mask = ~W{}; }
        if (mode == 2) { mask = W::random(rng); }
        if (mode == 3) {
            mask ^= W::from_u32_broadcast(0x11111111ULL);
            mask.u64_iadd(W::from_u32_broadcast(0xBCDEF012ULL));
        }
        if (mode == 5) {
            counter.masked_increments(counter.bit_totals[0]);
        } else { counter.masked_increments(mask); }
    }
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (clear) { counter.clear(); }
    std::cout << unsigned(counter.step) << ' ' << counter.shared_count << ' ' << rng() << ' ';
    for (auto word : counter.bit_totals) { print_word(word); }
    for (auto word : counter.paired_totals) { print_word(word); }
    for (size_t k = 0; k < W::BIT_SIZE; k++) { std::cout << counter.compute_total(k) << ' '; }
    std::cout << '\n';
    std::cerr << W::BIT_SIZE << ' ' << mode << ' ' << repeats << ' ' << elapsed << '\n';
}

int main() {
    size_t width, repeats;
    uint64_t seed;
    int mode;
    bool clear;
    while (std::cin >> width >> seed >> repeats >> mode >> clear) {
        if (width == 64) { run<b64_polyfill>(seed, repeats, mode, clear); }
        if (width == 128) { run<b128_polyfill>(seed, repeats, mode, clear); }
        if (width == 256) { run<b256_polyfill>(seed, repeats, mode, clear); }
        if (width == 512) { run<b512_polyfill>(seed, repeats, mode, clear); }
    }
}
