// Execute the complete original header with explicit lifecycle observations.
#include "kickmix/simd/simd.h"
#include "kickmix/util/chacha.h"

#include <iomanip>
#include <iostream>
#include <vector>

using namespace kickmix;

template<typename W> void print_word(W word) {
    std::cout << std::hex;
    for (size_t k = W::BIT_SIZE / 64; k--;) {
        std::cout << std::setw(16) << std::setfill('0') << word.u64(k);
    }
    std::cout << std::dec << ' ';
}

template<typename W> W rotate(W word, size_t amount) {
    switch (amount) {
#define ROTATE_CASE(K) case K: return rol32<K>(word);
        ROTATE_CASE(1) ROTATE_CASE(2) ROTATE_CASE(3) ROTATE_CASE(4)
        ROTATE_CASE(5) ROTATE_CASE(6) ROTATE_CASE(7) ROTATE_CASE(8)
        ROTATE_CASE(9) ROTATE_CASE(10) ROTATE_CASE(11) ROTATE_CASE(12)
        ROTATE_CASE(13) ROTATE_CASE(14) ROTATE_CASE(15) ROTATE_CASE(16)
        ROTATE_CASE(17) ROTATE_CASE(18) ROTATE_CASE(19) ROTATE_CASE(20)
        ROTATE_CASE(21) ROTATE_CASE(22) ROTATE_CASE(23) ROTATE_CASE(24)
        ROTATE_CASE(25) ROTATE_CASE(26) ROTATE_CASE(27) ROTATE_CASE(28)
        ROTATE_CASE(29) ROTATE_CASE(30) ROTATE_CASE(31)
#undef ROTATE_CASE
    }
    throw std::invalid_argument("Unsupported rotation");
}

template<typename W, size_t R> void run(uint64_t seed, int mode, size_t count) {
    ChaCha<W, R> stream;
    std::mt19937_64 rng(seed);
    if (mode == 1 || mode == 5 || mode == 6) { stream.randomize(rng); }
    if (mode == 2 || mode == 7 || mode == 8 || mode == 10 || mode == 11) {
        for (auto &word : stream.state) { word.randomize(rng); }
        for (auto &word : stream.buffer) { word.randomize(rng); }
    }
    if (mode == 2) { stream.state[12] = W::from_u32_broadcast(UINT32_MAX); }
    if (mode == 5) {
        for (size_t k = 0; k < 7; k++) { stream.next(); }
        stream.randomize(rng);
    }
    if (mode == 6) { stream.refill(); stream.refill(); }
    if (mode == 7) { stream.buffer_idx = 0; }
    if (mode == 8) { stream.buffer_idx = SIZE_MAX; }
    if (mode == 10) {
        chacha_quarter_round(stream.state[seed & 3], stream.state[(seed >> 2) & 3],
                             stream.state[(seed >> 4) & 3], stream.state[(seed >> 6) & 3]);
    }
    if (mode == 11) { stream.state[0] = rotate(stream.state[0], count); }
    std::vector<W> output;
    if (mode < 10) {
        for (size_t k = 0; k < count; k++) { output.push_back(stream.next()); }
    }
    std::cout << stream.buffer_idx << ' ' << rng() << ' ';
    for (auto word : stream.state) { print_word(word); }
    for (auto word : stream.buffer) { print_word(word); }
    for (auto word : output) { print_word(word); }
    std::cout << '\n';
}

template<typename W> void dispatch(size_t rounds, uint64_t seed, int mode, size_t count) {
    switch (rounds) {
        case 2: run<W, 2>(seed, mode, count); break;
        case 8: run<W, 8>(seed, mode, count); break;
        case 12: run<W, 12>(seed, mode, count); break;
        case 20: run<W, 20>(seed, mode, count); break;
        default: throw std::invalid_argument("Unsupported fixture round count");
    }
}

int main() {
    size_t width, rounds, count;
    uint64_t seed;
    int mode;
    while (std::cin >> width >> rounds >> seed >> mode >> count) {
        if (width == 64) { dispatch<b64_polyfill>(rounds, seed, mode, count); }
        if (width == 128) { dispatch<b128_polyfill>(rounds, seed, mode, count); }
        if (width == 256) { dispatch<b256_polyfill>(rounds, seed, mode, count); }
        if (width == 512) { dispatch<b512_polyfill>(rounds, seed, mode, count); }
    }
}
