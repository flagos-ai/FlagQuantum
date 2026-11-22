// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Probe unmodified locked Kickmix SIMD implementations.
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include "kickmix/simd/simd.h"

using namespace kickmix;

template <typename W> std::string hex_word(const W &word) {
    std::stringstream text;
    for (size_t k = W::BIT_SIZE / 64; k--;) {
        text << std::hex << std::setfill('0') << std::setw(16) << word.u64(k);
    }
    return text.str();
}

template <typename W> void probe() {
    std::mt19937_64 rng(123);
    for (size_t sample = 0; sample < 8; sample++) {
        std::array<uint64_t, W::BIT_SIZE / 64> av{}, bv{};
        for (size_t k = 0; k < av.size(); k++) {
            av[k] = sample == 0 ? 0 : sample == 1 ? UINT64_MAX : rng();
            bv[k] = sample == 0 ? UINT64_MAX : sample == 1 ? 1 : rng();
        }
        if (sample == 2 || sample == 4) {
            for (size_t k = 0; k < av.size(); k++) if (sample == 4 || k % 2 == 0) bv[k] = av[k];
        }
        if (sample == 3) for (size_t k = 0; k < av.size(); k++) bv[k] = av[k] ^ (uint64_t{1} << 32);
        W a = W::from_u64s(av), b = W::from_u64s(bv);
        auto row = [&](std::string op, size_t argument, const W &value) {
            std::cout << W::NAME << '\t' << W::BIT_SIZE << '\t' << sample << '\t'
                      << op << '\t' << argument << '\t' << hex_word(a) << '\t'
                      << hex_word(b) << '\t' << hex_word(value) << '\n';
        };
        std::cout << "input64\t" << W::BIT_SIZE << '\t';
        for (uint64_t value : av) std::cout << value << ',';
        std::cout << '\t' << hex_word(a) << '\n';
        row("from_u64s", 0, a);
        row("xor", 0, a ^ b); row("and", 0, a & b); row("or", 0, a | b);
        row("not", 0, ~a);
        W temp = a; temp ^= b; row("ixor", 0, temp);
        temp = a; temp &= b; row("iand", 0, temp);
        temp = a; temp |= b; row("ior", 0, temp);
        row("add64", 0, a.u64_add(b));
        temp = a; temp.u64_iadd(b); row("iadd64", 0, temp);
        row("add32", 0, a.u32_add(b)); row("eq32", 0, a.u32_eq(b));
        row("broadcast", uint32_t(av[0]), W::from_u32_broadcast(uint32_t(av[0])));
        temp.clear_to_zero(); row("zero", 0, temp);
        temp.clear_to_max(); row("max", 0, temp);
        for (size_t shift = 0; shift < 64; shift++) {
            if (sample > 1 && shift != 0 && shift != 1 && shift != 31 && shift != 32 && shift != 63) continue;
            row("right64", shift, a.u64_right_shift(shift));
            row("left64", shift, a.u64_left_shift(shift));
            if (shift < 32) {
                row("right32", shift, a.u32_right_shift(shift));
                row("left32", shift, a.u32_left_shift(shift));
            }
        }
#ifdef __SSE__
        if constexpr (std::is_same_v<W, b128_sse>) {
            for (size_t shift : {size_t{32}, size_t{64}, size_t{255}, size_t{1} << 32, (size_t{1} << 32) + 1, SIZE_MAX}) {
                row("right64", shift, a.u64_right_shift(shift));
                row("left64", shift, a.u64_left_shift(shift));
                row("right32", shift, a.u32_right_shift(shift));
                row("left32", shift, a.u32_left_shift(shift));
            }
        }
#endif
        for (size_t bit = 0; bit < W::BIT_SIZE; bit++) {
            if (sample && bit != 0 && bit != 1 && bit != 31 && bit != 32 && bit != 63 && bit != W::BIT_SIZE - 1) continue;
            temp = a; temp.set_bit(bit, false); row("clear_bit", bit, temp);
            temp = a; temp.set_bit(bit, true); row("set_bit", bit, temp);
        }
        std::cout << "lanes32\t" << W::NAME << '\t' << W::BIT_SIZE << '\t' << hex_word(a) << '\t';
        for (size_t k = 0; k < W::BIT_SIZE / 32; k++) std::cout << a.u32(k) << ',';
        std::cout << '\n';
        if constexpr (W::BIT_SIZE == 64) std::cout << "str\t" << W::BIT_SIZE << '\t' << hex_word(a) << '\t' << a.str() << '\n';
        std::stringstream diagnostic; diagnostic << a;
        std::cout << "diagnostic\t" << W::NAME << '\t' << W::BIT_SIZE << '\t' << hex_word(a)
                  << '\t' << hex_word(b) << '\t' << a.non_zero() << '\t' << (a == a) << '\t' << (a == b)
                  << '\t' << diagnostic.str() << '\n';
    }
    for (uint64_t seed : {uint64_t{0}, uint64_t{1}, UINT64_MAX}) {
        std::mt19937_64 stream(seed);
        W value = W::random(stream);
        std::cout << "random\t" << W::NAME << '\t' << W::BIT_SIZE << '\t' << seed
                  << '\t' << hex_word(value) << '\t' << stream() << '\n';
        value.randomize(stream);
        std::cout << "randomize\t" << W::NAME << '\t' << W::BIT_SIZE << '\t' << seed
                  << '\t' << hex_word(value) << '\t' << stream() << '\n';
    }
}

int main() {
    probe<b64_polyfill>(); probe<b128_polyfill>(); probe<b256_polyfill>(); probe<b512_polyfill>();
#ifdef __SSE__
    probe<b128_sse>();
#endif
    std::cout << "aliases\t" << b64::NAME << '\t' << b128::NAME << '\t' << b256::NAME << '\t' << b512::NAME << '\n';
}
