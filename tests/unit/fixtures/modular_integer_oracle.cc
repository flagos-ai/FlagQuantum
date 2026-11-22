// Feed unchanged locked modular scalar sources with explicit constructor routes.
#include "kickmix/util/mod_int.h"

#include <iomanip>
#include <iostream>
#include <sstream>

using kickmix::FixedWidthInt;
using kickmix::ModInt;

FixedWidthInt raw(size_t width, const std::string &text) {
    FixedWidthInt value(width);
    auto parsed = FixedWidthInt::from_str_hex(text);
    for (size_t k = 0; k < std::min(value.num_words, parsed.num_words); k++) {
        value.words[k] = parsed.words[k];
    }
    return value;
}

void output(const ModInt &value, bool failed, uint64_t next_word = 0) {
    bool valid = true;
    try { value.verify_invariants(); } catch (const std::invalid_argument &) { valid = false; }
    std::cout << failed << ' ' << value.value.num_bits << ' ' << std::hex;
    for (size_t k = value.value.num_words; k--;) {
        std::cout << std::setw(16) << std::setfill('0') << value.value.words[k];
    }
    if (!value.value.num_words) { std::cout << '0'; }
    std::cout << std::dec << ' ' << valid << ' ' << bool(value) << ' '
              << next_word << ' ' << std::quoted(value.str()) << '\n';
}

int main() {
    std::string op, a, b, m, n, literal;
    size_t width, bwidth, mwidth;
    int64_t scalar;
    uint64_t seed;
    while (std::cin >> op >> width >> a >> bwidth >> b >> mwidth >> m >> n
                   >> scalar >> seed >> std::quoted(literal)) {
        auto modulus = std::make_shared<FixedWidthInt>(raw(mwidth, m));
        ModInt value(modulus);
        value.value = raw(width, a);
        auto other_modulus = std::make_shared<FixedWidthInt>(raw(bwidth, n));
        ModInt other(other_modulus);
        other.value = raw(bwidth, b);
        std::mt19937_64 rng(seed);
        try {
            if (op == "zero") { value = ModInt(modulus); }
            else if (op == "signed") { value = ModInt(scalar, modulus); }
            else if (op == "text") { value = ModInt(literal, modulus); }
            else if (op == "word") { const auto input = value.value; value = ModInt(input, modulus); }
            else if (op == "word_move") { auto input = value.value; value = ModInt(std::move(input), modulus); }
            else if (op == "copy") { auto copy = value; value = copy; }
            else if (op == "move") { auto copy = std::move(value); value = std::move(copy); }
            else if (op == "assign") { value = other; }
            else if (op == "assign_move") { value = std::move(other); }
            else if (op == "equal") { std::cout << "equal " << (value == other) << '\n'; continue; }
            else if (op == "add") { value += other; }
            else if (op == "add_scalar") { value += scalar; }
            else if (op == "sub") { value -= other; }
            else if (op == "mul") { value *= other; }
            else if (op == "mul_self") { value *= value; }
            else if (op == "mul_scalar") { value *= scalar; }
            else if (op == "square") { value = value.squared(); }
            else if (op == "divide") { value /= other; }
            else if (op == "divide_move") { value /= std::move(other); }
            else if (op == "divide_scalar") { value /= scalar; }
            else if (op == "divide_self") { value /= value; }
            else if (op == "divide_self_move") { value /= std::move(value); }
            else if (op == "negate") { value.negate(); }
            else if (op == "increment") { value.increment(); }
            else if (op == "inverse") { value.invert(); }
            else if (op == "double") { value.idouble(); }
            else if (op == "halve") { value.ihalve(); }
            else if (op == "left64") { value <<= scalar; }
            else if (op == "right64") { value >>= scalar; }
            else if (op == "left32") { value <<= static_cast<int32_t>(scalar); }
            else if (op == "right32") { value >>= static_cast<int32_t>(scalar); }
            else if (op == "left_u64") { value <<= static_cast<uint64_t>(scalar); }
            else if (op == "right_u64") { value >>= static_cast<uint64_t>(scalar); }
            else if (op == "left_u32") { value <<= static_cast<uint32_t>(scalar); }
            else if (op == "right_u32") { value >>= static_cast<uint32_t>(scalar); }
            else if (op == "random") { value = ModInt::random(rng, modulus); }
            else if (op == "randomize") { value.randomize(rng); value.randomize(rng); }
            else if (op != "inspect") { throw std::invalid_argument("Unknown operation"); }
            output(value, false, op == "random" || op == "randomize" ? rng() : 0);
        } catch (const std::invalid_argument &) { output(value, true); }
    }
}
