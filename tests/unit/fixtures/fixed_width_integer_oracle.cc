// Compare native scalar inputs against the unchanged locked implementation.
#include "kickmix/util/fixed_width_int.h"

#include <iomanip>
#include <iostream>
#include <sstream>

using kickmix::FixedWidthInt;

FixedWidthInt raw_value(size_t width, const std::string &text) {
    FixedWidthInt result(width);
    auto source = FixedWidthInt::from_str_hex(text);
    for (size_t index = 0; index < std::min(result.num_words, source.num_words); index++) {
        result.words[index] = source.words[index];
    }
    return result;
}

void output(const FixedWidthInt &value, bool flag, const std::string &suffix = "") {
    std::cout << value.num_bits << ' ' << std::hex;
    if (!value.num_words) {
        std::cout << '0';
    }
    for (size_t index = value.num_words; index--;) {
        std::cout << std::setfill('0') << std::setw(16) << value.words[index];
    }
    std::cout << std::dec << ' ' << flag << suffix << '\n';
}

int main() {
    std::string operation, literal, right_literal, third_literal;
    size_t width, right_width, third_width;
    uint64_t scalar;
    int64_t shift;
    int alias;
    while (std::cin >> operation >> width >> std::quoted(literal)
                   >> right_width >> right_literal >> third_width >> third_literal
                   >> scalar >> shift >> alias) {
        try {
            if (operation == "parse") {
                std::optional<size_t> declared = alias ? std::optional<size_t>(width) : std::nullopt;
                FixedWidthInt parsed;
                if (scalar == 2) {
                    parsed = FixedWidthInt::from_str_binary(literal, declared);
                } else if (scalar == 16) {
                    parsed = FixedWidthInt::from_str_hex(literal, declared);
                } else if (scalar == 10) {
                    parsed = FixedWidthInt::from_str_dec(literal, declared);
                } else {
                    parsed = FixedWidthInt::from_str(literal, declared);
                }
                output(parsed, false);
                continue;
            }
            auto target = raw_value(width, literal);
            auto right = raw_value(right_width, right_literal);
            auto third_value = raw_value(third_width, third_literal);
            const auto &other = alias == 1 ? target : right;
            const auto &third = alias == 2 ? target : third_value;
            bool flag = false;
            if (operation == "random" || operation == "random_mod" || operation == "random_coprime") {
                std::mt19937_64 rng(scalar);
                if (operation == "random") {
                    target.randomize(rng);
                } else if (operation == "random_mod") {
                    target.randomize_mod(rng, other);
                } else {
                    target.randomize_coprime_mod(rng, other);
                }
                output(target, false, " " + std::to_string(rng()));
                continue;
            }
            if (operation.starts_with("perf_")) {
                for (int64_t iteration = 0; iteration < shift; iteration++) {
                    if (operation == "perf_scalar") {
                        target *= scalar;
                    } else if (operation == "perf_product") {
                        target.iadd_product(other, third);
                    } else if (operation == "perf_product_scalar") {
                        target.iadd_product(scalar, other);
                    } else if (operation == "perf_remainder") {
                        target ^= third;
                        target %= other;
                    } else {
                        throw std::invalid_argument("Unknown workload");
                    }
                }
                output(target, false);
                continue;
            }
            if (operation == "add") {
                flag = target.iadd_carry(other, scalar !=  0);
            }
            else if (operation == "sub") {
                flag = target.isub_borrow(other);
            }
            else if (operation == "scalar_carry") {
                flag = target.iadd_carry(scalar, shift !=  0);
            }
            else if (operation == "scalar_add") {
                target+= scalar;
            }
            else if (operation == "scalar_sub") {
                target-= scalar;
            }
            else if (operation == "signed_add") {
                target+= static_cast<int64_t>(scalar);
            }
            else if (operation == "signed_sub") {
                target-= static_cast<int64_t>(scalar);
            }
            else if (operation == "increment") {
                flag = target.increment(scalar);
            }
            else if (operation == "decrement") {
                flag = target.decrement(scalar);
            }
            else if (operation == "clip") {
                flag = target.clip_hanging_bits();
            }
            else if (operation == "write") {
                target.write_from(other);
            }
            else if (operation == "and") {
                target&= other;
            }
            else if (operation == "or") {
                target|= other;
            }
            else if (operation == "xor") {
                target^= other;
            }
            else if (operation == "xor_int") {
                target^= static_cast<int>(scalar);
            }
            else if (operation == "left") {
                target<<= shift;
            }
            else if (operation == "right") {
                target>>= shift;
            }
            else if (operation == "unsigned_left") {
                target<<= scalar;
            }
            else if (operation == "unsigned_right") {
                target>>= scalar;
            }
            else if (operation == "unsigned_read") {
                target = FixedWidthInt(64, target.read_bit_shifted_word(static_cast<size_t>(scalar)));
            }
            else if (operation == "full_left") {
                target = target<<shift;
            }
            else if (operation == "full_right") {
                target = target>>shift;
            }
            else if (operation == "signed_right") {
                target.isigned_right_shift(static_cast<uint64_t>(shift));
            }
            else if (operation == "add_shifted") {
                target.iadd_shifted(other, shift);
            }
            else if (operation == "sub_shifted") {
                target.isub_shifted(other, shift, &flag);
            }
            else if (operation == "multiply") {
                target*= other;
            }
            else if (operation == "multiply_scalar") {
                target*= scalar;
            }
            else if (operation == "product") {
                target.iadd_product(other, third);
            }
            else if (operation == "product_scalar") {
                target.iadd_product(scalar, other);
            }
            else if (operation == "product_shifted") {
                target.iadd_product_shifted(scalar, other, shift);
            }
            else if (operation == "subtract_product") {
                target.isub_product(scalar, other);
            }
            else if (operation == "subtract_product_shifted") {
                target.isub_product_shifted(scalar, other, shift);
            }
            else if (operation == "times5") {
                flag = target.inplace_times5();
            }
            else if (operation == "div5") {
                target.inplace_div5();
            }
            else if (operation == "negate") {
                target.negate();
            }
            else if (operation == "sum") {
                target = target+other;
            }
            else if (operation == "difference") {
                target = target-other;
            }
            else if (operation == "negative") {
                target = -target;
            }
            else if (operation == "full_product") {
                target = target*other;
            }
            else if (operation == "remainder") {
                target%= other;
            }
            else if (operation == "mod") {
                target = target%other;
            }
            else if (operation == "mod_add") {
                target.iadd_mod(other, third);
            }
            else if (operation == "mod_sub") {
                target.isub_mod(other, third);
            }
            else if (operation == "mod_double") {
                target.idouble_mod(other);
            }
            else if (operation == "mod_half") {
                target.ihalve_mod(other);
            }
            else if (operation == "mod_negate") {
                target.negate_mod(other);
            }
            else if (operation == "inverse") {
                target = target.mult_inverse(other);
            }
            else if (operation == "coprime") {
                flag = target.is_coprime_to(other);
            }
            else if (operation == "shrink") {
                target.resize_shrink(scalar);
            }
            else if (operation == "clear") {
                target.resize_clear(scalar);
            }
            else if (operation == "set") {
                target.set_bit(scalar);
            }
            else if (operation == "reset") {
                target.reset_bit(scalar);
            }
            else if (operation == "flip") {
                target.flip_bit(scalar);
            }
            else if (operation == "assign") {
                target = scalar;
            }
            else if (operation == "assign_signed") {
                target = static_cast<int64_t>(scalar);
            }
            else if (operation == "ctor") {
                target = FixedWidthInt(width, scalar);
            }
            else if (operation == "ctor_signed") {
                target = FixedWidthInt(width, static_cast<int64_t>(scalar));
            }
            else if (operation == "read") {
                target = FixedWidthInt(64, target.read_bit_shifted_word(shift));
            } else if (operation == "format") {
                std::cout << std::quoted(target.bin()) << ' ' << std::quoted(target.hex())
                          << ' ' << std::quoted(target.str()) << ' ' << std::quoted(target.decimal()) << '\n';
                continue;
            } else if (operation == "float") {
                std::cout << std::hexfloat << static_cast<double>(target) << ' '
                          << target.to_approx_mantissa() << std::defaultfloat << '\n';
                continue;
            } else if (operation == "compare") {
                std::cout << (target == other) << ' ' << (target < other) << ' ' << (target <= other)
                          << ' ' << (target == static_cast<int64_t>(scalar))
                          << ' ' << (target < static_cast<int64_t>(scalar))
                          << ' ' << (target <= static_cast<int64_t>(scalar)) << '\n';
                continue;
            } else {
                throw std::invalid_argument("Unknown operation");
            }
            output(target, flag);
        } catch (const std::invalid_argument &) {
            std::cout << "REJECTED\n";
        }
    }
}
