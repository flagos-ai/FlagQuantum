// SPDX-License-Identifier: Apache-2.0
// Observe the unchanged locked builder implementation through its public header.

#include <iostream>

#include "kickmix/build/circuit_builder.h"

using namespace kickmix;

void show(const char *name, const CircuitBuilder &builder) {
    std::cout << name << "\nnext " << builder.next_bit_id << "\nfree";
    for (auto bit : builder.raii_bits) {
        std::cout << ' ' << bit.untagged_id();
    }
    std::cout << "\nops";
    builder.mut.op_types.iter_spans([](auto values) {
        for (auto operation : values) {
            std::cout << ' ' << static_cast<int>(operation);
        }
    });
    auto show_operands = [](auto values) {
        for (auto operand : values) {
            std::cout << ' ' << (operand & UNTAGGED_MASK);
        }
    };
    std::cout << "\nb0";
    builder.mut.b0.iter_spans(show_operands);
    std::cout << "\nbc";
    builder.mut.bc.iter_spans(show_operands);
    std::cout << "\nq0";
    builder.mut.q0.iter_spans(show_operands);
    std::cout << '\n';
}

int main() {
    CircuitBuilder builder;
    builder.reserve_bits(2);
    {
        auto dirty = builder.alloc_dirty_raii_bit();
        builder.bit_store1(dirty.bit);
    }
    show("dirty_released", builder);
    {
        auto clean = builder.alloc_clean_raii_bit();
        builder.x_if(QubitId{0}, clean.bit);
    }
    show("clean_released", builder);
    {
        auto first = builder.alloc_dirty_raii_bit();
        auto second = builder.alloc_dirty_raii_bit();
        builder.bit_store1(second.bit);
    }
    show("nested_released", builder);
    {
        auto measured = builder.hmr_raii_push_condition(QubitId{1});
        builder.z(QubitId{2});
    }
    show("measured_released", builder);
    {
        auto detached = builder.alloc_dirty_raii_bit();
        auto value = detached.disarm();
        builder.bit_store0(value);
    }
    show("detached", builder);
    {
        auto remaining = builder.alloc_dirty_raii_bit();
        builder.debug_print(remaining.bit);
    }
    show("remaining", builder);
    CircuitBuilder fresh;
    {
        auto clean = fresh.alloc_clean_raii_bit();
        fresh.x_if(QubitId{0}, clean.bit);
    }
    show("fresh_clean", fresh);
}
