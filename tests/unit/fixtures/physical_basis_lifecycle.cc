// SPDX-License-Identifier: Apache-2.0
// Exercise unchanged locked Kickmix simulator methods.
#include <iostream>
#include <sstream>

#include "kickmix/sim/sim.h"
#include "kickmix/simd/simd.h"

using namespace kickmix;

template <typename W>
void lifecycle() {
    std::cout << "width:" << W::BIT_SIZE << '\n';
    Sim<W, true> sim(std::mt19937_64{123});
    sim.configure_for(Circuit("PUSH_CONDITION if b0\nX q0"));
    sim.bit_span()[0] = W::from_u32_broadcast(0xaaaaaaaa);
    sim.apply(Circuit("PUSH_CONDITION if b0\nX q0"));
    sim.apply(Circuit("X q0"));
    std::cout << "retained:" << sim.qubit_span()[0] << ':'
              << sim.condition_stack.size() << '\n';
    sim.apply(Circuit("POP_CONDITION\nX q0"));
    std::cout << "popped:" << sim.qubit_span()[0] << ':'
              << sim.condition_stack.size() << '\n';
    sim.global_phase_ref() = W::from_u32_broadcast(0x55555555);
    sim.angles[0] = FixedPrecisionAngle128::from_half_turns_exact_double(0.125);
    sim.ensure_big_enough_state_for(3, 0);
    std::cout << "grown:" << sim.num_qubits << ':' << sim.num_bits << ':'
              << sim.global_phase_ref() << ':' << sim.qubit_span()[0] << ':'
              << sim.bit_span()[0] << ':' << sim.read_shot_phase(0) << '\n';
    sim.ensure_big_enough_state_for(0, 3);
    std::cout << "grown2:" << sim.num_qubits << ':' << sim.num_bits << ':'
              << sim.global_phase_ref() << ':' << sim.qubit_span()[0] << ':'
              << sim.bit_span()[0] << ':' << sim.read_shot_phase(0) << '\n';

    Circuit debug("DEBUG_PRINT q0 if b0\nNEG\nDEBUG_PRINT\nZ_POW q0 0.25\nDEBUG_PRINT");
    sim.configure_for(debug);
    sim.qubit_span()[0] = ~W{};
    size_t calls = 0;
    sim.debug_print_handler = [&](QubitOrBitOrBool target, const W &value,
                                  BitOrBool condition, const W &active) {
        std::cout << "debug:" << ++calls << ':' << target << ':' << value << ':'
                  << condition << ':' << active << ':'
                  << sim.read_shot_phase(0) << '\n';
    };
    sim.apply(debug);
    std::cout << "phase:" << sim.read_shot_phase(0) << '\n';
    sim.clear_for_shot();
    std::cout << "cleared:" << sim.global_phase_ref() << ':' << sim.qubit_span()[0]
              << ':' << sim.read_shot_phase(0) << ':' << sim.condition_stack.size()
              << ':' << sim.new_op_counters[(uint8_t)OpType::X].compute_total(0) << '\n';
    sim.clear_operation_statistics();
    std::cout << "counts_clear:"
              << sim.new_op_counters[(uint8_t)OpType::X].compute_total(0) << '\n';
}

template <typename W>
void copies() {
    Sim<W, true> original(std::mt19937_64{123});
    original.configure_for(Circuit("X q0"));
    original.apply(Circuit("X q0"));
    auto cloned = original;
    auto assigned = original;
    std::cout << "copy:" << W::BIT_SIZE << ':'
              << (original.rng() == cloned.rng()) << ':'
              << (original.chacha_rng.next() == cloned.chacha_rng.next()) << ':'
              << (assigned.qubit_span()[0] == original.qubit_span()[0]) << '\n';
    cloned.apply(Circuit("X q0"));
    std::cout << "isolated:"
              << (cloned.qubit_span()[0] != original.qubit_span()[0]) << ':'
              << original.new_op_counters[(uint8_t)OpType::X].compute_total(0) << ':'
              << cloned.new_op_counters[(uint8_t)OpType::X].compute_total(0) << '\n';
    original.debug_print_handler = [](QubitOrBitOrBool, const W &, BitOrBool, const W &) {
        throw std::runtime_error("stop");
    };
    try {
        original.apply(Circuit("NEG\nDEBUG_PRINT\nX q0"));
    } catch (const std::runtime_error &) {
    }
    std::cout << "throw:" << original.read_shot_phase(0) << ':'
              << original.new_op_counters[(uint8_t)OpType::NEG].compute_total(0) << ':'
              << original.new_op_counters[(uint8_t)OpType::DEBUG_PRINT_EMPTY].compute_total(0) << ':'
              << original.new_op_counters[(uint8_t)OpType::X].compute_total(0) << '\n';

    Sim<W, false> cached(std::mt19937_64{0});
    Circuit scan("DEBUG_PRINT\nX q0");
    cached.configure_for(scan);
    cached.debug_print_handler = [&](QubitOrBitOrBool, const W &, BitOrBool, const W &) {
        cached.condition_stack = {W{}};
    };
    cached.apply(scan);
    std::cout << "cached:" << cached.qubit_span()[0];
    cached.apply(Circuit("X q0"));
    std::cout << ':' << cached.qubit_span()[0];
    cached.apply(Circuit("POP_CONDITION\nX q0"));
    std::cout << ':' << cached.qubit_span()[0] << '\n';
}

template <typename W, bool Count>
void workload(const std::string &text, int mode) {
    Circuit circuit(text);
    Sim<W, Count> sim(std::mt19937_64{0});
    sim.configure_for(circuit);
    std::cout << "width:" << W::BIT_SIZE << '\n';
    for (int repeat = 0; repeat < 2; repeat++) {
        sim.clear_for_shot();
        if (mode == 2) {
            for (auto &qubit : sim.qubit_span()) {
                qubit.randomize(sim.rng);
            }
            sim.copy_bit_packed_state_into_register_buffer();
            std::swap(sim.register_buffers, sim.register_buffers2);
        }
        sim.apply(circuit);
        if (mode == 2) {
            sim.copy_bit_packed_state_into_register_buffer();
        }
        for (const auto &value : sim.state_block) {
            std::cout << "state:" << value << '\n';
        }
        for (size_t lane = 0; lane < sim.BATCH_SIZE; lane++) {
            std::cout << "phase:" << sim.read_shot_phase(lane) << '\n';
            for (const auto &value : sim.register_buffers[lane]) {
                std::cout << "out:" << value << '\n';
            }
            for (const auto &value : sim.register_buffers2[lane]) {
                std::cout << "before:" << value << '\n';
            }
        }
        if constexpr (Count) {
            for (size_t op = 0; op < circuit.num_ops; op++) {
                auto kind = circuit.op_types[op];
                std::cout << "count:" << kind;
                for (size_t lane = 0; lane < sim.BATCH_SIZE; lane++) {
                    std::cout << ':' << sim.new_op_counters[(uint8_t)kind].compute_total(lane);
                }
                std::cout << '\n';
            }
        }
    }
}

template <bool Count>
void all_workload_widths(const std::string &text, int mode) {
    workload<b64, Count>(text, mode);
    workload<b128, Count>(text, mode);
    workload<b256, Count>(text, mode);
    workload<b512, Count>(text, mode);
}

int main(int argc, char **argv) {
    if (argc > 1) {
        int mode = std::stoi(argv[1]);
        std::stringstream input;
        input << std::cin.rdbuf();
        if (mode == 1) {
            all_workload_widths<true>(input.str(), mode);
        } else {
            all_workload_widths<false>(input.str(), mode);
        }
        return 0;
    }
    copies<b64>();
    copies<b128>();
    copies<b256>();
    copies<b512>();
    lifecycle<b64>();
    lifecycle<b128>();
    lifecycle<b256>();
    lifecycle<b512>();
}
