// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Observe the unmodified locked CircuitFuzzer through its public interface.
#include <iostream>
#include <sstream>
#include <iomanip>
#include "kickmix/sim/fuzzer.h"
using namespace kickmix;

static std::string hex(const std::string &text) {
    std::stringstream out;
    for (unsigned char c : text) out << std::hex << std::setw(2) << std::setfill('0') << (unsigned)c;
    return out.str();
}

int main() {
    for (int mode = 0; mode < 32; mode++) {
        std::stringstream trace;
        CircuitFuzzer f(std::mt19937_64{123});
        f.disarmed_warning = true;
        size_t inputs = 0, outputs = 0, builds = 0;
        auto build = [&](CircuitBuilder &b, std::mt19937_64 &rng) {
            trace << "build:" << ++builds << ":" << rng() << "\n";
            if (mode == 27) return;
            if (mode == 20) throw std::invalid_argument("builder failure");
            if (mode == 14) {
                b.append_register(5, "z"); b.append_register(5, "a"); return;
            }
            if (mode == 7) {
                b.append_register(3, "same"); b.append_register(3, "same"); return;
            }
            if (mode == 6) {b.append_register(1, "@other"); return;}
            auto q = b.append_register(mode == 19 ? 65 : 3, "target");
            if (mode == 8) b.x(QubitId{5});
            if (mode == 17) {
                b.mut.register_data.push_back(RegisterData{"mixed", {q[0], BitId{0}, q[2]}});
                b.next_register_id++;
                b.bit_store1(BitId{0});
            }
            if (mode == 18 || mode == 5) b.append_register(2, "@dirty");
            if (mode == 18) b.append_register(2, "@clean");
            if (mode == 0 || mode == 1 || mode == 12 || mode == 24 || mode == 25 || mode == 26) {
                b.ccx(q[0], q[1], q[2]); b.cx(q[0], q[1]);
                if (mode != 26) b.x(q[0]);
                if (mode == 24) b.x(q[2]);
                if (mode == 25) b.z(q[2]);
            }
            if (mode == 9 || mode == 10 || mode == 11 || mode == 31) b.z(q[0]);
            if (mode == 16 || mode == 30) b.hmr(q[0], BitId{0});
        };
        if (mode != 21) f.use_circuit_builder(build);
        if (mode == 2 || mode == 4 || mode == 5 || mode == 23 || mode == 28 || mode == 29) {
            f.use_input_and_output_sampler([&](InputSample &in, OutputSample &out, std::mt19937_64 &rng) {
                inputs++;
                if (mode != 29) in["target"] = 0;
                if (mode != 4 && (mode != 2 || inputs <= 64)) out["target"] = 0;
                trace << "both:" << inputs << ":" << rng() << "\n";
            });
        }
        if (mode == 3 || mode == 5 || mode == 15 || mode == 17 || mode == 18 || mode == 19 || mode == 22 || mode == 23) {
            f.use_input_sampler([&](InputSample &in, std::mt19937_64 &rng) {
                inputs++;
                trace << "input:" << inputs << ":" << rng() << "\n";
                if (mode == 3) return;
                in["target"] = (uint64_t)(inputs & 7);
                if (mode == 17) in["mixed"] = 0;
                if (mode == 18) {
                    trace << "special:" << in["@dirty"].bin() << ":" << in["@clean"].bin() << "\n";
                    in["@clean"] = 3;
                }
            });
        }
        if (mode == 28 || mode == 31 || mode == 24 || mode == 25 || mode == 26 || mode == 30 || mode == 0 || mode == 1 || mode == 9 || mode == 11 || mode == 12 || mode == 14 || mode == 15 || mode == 16 || mode == 17 || mode == 19 || mode == 22) {
            f.use_output_sampler([&](OutputSample &out) {
                outputs++;
                if (mode == 14) {
                    trace << "lex:" << out.old("z").bin() << ":" << out.old("a").bin() << "\n";
                    return;
                }
                trace << "old:" << outputs << ":" << out.old("target").bin() << "\n";
                if (mode == 0 || mode == 24 || mode == 25 || mode == 26) out["target"].increment();
                if (mode == 1 || mode == 12) out["target"].decrement();
                if (mode == 9) out.phase_half_turns = out["target"][0];
                if (mode == 11) out["target"].increment();
                if (mode == 16 || mode == 30) out["target"].front_ref() = false;
                if (mode == 17) {out["mixed"].bit_ref(1) = true; out["target"] = 0;}
            });
        }
        f.ignore_global_phase = mode == 10 || mode == 11;
        try {
            size_t shots = mode == 2 || mode == 15 || mode == 19 ? 129 : mode == 30 ? 64 : mode == 22 ? 0 : 3;
            f.fuzz(mode == 13 ? 0 : 1, shots);
            trace << "ok\n";
        } catch (const std::exception &ex) { trace << "error:" << ex.what() << "\n"; }
        if (mode == 12) {
            try { f.fuzz_current_circuit(1); }
            catch (const std::exception &ex) { trace << "again:" << ex.what() << "\n"; }
        }
        f.use_circuit_builder([&](CircuitBuilder &, std::mt19937_64 &rng) {
            trace << "next:" << rng() << "\n";
        });
        f.fuzz(1, 0);
        std::cout << mode << '\t' << hex(trace.str()) << '\n';
    }
    for (int mode = 0; mode < 4; mode++) {
        std::stringstream warning;
        auto *previous = std::cerr.rdbuf(warning.rdbuf());
        {
            CircuitFuzzer f(std::mt19937_64{123});
            if (mode == 1) f.disarmed_warning = true;
            if (mode == 2) f.fuzz_current_circuit(0);
            if (mode == 3) f.fuzz_current_circuit(1);
        }
        std::cerr.rdbuf(previous);
        std::cout << "warning:" << mode << '\t' << hex(warning.str()) << '\n';
    }

}
