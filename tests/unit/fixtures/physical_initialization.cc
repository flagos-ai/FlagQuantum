// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Observe the unchanged simulator initialization grammar and value records.
#include <iomanip>
#include <iostream>
#include <sstream>
#include "kickmix/sim/sim.h"
using namespace kickmix;
static std::string hex(const std::string &text) {
    std::stringstream out;
    for (unsigned char c : text) out << std::hex << std::setw(2) << std::setfill('0') << (unsigned)c;
    return out.str();
}
int main() {
    for (const std::string text : {"", "r", "rr", "r=", "r<", "r=*", "r<0", "r<0x0", "r0=5", "r1<0x500", "r2=*", "r0002=0b0001", "r4294967295=1", "r4294967296=1", "r999999999999999999999999=1", "r0=-1", " r0=1", "r0= 5 ", "r0=* ", "r0<*", "r0=0b", "r0=0x", "r0=0_0", "r0=340282366920938463463374607431768211455", "r0=5,", "r0=5,,", ",", ",r0=1", "r0=5,r1<0x500,r2=*", "r=0,r<0b0010", "r0=0\n"}) {
        for (int many = 0; many < 2; many++) {
            std::stringstream result;
            try {
                auto items = many ? SimInitInstruction::from_str_many(text) : std::vector<SimInitInstruction>{SimInitInstruction::from_str(text)};
                result << "ok:" << items.size() << "\n";
                for (const auto &item : items) result << item.target_register.id << ':' << item.value.num_bits << ':' << item.value.hex() << ':' << item.randomize << ':' << item << '\n';
                for (const auto &left : items) for (const auto &right : items) result << (left == right);
                result << '\n';
            } catch (const std::exception &error) {result << "error:" << error.what() << '\n';}
            std::cout << many << '\t' << hex(text) << '\t' << hex(result.str()) << '\n';
        }
    }
}
