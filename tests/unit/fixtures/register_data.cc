// SPDX-License-Identifier: Apache-2.0
// Probe unchanged locked circuit/register_data.h and scalar dependencies.
#include <iostream>
#include <sstream>
#include "kickmix/circuit/register_data.h"
using namespace kickmix;

std::string hex(const std::string &s) {
    std::string result;
    for (unsigned char value : s) {
        result += "0123456789abcdef"[value >> 4];
        result += "0123456789abcdef"[value & 15];
    }
    return result;
}

int main() {
    std::vector<RegisterData> cases;
    for (const auto &name : std::vector<std::string>{"", "word", "a\"b\\c\r\n", "\xc3\xa9", std::string("x\0y", 3)}) {
        for (const auto &members : std::vector<std::vector<QubitOrBit>>{
            {}, {QubitId(0)}, {BitId(0)}, {QubitId(1), BitId(2), QubitId(1)},
        }) {
            cases.push_back(RegisterData{name, members});
        }
    }
    for (const auto &item : cases) {
        std::stringstream text;
        text << item;
        std::cout << "value\t" << hex(item.name) << '\t';
        for (size_t k = 0; k < item.contents.size(); k++) {
            if (k) std::cout << ',';
            std::cout << item.contents[k].tagged_id;
        }
        std::cout << '\t' << hex(text.str()) << '\t';
        for (const auto &other : cases) std::cout << (item == other);
        std::cout << '\n';
    }
    RegisterData original{"independent", {QubitId(1)}};
    auto copy = original;
    copy.name = "changed";
    copy.contents[0] = BitId(2);
    std::cout << "copy\t" << original.name << '\t' << original.contents[0].tagged_id << '\t' << (original == copy) << '\n';
    RegisterData sentinel{"x", {QubitOrBit{}}};
    RegisterData bit{"x", {BitId(0)}};
    std::stringstream a, b;
    a << sentinel; b << bit;
    std::cout << "sentinel\t" << hex(a.str()) << '\t' << hex(b.str()) << '\t'
        << (sentinel == bit) << '\t' << sentinel.contents[0].tagged_id << '\t'
        << bit.contents[0].tagged_id << '\n';
}
