// SPDX-License-Identifier: Apache-2.0
// Execute unchanged locked scalar identity definitions for native conformance.
#include <iostream>
#include <sstream>
#include <vector>
#include "kickmix/id/bit_id.h"
#include "kickmix/id/qubit_id.h"
#include "kickmix/id/register_id.h"
#include "kickmix/id/xbit_id.h"
#include "kickmix/id/xbool.h"
using namespace kickmix;
template<class T> void id_row(const char *name, uint32_t index) {
    try {
        T value(index);
        T zero;
        std::ostringstream stream;
        stream << value;
        std::cout << name << '\t' << index << '\t' << value.tagged_id << '\t'
                  << value.untagged_id() << '\t' << value.str() << '\t' << stream.str()
                  << '\t' << zero.tagged_id << '\t' << (value == T(index))
                  << '\t' << (value < zero) << '\t' << (zero < value) << '\n';
    } catch(const std::invalid_argument &error) {
        std::cout << "error\t" << name << '\t' << index << '\t' << error.what() << '\n';
    }
}
int main() {
    std::cout << "limits\t" << TAG_MASK << '\t' << UNTAGGED_MASK << '\t' << MAX_NUM_QUBITS << '\t' << MAX_NUM_BITS << '\t' << MAX_NUM_REGISTERS << '\n';
    for(uint32_t n: {0u,1u,2u,17u,499999999u,500000000u,536870911u,536870912u,4294967295u}) {
        id_row<BitId>("b", n);
        id_row<QubitId>("q", n);
        id_row<XBitId>("xb", n);
        RegisterId r{n};
        std::ostringstream stream; stream << r;
        std::cout << "r\t" << n << '\t' << r.id << '\t' << r.str() << '\t' << stream.str()
                  << '\t' << (r == RegisterId{n}) << '\t' << (RegisterId{0} < r) << '\n';
        if(n <= UNTAGGED_MASK) {
            std::cout << "h\t" << n << '\t' << BitId(n).conjugated_by_h().tagged_id
                      << '\t' << XBitId(n).conjugated_by_h().tagged_id << '\n';
        }
    }
    std::cout << "kets\t" << PLUS_KET.tagged_id << '\t' << MINUS_KET.tagged_id
              << '\t' << XBool().tagged_id << '\n';
    for(bool value: {false,true}) {
        XBool b(value);std::ostringstream stream;stream << b;
        std::cout << "xbool\t" << value << '\t' << b.tagged_id << '\t' << b.str()
                  << '\t' << stream.str() << '\t' << b.is_minus_ket() << '\t' << b.is_plus_ket()
                  << '\t' << b.conjugated_by_h() << '\t' << (b == XBool(value)) << '\n';
    }
    for(unsigned tag = 0; tag < 256; tag++) {
        QXZTypeTag8 value = (QXZTypeTag8)tag;
        std::cout << "tag\t" << (int)tag << '\t' << is_not_mixed(value) << '\t' << value << '\n';
    }
}
