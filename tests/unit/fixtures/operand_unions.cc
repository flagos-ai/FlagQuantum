// SPDX-License-Identifier: Apache-2.0
// Exercise unchanged locked operand union definitions through their actual C++ methods.
#include <iostream>
#include <type_traits>
#include <sstream>
#include <vector>
#include "kickmix/id/bit_or_bool.h"
#include "kickmix/id/bit_or_false.h"
#include "kickmix/id/qubit_or_bit.h"
#include "kickmix/id/qubit_or_bit_or_bool.h"
#include "kickmix/id/qubit_or_false.h"
#include "kickmix/id/qubit_or_minus_state.h"
#include "kickmix/id/qubit_or_true.h"
#include "kickmix/id/qubit_or_xbit_or_xbool.h"
#include "kickmix/id/qubit_or_xzbit_or_xzbool.h"
using namespace kickmix;

template <class T> std::string streamed(const T &value) {
    std::ostringstream out;
    out << value;
    return out.str();
}
void zrow(const QubitOrBitOrBool &value) {
    std::cout << "z\t" << value.tagged_id << '\t' << value.untagged_id()
              << '\t' << (int)value.tag() << '\t' << value.str() << '\t' << streamed(value)
              << '\t' << value.is_bool() << '\t' << value.is_false() << '\t' << value.is_true()
              << '\t' << value.is_bit() << '\t' << value.is_qubit() << '\t' << (bool)value
              << '\t' << ((BitId)value).tagged_id << '\t' << ((QubitId)value).tagged_id
              << '\t' << (value == QubitOrBitOrBool(value)) << '\n';
}
void xrow(const QubitOrXBitOrXBool &value) {
    std::cout << "x\t" << value.tagged_id << '\t' << value.untagged_id()
              << '\t' << (int)value.tag() << '\t' << value.str() << '\t' << streamed(value)
              << '\t' << value.is_xbool() << '\t' << value.is_xbit() << '\t' << value.is_qubit()
              << '\t' << value.is_plus_ket() << '\t' << value.is_minus_ket()
              << '\t' << ((XBitId)value).tagged_id << '\t' << ((QubitId)value).tagged_id
              << '\t' << (value == QubitOrXBitOrXBool(value)) << '\n';
}
void mixedrow(const QubitOrXZBitOrXZBool &value) {
    std::cout << "mixed\t" << value.tagged_id << '\t' << value.untagged_id()
              << '\t' << (int)value.tag() << '\t' << value.str() << '\t' << streamed(value)
              << '\t' << value.is_qubit_or_bit_or_bool() << '\t' << value.is_qubit_or_xbit_or_xbool()
              << '\t' << value.is_bool() << '\t' << value.is_bit() << '\t' << value.is_qubit()
              << '\t' << value.is_xbool() << '\t' << value.is_xbit() << '\t' << value.is_true()
              << '\t' << value.is_false() << '\t' << value.is_plus_ket() << '\t' << value.is_minus_ket()
              << '\t' << (bool)value << '\t' << ((XBool)value).tagged_id
              << '\t' << ((BitId)value).tagged_id << '\t' << ((QubitId)value).tagged_id
              << '\t' << ((XBitId)value).tagged_id
              << '\t' << ((QubitOrBitOrBool)value).tagged_id
              << '\t' << ((QubitOrXBitOrXBool)value).tagged_id
              << '\t' << (value == QubitOrXZBitOrXZBool(value)) << '\n';
}
template <class Union> void constructors(const char *name) {
    std::cout << "constructible\t" << name
              << '\t' << std::is_constructible_v<Union, bool>
              << '\t' << std::is_constructible_v<Union, BitId>
              << '\t' << std::is_constructible_v<Union, QubitId>
              << '\t' << std::is_constructible_v<Union, XBool>
              << '\t' << std::is_constructible_v<Union, XBitId>
              << '\t' << std::is_constructible_v<Union, uint32_t>
              << '\t' << std::is_default_constructible_v<Union> << '\n';
}
int main() {
    constructors<BitOrBool>("classical_boolean");
    constructors<BitIdOrFalse>("classical_false");
    constructors<QubitOrBit>("quantum_classical");
    constructors<QubitOrBitOrBool>("z_operand");
    constructors<QubitOrFalse>("quantum_false");
    constructors<QubitOrMinusState>("quantum_minus");
    constructors<QubitOrTrue>("quantum_true");
    constructors<QubitOrXBitOrXBool>("x_operand");
    constructors<QubitOrXZBitOrXZBool>("mixed_operand");
    for (uint32_t n : {0u, 1u, 17u, 500000000u, UNTAGGED_MASK}) {
        BitId b(n); QubitId q(n); XBitId xb(n);
        BitOrBool bb(b);
        std::cout << "bitbool\t" << bb.tagged_id << '\t' << bb.untagged_id()
                  << '\t' << bb.str() << '\t' << streamed(bb) << '\t' << bb.is_bool()
                  << '\t' << bb.is_bit() << '\t' << (bool)bb << '\t' << ((BitId)bb).tagged_id
                  << '\t' << (bb == b) << '\t' << (bb == BitOrBool(b)) << '\t' << (bb == false) << '\n';
        BitIdOrFalse bf(b); QubitOrFalse qf(q); QubitOrBit qb(q);
        std::cout << "restricted\t" << n << '\t' << bf.tagged_id << '\t' << bf.is_bit()
                  << '\t' << bf.bit().tagged_id << '\t' << bf.untagged_id() << '\t' << streamed(bf)
                  << '\t' << (bf == BitIdOrFalse(b))
                  << '\t' << qf.tagged_id << '\t' << qf.is_qubit() << '\t' << qf.qubit().tagged_id
                  << '\t' << qf.untagged_id() << '\t' << streamed(qf) << '\t' << (qf == QubitOrFalse(q)) << '\n';
        for (QubitOrBit v : {QubitOrBit{}, QubitOrBit(b), qb}) {
            std::cout << "qubitbit\t" << v.tagged_id << '\t' << v.untagged_id()
                      << '\t' << v.is_bit() << '\t' << v.is_qubit() << '\t' << streamed(v)
                      << '\t' << ((BitId)v).tagged_id << '\t' << ((QubitId)v).tagged_id
                      << '\t' << (v == QubitOrBit(v)) << '\n';
        }
        QubitOrTrue qt(q); QubitOrMinusState qm(q);
        std::cout << "qubitconstant\t" << n << '\t' << qt.tagged_id << '\t' << qt.is_true()
                  << '\t' << qt.is_qubit() << '\t' << qt.untagged_id() << '\t' << ((QubitId)qt).tagged_id
                  << '\t' << qm.tagged_id << '\t' << qm.is_minus_ket() << '\t' << qm.is_qubit()
                  << '\t' << qm.untagged_id() << '\t' << ((QubitId)qm).tagged_id << '\t' << qm.str()
                  << '\t' << (qm == QubitOrMinusState(q)) << '\n';
        for (QubitOrBitOrBool z : {QubitOrBitOrBool(b), QubitOrBitOrBool(q), QubitOrBitOrBool(qt), QubitOrBitOrBool(qb), QubitOrBitOrBool(bb)}) zrow(z);
        for (QubitOrXBitOrXBool x : {QubitOrXBitOrXBool(q), QubitOrXBitOrXBool(xb), QubitOrXBitOrXBool(qm)}) xrow(x);
        for (QubitOrXZBitOrXZBool v : {QubitOrXZBitOrXZBool(b), QubitOrXZBitOrXZBool(q), QubitOrXZBitOrXZBool(xb), QubitOrXZBitOrXZBool(qm), QubitOrXZBitOrXZBool(QubitOrBitOrBool(b)), QubitOrXZBitOrXZBool(QubitOrXBitOrXBool(xb))}) mixedrow(v);
    }
    for (bool value : {false, true}) {
        BitOrBool bb(value);
        std::cout << "bitbool\t" << bb.tagged_id << '\t' << bb.untagged_id() << '\t' << bb.str()
                  << '\t' << streamed(bb) << '\t' << bb.is_bool() << '\t' << bb.is_bit()
                  << '\t' << (bool)bb << '\t' << ((BitId)bb).tagged_id << '\t' << (bb == BitId(0))
                  << '\t' << (bb == BitOrBool(value)) << '\t' << (bb == value) << '\n';
        zrow(QubitOrBitOrBool(value)); xrow(QubitOrXBitOrXBool(XBool(value)));
        mixedrow(QubitOrXZBitOrXZBool(value)); mixedrow(QubitOrXZBitOrXZBool(XBool(value)));
    }
    std::cout << "numericfalse\t" << BitIdOrFalse(uint32_t{0}).tagged_id << '\n';
    BitIdOrFalse bf; QubitOrFalse qf;
    constexpr QubitOrTrue qt(true); constexpr QubitOrMinusState qm(MINUS_KET);
    std::cout << "defaults\t" << BitOrBool().tagged_id << '\t' << bf.tagged_id << '\t' << bf.is_bit()
              << '\t' << bf.bit().tagged_id << '\t' << streamed(bf) << '\t' << BitIdOrFalse(false).tagged_id
              << '\t' << qf.tagged_id << '\t' << qf.is_qubit() << '\t' << qf.qubit().tagged_id << '\t' << streamed(qf)
              << '\t' << qt.tagged_id << '\t' << qt.is_true() << '\t' << qm.tagged_id << '\t' << qm.is_minus_ket()
              << '\t' << qm.str() << '\t' << QubitOrBitOrBool().tagged_id << '\t' << QubitOrXBitOrXBool().tagged_id
              << '\t' << QubitOrXZBitOrXZBool().tagged_id << '\n';
    std::vector<QubitOrBitOrBool> zs = {false,true,BitId(0),BitId(17),QubitId(0),QubitId(17)};
    for (auto a : zs) {
        std::cout << "zequals\t" << a.tagged_id << '\t' << (a == false) << '\t' << (a == true)
                  << '\t' << (a == BitId(0)) << '\t' << (a == BitId(17)) << '\t' << (a == QubitId(0))
                  << '\t' << (a == QubitId(17)) << '\t' << (a == BitOrBool(false)) << '\t' << (a == BitOrBool(true))
                  << '\t' << (a == BitOrBool(BitId(0))) << '\t' << (a == BitOrBool(BitId(17))) << '\n';
        for (auto b : zs) std::cout << "zorder\t" << a.tagged_id << '\t' << b.tagged_id << '\t' << (a == b) << '\t' << (a < b) << '\n';
    }
    for (auto value : {QubitOrXBitOrXBool(QubitId(0)), QubitOrXBitOrXBool(XBitId(0)), QubitOrXBitOrXBool(MINUS_KET)}) {
        std::cout << "xequals\t" << value.tagged_id << '\t' << (value == QubitId(0)) << '\n';
    }
    std::vector<QubitOrXZBitOrXZBool> all = {false,true,BitId(0),BitId(17),QubitId(0),QubitId(17),XBool(false),XBool(true),XBitId(0),XBitId(17)};
    for (const auto &a : all) for (const auto &b : all) {
        std::cout << "order\t" << a.tagged_id << '\t' << b.tagged_id << '\t' << (a == b) << '\t' << (a < b) << '\n';
    }
    for (uint32_t word : {2u, 0x60000002u, 0xA0000000u, 0xA0000011u, 0xFFFFFFFFu}) {
        BitOrBool bb; bb.tagged_id = word;
        BitIdOrFalse bf; bf.tagged_id = word;
        QubitOrFalse qf; qf.tagged_id = word;
        QubitOrMinusState qm(MINUS_KET); qm.tagged_id = word;
        QubitOrBit qb; qb.tagged_id = word;
        QubitOrTrue qt(true); qt.tagged_id = word;
        std::cout << "rawrestricted\t" << word << '\t' << bb.str() << '\t' << streamed(bf)
                  << '\t' << streamed(qf) << '\t' << qm.str() << '\t' << streamed(qb)
                  << '\t' << qt.is_true() << '\t' << qt.is_qubit() << '\t' << qt.untagged_id()
                  << '\t' << ((QubitId)qt).tagged_id << '\n';
        QubitOrXZBitOrXZBool v;v.tagged_id=word;mixedrow(v);
        QubitOrBitOrBool z;z.tagged_id=word;zrow(z);
        QubitOrXBitOrXBool x;x.tagged_id=word;xrow(x);
    }
}
