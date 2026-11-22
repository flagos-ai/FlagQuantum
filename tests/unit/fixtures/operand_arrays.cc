// SPDX-License-Identifier: Apache-2.0
// Probe unchanged locked Kickmix array and strided-view definitions.
#include <iostream>
#include <sstream>
#include <vector>
#include "kickmix/id/array_xz.h"
using namespace kickmix;

template <typename T> std::string words(const T &v) {
    std::stringstream out;
    for (size_t k = 0; k < v.size(); k++) {
        if (k) out << ',';
        out << v[k].tagged_id;
    }
    return out.str();
}

template <typename T> void emit(
    const std::string &domain, const std::string &base, int hint,
    const std::string &operation, int argument, const T &view,
    const QubitOrXZBitOrXZBool *origin) {
    stride_span_xz x(view);
    std::cout << "view\t" << domain << '\t' << base << '\t' << hint << '\t'
        << operation << '\t' << argument << '\t'
        << (x.count ? x.ptr - origin : 0) << '\t' << x.stride << '\t' << x.count
        << '\t' << (int)x.common_type << '\t' << (int)x.compute_common_type() << '\t'
        << x.is_all_qubit_ids() << x.is_all_bit_ids() << x.is_all_bools()
        << x.is_all_xbit_ids() << x.is_all_xbools() << x.is_all_xlike()
        << x.is_all_zlike() << x.is_all_qubit_or_bit()
        << x.is_all_classical_and_xlike() << x.is_all_classical_and_zlike()
        << '\t' << words(x) << '\t' << view.str() << '\t';
    try { x.checked_cast_to_qubit_ids("probe"); std::cout << '1'; } catch (const std::invalid_argument &) { std::cout << '0'; }
    try { x.checked_cast_to_bit_ids("probe"); std::cout << '1'; } catch (const std::invalid_argument &) { std::cout << '0'; }
    try { x.checked_cast_to_xbit_ids("probe"); std::cout << '1'; } catch (const std::invalid_argument &) { std::cout << '0'; }
    try { x.checked_cast_to_qubit_or_bit("probe"); std::cout << '1'; } catch (const std::invalid_argument &) { std::cout << '0'; }
    try { x.checked_cast_to_qubit_or_bit_or_bool("probe"); std::cout << '1'; } catch (const std::invalid_argument &) { std::cout << '0'; }
    try { x.checked_cast_to_qubit_or_xbit_or_xbool("probe"); std::cout << '1'; } catch (const std::invalid_argument &) { std::cout << '0'; }
    if constexpr (std::is_same_v<T, stride_span_z>) {
        std::cout << '\t' << view.is_all_classical() << view.has_any_qubits();
        auto recomputed = view;
        recomputed.recompute_common_type_data();
        std::cout << '\t' << (int)recomputed.common_type;
    } else {
        std::cout << "\t-\t-";
    }
    std::cout << '\n';
}

template <typename T> void transforms(
    const std::string &domain, const std::string &base, int hint,
    const T &span, const QubitOrXZBitOrXZBool *origin) {
    emit(domain, base, hint, "identity", 0, span, origin);
    if (!span.empty()) {
        emit(domain, base, hint, "reversed", 0, span.reversed(), origin);
        for (int n : {0, 1, 3, 9}) {
            emit(domain, base, hint, "skip", n, span.skip(n), origin);
            emit(domain, base, hint, "keep", n, span.keep(n), origin);
            emit(domain, base, hint, "skip_last", n, span.skip_last(n), origin);
            emit(domain, base, hint, "keep_last", n, span.keep_last(n), origin);
        }
    }
}

int main() {
    for (int length = 0; length <= 3; length++) {
        int combinations = 1;
        for (int k = 0; k < length; k++) combinations *= 5;
        for (int code = 0; code < combinations; code++) {
            std::vector<QubitOrXZBitOrXZBool> data;
            int remaining = code;
            for (int k = 0; k < length; k++) {
                QubitOrXZBitOrXZBool item;
                int tag = remaining % 5;
                remaining /= 5;
                item.tagged_id = (uint32_t(tag) << 29) | uint32_t(k % 2);
                data.push_back(item);
            }
            stride_span_xz raw(stride_span<const QubitOrXZBitOrXZBool>(data.data(), 1, data.size()));
            std::vector<int> hints{192};
            int computed = (int)raw.compute_common_type();
            if (computed != 192) hints.push_back(computed);
            if (raw.is_all_zlike() && computed != 128) hints.push_back(128);
            if (raw.is_all_xlike() && computed != 64) hints.push_back(64);
            for (int hint : hints) {
                stride_span_xz span(data.data(), 1, data.size(), (QXZTypeTag8)hint);
                transforms("mixed", words(span), hint, span, data.data());
                if (data.size() > 1) {
                    stride_span_xz stepped(data.data(), 2, (data.size() + 1) / 2, (QXZTypeTag8)hint);
                    transforms("mixed_step2", words(span), hint, stepped, data.data());
                }
                if (raw.is_all_zlike()) {
                    stride_span_z z((const QubitOrBitOrBool *)data.data(), 1, data.size(), (QXZTypeTag8)hint);
                    transforms("z", words(span), hint, z, data.data());
                    if (data.size() > 1) {
                        stride_span_z stepped((const QubitOrBitOrBool *)data.data(), 2, (data.size() + 1) / 2, (QXZTypeTag8)hint);
                        transforms("z_step2", words(span), hint, stepped, data.data());
                    }
                }
            }
            std::cout << "cast\t" << words(raw) << '\t';
            auto qcast = raw.cast_data<QubitId>();
            auto bcast = raw.cast_data<BitId>();
            auto xbcast = raw.cast_data<XBitId>();
            auto boolcast = raw.cast_data<bool>();
            auto xboolcast = raw.cast_data<XBool>();
            for (size_t k = 0; k < raw.size(); k++) {
                if (k) std::cout << ',';
                std::cout << qcast[k].untagged_id() << ':' << bcast[k].untagged_id()
                    << ':' << xbcast[k].untagged_id() << ':' << boolcast[k]
                    << ':' << xboolcast[k].is_minus_ket();
            }
            std::cout << '\t' << boolcast.stride << '\t' << qcast.stride << '\n';
            if (raw.is_all_xlike()) {
                stride_span_x x(raw.cast_data<QubitOrXBitOrXBool>());
                stride_span_xz from_x(x);
                std::cout << "xview\t" << words(raw) << '\t' << (int)from_x.common_type
                    << '\t' << from_x.stride << '\t' << words(from_x)
                    << '\t' << (from_x.ptr == raw.ptr) << '\n';
            }
            auto owner = array_xz::copy_of(raw.empty() ? raw : raw.reversed());
            auto concat = array_xz::copy_of_concat(raw, owner.as_ptr());
            auto saved = owner.items;
            auto moved = std::move(owner);
            bool transferred = moved.items == saved && owner.items == nullptr && owner.count == 0;
            moved = std::move(moved);
            std::cout << "owner\t" << words(raw) << '\t' << words(moved.as_ptr()) << '\t'
                << (int)moved.common_type << '\t' << (int)owner.common_type << '\t'
                << words(concat.as_ptr()) << '\t' << (int)concat.common_type << '\t'
                << transferred << '\t' << (moved.items == saved) << '\t' << moved.str() << '\n';
            if (raw.is_all_zlike()) {
                auto z = array_z::copy_of(raw.checked_cast_to_qubit_or_bit_or_bool("z"));
                auto copy = array_z::copy_of_concat(z.as_ptr().skip(1), z.as_ptr().keep(1));
                auto saved_z = z.items;
                auto from_z = array_xz(std::move(z));
                std::cout << "zowner\t" << words(raw) << '\t' << (int)from_z.common_type
                    << '\t' << (int)z.common_type << '\t' << words(copy.as_ptr()) << '\t'
                    << (int)copy.common_type << '\t'
                    << ((void *)from_z.items == (void *)saved_z && !z.items && z.empty()) << '\n';
            }
        }
    }
    auto repeat = stride_span_xz::repeat_false(7);
    std::cout << "repeat\t" << repeat.stride << '\t' << repeat.count << '\t'
        << (int)repeat.common_type << '\t' << words(repeat) << '\n';
    {
        std::vector<QubitId> ids{QubitId(3), QubitId(2), QubitId(1)};
        auto a = array_z::copy_of(ids);
        auto copied = array_z::copy_of(a);
        a[0] = QubitId(4);
        auto saved = a.items;
        auto assigned = array_z::copy_of(ids);
        assigned = std::move(a);
        bool zmove = assigned.items == saved && !a.items && a.empty();
        assigned = std::move(assigned);
        auto mixed = array_xz(std::move(assigned));
        auto mx_saved = mixed.items;
        array_xz destination;
        destination = std::move(mixed);
        std::cout << "ownership\t" << zmove << '\t' << (destination.items == mx_saved && !mixed.items && !mixed.count)
            << '\t' << (int)mixed.common_type << '\t' << (int)assigned.common_type
            << '\t' << (int)a.common_type << '\t' << copied.front().untagged_id()
            << '\t' << destination.as_ptr().front().untagged_id()
            << '\t' << (copied == array_z::copy_of(ids))
            << '\t' << (destination == array_xz::copy_of(destination.as_ptr())) << '\n';
    }
    array_xz x; array_z z; stride_span_x sx; stride_span_z sz; stride_span_xz sxz;
    std::cout << "defaults\t" << (int)x.common_type << '\t' << (int)z.common_type << '\t'
        << (int)sx.common_type << '\t' << (int)sz.common_type << '\t' << (int)sxz.common_type
        << '\t' << sx.stride << '\t' << sz.stride << '\t' << sxz.stride << '\n';
    for (int tag : {0, 1, 2, 3, 4, 64, 128, 192}) {
        auto uninitialized = array_xz::alloc_noinit(4, (QXZTypeTag8)tag);
        std::cout << "allocation\t" << tag << '\t' << uninitialized.count << '\t'
            << (int)uninitialized.common_type << '\t' << (uninitialized.items != nullptr) << '\n';
    }
}
