// SPDX-License-Identifier: Apache-2.0
// Observe unchanged locked generic memory support and its original allocations.
#include <iostream>
#include <sstream>
#include "kickmix/mem/monotonic_arena.h"
#include "kickmix/mem/util.h"
using namespace kickmix;

template <typename T> std::string items(stride_span<T> span) {
    std::stringstream out;
    for (size_t k = 0; k < span.size(); k++) {
        if (k) out << ',';
        out << +span[k];
    }
    return out.str();
}

void emit_span(const std::string &operation, int argument, stride_span<int> span,
               int *base, int offset, int stride, int count) {
    std::cout << "span\t" << offset << '\t' << stride << '\t' << count << '\t'
        << operation << '\t' << argument << '\t' << span.ptr - base << '\t'
        << span.stride << '\t' << span.count << '\t' << items(span) << '\n';
}

template <typename T, size_t alignment> void arena_case() {
    MonotonicArena<T, alignment> arena;
    auto emit = [&](const std::string &action) {
        std::stringstream values, chunks;
        bool first = true;
        arena.iter_spans([&](std::span<T> chunk) {
            if (!first) chunks << ',';
            first = false;
            chunks << chunk.size();
            for (auto value : chunk) {
                if (values.tellp() > 0) values << ',';
                values << (uint64_t)value;
            }
        });
        std::cout << "arena\t" << sizeof(T) << '\t' << alignment << '\t' << action
            << '\t' << arena.size() << '\t' << chunks.str() << '\t' << values.str() << '\n';
    };
    emit("empty");
    arena.push_back_repeat(T{2}, 0); emit("repeat_zero");
    for (int k = 0; k < 70; k++) arena.push_back(T(k));
    emit("single70");
    arena.push_back_many(std::vector<T>{7, 8, 9}); emit("many");
    T data[]{T(21), T(22), T(23), T(24), T(25)};
    arena.push_back_many(stride_span<const T>{data + 4, -2, 3}); emit("strided");
    arena.push_back_many_reversed(std::span<const T>{data, 5}); emit("reverse");
    arena.push_back_repeat(T{6}, 9); emit("repeat9");
    T *saved = arena.grab_writeable(100);
    for (int k = 0; k < 100; k++) saved[k] = T(99);
    arena.rewind_tail(saved + 3); emit("rewind3");
    auto reused = arena.grab_writeable(4);
    for (int k = 0; k < 4; k++) reused[k] = T(50 + k);
    std::cout << "reuse\t" << sizeof(T) << '\t' << alignment << '\t' << (reused == saved + 3) << '\n';
    emit("reuse4");
    MonotonicArena<T, alignment> copy;
    copy.push_back_many(arena);
    copy.push_back_many_reversed(arena);
    std::vector<T> expected;
    copy.iter_spans([&](std::span<T> chunk) { expected.insert(expected.end(), chunk.begin(), chunk.end()); });
    std::cout << "copied\t" << sizeof(T) << '\t' << alignment << '\t' << items(stride_span<const T>(expected)) << '\n';
    std::cout << "copy_chunks\t" << sizeof(T) << '\t' << alignment << '\t';
    bool first_chunk = true;
    copy.iter_spans([&](std::span<T> chunk) {
        if (!first_chunk) std::cout << ',';
        first_chunk = false;
        std::cout << chunk.size();
    });
    std::cout << '\n';
    std::vector<T> flat(arena.size());
    arena.memcpy_into(flat.data());
    std::cout << "flat\t" << sizeof(T) << '\t' << alignment << '\t' << items(stride_span<const T>(flat)) << '\n';
    auto moved = std::move(arena);
    std::cout << "move\t" << sizeof(T) << '\t' << alignment << '\t' << arena.size() << '\t' << moved.size() << '\n';
    arena = std::move(moved); arena = std::move(arena); emit("move_back");
    arena.clear(); emit("clear");
    arena.push_back(T{3}); emit("reuse_after_clear");
    bool refused = false;
    try { arena.rewind_tail(data); } catch (const std::invalid_argument &) { refused = true; }
    std::cout << "rewind_refusal\t" << sizeof(T) << '\t' << alignment << '\t' << refused << '\n';
}

int main() {
    int data[40];
    for (int k = 0; k < 40; k++) data[k] = k;
    for (int stride : {-2, -1, 0, 1, 2}) {
        for (int displacement : {-3, 0, 1, 3}) {
            stride_ptr<int> p{data + 12, stride};
            auto shifted = p + displacement;
            auto old = p++;
            auto &next = ++p;
            auto same_address = stride_ptr<int>{shifted.ptr, stride + 1};
            std::cout << "pointer\t" << stride << '\t' << displacement << '\t'
                << *shifted << '\t' << *old << '\t' << *next << '\t'
                << (shifted == same_address) << '\t' << *(displacement + old) << '\t';
            p -= 2; p += displacement;
            std::cout << *p << '\t' << *(p - displacement) << '\n';
        }
        for (int count : {0, 1, 4}) {
            stride_span<int> span{data + 12, stride, size_t(count)};
            emit_span("identity", 0, span, data, 12, stride, count);
            emit_span("reversed", 0, span.reversed(), data, 12, stride, count);
            for (int n : {0, 1, 3, 4, 1000}) {
                emit_span("skip", n, span.skip(n), data, 12, stride, count);
                emit_span("keep", n, span.keep(n), data, 12, stride, count);
                emit_span("skip_last", n, span.skip_last(n), data, 12, stride, count);
                emit_span("keep_last", n, span.keep_last(n), data, 12, stride, count);
                emit_span("subspan", n, span.subspan(n, 2), data, 12, stride, count);
            }
            auto indexed = span[iota{2, true}];
            std::cout << "iota\t" << stride << '\t' << count << '\t' << indexed.ptr - data << '\t' << indexed.stride << '\t' << *indexed << '\n';
            bool low = false, high = false;
            try { span.at(size_t(-1)); } catch (const std::out_of_range &) { low = true; }
            try { span.at(count); } catch (const std::out_of_range &) { high = true; }
            std::cout << "bounds\t" << stride << '\t' << count << '\t' << low << high << '\n';
        }
    }
    stride_ptr<int> zero{data + 3, 0}; zero[-999] = -1;
    std::cout << "mutate\t" << zero[1000] << '\t' << data[3] << '\n';
    uint32_t words[]{0x01020304u, 0x05060708u, 0x090a0b0cu};
    stride_span<uint32_t> raw{words, 1, 3};
    auto bytes = raw.cast_data<uint8_t>();
    std::cout << "cast\t" << bytes.stride << '\t' << bytes.count << '\t' << items(bytes) << '\n';
    for (size_t alignment : {1, 2, 4, 8, 16, 32, 64}) {
        MonotonicArena_Helper helper;
        for (size_t length : {0, 1, 13, 1000, 3, 2000}) {
            auto pointer = helper.grab_writeable(length, alignment);
            for (size_t k = 0; k < length; k++) pointer[k] = char(k % 100);
            std::cout << "helper\t" << alignment << '\t' << length << '\t'
                << helper.cur_length_used << '\t' << helper.cur_length_cap << '\t'
                << helper.total_frozen_length_used << '\t' << helper.frozen_allocations.size()
                << '\t' << (uintptr_t(pointer) % alignment == 0) << '\n';
        }
        auto copy = std::move(helper);
        helper = std::move(copy); helper = std::move(helper);
        helper.clear();
        std::cout << "helper_clear\t" << alignment << '\t' << helper.cur_length_cap
            << '\t' << helper.cur_length_used << '\t' << helper.frozen_allocations.size() << '\n';
    }
    arena_case<uint8_t, 32>(); arena_case<uint16_t, 4>();
    arena_case<uint32_t, 4>(); arena_case<uint64_t, 32>();
    for (int count : {0, 1, 3, 10, 100}) {
        std::vector<uint32_t> v;
        for (int k = 0; k < count; k++) v.push_back(k * k % 255);
        std::cout << "maximum\t" << count << '\t' << max_u32(v.data(), v.size()) << '\n';
    }
    for (int length : {0, 1, 3, 10}) for (int count : {0, 1, 2, 3, 4, 9, 16}) {
        std::string source;
        for (int k = 0; k < length; k++) source += char('a' + k);
        std::string destination(length * count, '?');
        memcpy_repeat(destination.data(), source.data(), length, count);
        std::cout << "repeat\t" << length << '\t' << count << '\t' << destination << '\n';
    }
    bool overflow = false;
    try { auto p = aligned_alloc_else_throw<uint32_t, 32>(SIZE_MAX); free_aligned(p); }
    catch (const std::runtime_error &) { overflow = true; }
    std::cout << "allocation_overflow\t" << overflow << '\n';
}
