// SPDX-License-Identifier: Apache-2.0
// Harness for unchanged frozen main.perf.cc/perf.perf.h definitions.
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <time.h>

static bool controlled_clock = false;
static uint64_t elapsed_nanos = 0;
std::chrono::steady_clock::time_point std::chrono::steady_clock::now() noexcept {
    if (controlled_clock) return time_point(duration(elapsed_nanos));
    timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return time_point(duration(uint64_t(value.tv_sec) * 1000000000 + value.tv_nsec));
}

#define main frozen_performance_main
#include "main.perf.cc"
#undef main

BENCHMARK(timing_alpha) {
    running_benchmark->results.emplace_back(0.002, 10);
    running_benchmark->results.back().show_rate("ops", uint32_t{32}).goal_micros(200);
}
BENCHMARK(timing_beta) {
    running_benchmark->results.emplace_back(0.01, 1);
    running_benchmark->results.back().show_rate("items", int16_t{3});
}
BENCHMARK(empty) {}
static BenchmarkResult custom_result(1, 1);
BENCHMARK(custom) { running_benchmark->results.push_back(custom_result); }

int main(int argc, const char **argv) {
    std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "si") {
        for (int k = 2; k < argc; k++) std::cout << si2(std::stod(argv[k])) << "\n";
        return 0;
    }
    if (mode == "adaptive") {
        controlled_clock = true;
        elapsed_nanos = 0;
        RegisteredBenchmark benchmark{"adaptive", [](){}};
        running_benchmark = &benchmark;
        BENCHMARK_CONFIG_TARGET_SECONDS = std::stod(argv[2]);
        uint64_t cost = std::stoull(argv[3]);
        uint64_t calls = 0;
        try {
            auto &result = benchmark_go([&](){calls++; elapsed_nanos += cost;});
            std::cout << calls << " " << result.total_reps << " "
                      << std::setprecision(17) << result.total_seconds << " "
                      << benchmark.results.size() << "\n";
        } catch (const std::invalid_argument &error) {
            std::cout << "error " << error.what() << "\n";
        }
        return 0;
    }
    if (mode == "report") {
        custom_result = BenchmarkResult(std::stod(argv[2]), std::stoull(argv[3]));
        custom_result.goal_seconds = std::stod(argv[4]);
        for (int k = 5; k < argc; k += 2) {
            custom_result.show_rate(argv[k], std::stod(argv[k + 1]));
        }
        const char *arguments[] = {"benchmark", "--only", "custom"};
        return frozen_performance_main(3, arguments);
    }
    if (mode == "goals") {
        BenchmarkResult initial(0.001, 3);
        auto nanos = initial;
        auto micros = initial;
        auto millis = initial;
        nanos.goal_nanos(std::stod(argv[2]));
        micros.goal_micros(std::stod(argv[2]));
        millis.goal_millis(std::stod(argv[2]));
        std::cout << std::setprecision(17) << nanos.goal_seconds << " "
                  << micros.goal_seconds << " " << millis.goal_seconds << "\n";
        return 0;
    }
    if (mode == "rates") {
        BenchmarkResult result(1, 1);
        result.show_rate("i8", int8_t{-128}).show_rate("u8", uint8_t{255})
              .show_rate("i16", int16_t{-32768}).show_rate("u16", uint16_t{65535})
              .show_rate("i32", int32_t{-2147483647 - 1}).show_rate("u32", uint32_t{4294967295})
              .show_rate("i64", int64_t{-9223372036854775807LL - 1})
              .show_rate("u64", uint64_t{18446744073709551615ULL})
              .show_rate("float", 1.23456789).show_rate("char16", char16_t{1234});
        for (auto &[name, value] : result.marginal_rates) {
            std::cout << name << " " << std::setprecision(17) << value << "\n";
        }
        return 0;
    }
    if (mode == "duplicates") {
        add_benchmark({"same", [](){
            running_benchmark->results.emplace_back(1, 1);
            running_benchmark->results.emplace_back(3, 1);
        }});
        add_benchmark({"same", [](){running_benchmark->results.emplace_back(2, 1);}});
        const char *arguments[] = {"benchmark", "--only", "same"};
        return frozen_performance_main(3, arguments);
    }
    return frozen_performance_main(argc, argv);
}
