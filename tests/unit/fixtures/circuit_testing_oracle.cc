// Observe unchanged source helpers, including their private static functions.
#include "kickmix/util/circuit_testing.test.cc"
#include "gtest/gtest-spi.h"
#include <iomanip>
#include <iostream>

void print_hex(std::string_view value) {
    std::cout << std::hex;
    for (unsigned char c : value) {
        std::cout << std::setw(2) << std::setfill('0') << unsigned(c);
    }
    std::cout << std::dec << '\n';
}

int main(int argc, char **argv) {
    if (argc > 1 && std::string(argv[1]) == "tests") {
        testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
    std::string operation, text;
    while (std::cin >> operation >> std::quoted(text)) {
        if (operation == "undent") {
            print_hex(undent(text));
        } else if (operation == "lines") {
            auto lines = split_lines(text);
            std::cout << lines.size() << '\n';
            for (auto line : lines) { print_hex(line); }
        } else if (operation == "circuit") {
            auto circuit = circuit_with_every_operation();
            print_hex(circuit.str());
            print_hex(circuit.text_diagram());
            auto file = tmpfile();
            if (file == nullptr) { return 2; }
            circuit.write_kmb_to(file);
            rewind(file);
            std::string packed;
            for (int c; (c = fgetc(file)) != EOF;) { packed.push_back(c); }
            fclose(file);
            print_hex(packed);
        } else if ((operation == "expect" || operation == "expect_classical")) {
            testing::TestPartResultArray failures;
            {
                testing::ScopedFakeTestPartResultReporter reporter(
                    testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD,
                    &failures);
                auto circuit = operation == "expect" ? Circuit("X q0") : Circuit("BIT_STORE0 b0\nBIT_INVERT b0 if b1\nPUSH_CONDITION if b0\nPOP_CONDITION");
                expect_circuit_has_text_diagram(circuit, text);
            }
            if (failures.size()) {
                print_hex(failures.GetTestPartResult(0).message());
            } else { std::cout << "ok\n"; }
        }
    }
}
