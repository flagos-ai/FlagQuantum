// SPDX-License-Identifier: Apache-2.0
// Test harness for unchanged frozen Kickmix test.test.cc/.h.
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "test.test.cc"

static void emit_hex(std::string_view value) {
    for (unsigned char byte : value) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (unsigned)byte;
    }
    std::cout << std::dec << "\n";
}

int main(int argc, char **argv) {
    std::string mode = argv[1];
    if (mode == "seed") {
        std::vector<uint64_t> values;
        for (int k = 2; k < argc; k++) values.push_back(std::stoull(argv[k]));
        std::seed_seq seq(values.begin(), values.end());
        std::mt19937_64 rng(seq);
        for (size_t k = 0; k < 630; k++) std::cout << rng() << "\n";
    } else if (mode == "independent") {
        shared_test_rng = std::mt19937_64(std::stoull(argv[2]));
        shared_test_rng_initialized = true;
        for (size_t k = 0; k < 3; k++) {
            auto rng = INDEPENDENT_TEST_RNG();
            for (size_t j = 0; j < 630; j++) std::cout << rng() << "\n";
        }
        std::cout << shared_test_rng() << "\n";
    } else if (mode == "external") {
        auto rng = externally_seeded_rng();
        std::cout << rng.state_size << " " << rng.word_size << "\n";
    } else if (mode == "lookup") {
        try {
            emit_hex(resolve_testdata_file_path(argv[2]));
        } catch (const std::invalid_argument &error) {
            std::cout << "error\n";
            emit_hex(error.what());
        }
    } else if (mode == "files") {
        std::string contents;
        for (unsigned k = 0; k < 256; k++) contents.push_back(k);
        int descriptor;
        std::string path;
        {
            RaiiTempNamedFile file(contents);
            descriptor = file.descriptor;
            path = file.path;
            emit_hex(file.read_contents());
            auto bytes = file.read_bytes();
            emit_hex(std::string(bytes.begin(), bytes.end()));
            file.write_contents("replacement");
            emit_hex(file.read_contents());
            FILE *stream = fopen(file.path.c_str(), "rb");
            getc(stream);
            emit_hex(rewind_read_close(stream));
        }
        std::cout << std::filesystem::exists(path) << " "
                  << (fcntl(descriptor, F_GETFD) != -1) << "\n";
        close(descriptor);
        {
            RaiiTempNamedFile file;
            emit_hex(file.read_contents());
            close(file.descriptor);
        }
    } else {
        return 2;
    }
}
