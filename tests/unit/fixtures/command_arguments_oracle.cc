// Complete scalar parser observation against unchanged original sources.
#include "kickmix/util/arg_parse.h"

#include <iomanip>
#include <fstream>
#include <iostream>
#include <limits>

using namespace kickmix;

void print_hex(const char *value) {
    std::cout << std::hex;
    for (auto p = reinterpret_cast<const unsigned char *>(value); *p; p++) {
        std::cout << std::setw(2) << std::setfill('0') << unsigned(*p);
    }
    std::cout << std::dec << '\n';
}

int main() {
    std::string op, name, def, low, high;
    size_t count;
    while (std::cin >> op >> std::quoted(name) >> std::quoted(def)
                    >> std::quoted(low) >> std::quoted(high) >> count) {
        std::vector<std::string> words(count);
        for (auto &word : words) { std::cin >> std::quoted(word); }
        std::vector<const char*> argv;
        for (auto &word : words) { argv.push_back(word.c_str()); }
        try {
            if (op == "file") {
                std::remove("argument-file");
                if (high != "absent") {
                    std::ofstream initial("argument-file", std::ios::binary);
                    initial << "abc\r\n";
                }
                auto fallback = def == "DEFAULT" ? tmpfile() : nullptr;
                try {
                    auto file = find_open_file_argument(name.c_str(), fallback, low.c_str(), count, argv.data());
                    const bool is_default = file == fallback;
                    if (high == "write") { fputs("Z", file); fflush(file); }
                    if (!is_default) { fclose(file); }
                    std::ifstream contents("argument-file", std::ios::binary);
                    std::string data((std::istreambuf_iterator<char>(contents)), {});
                    std::cout << "file " << is_default << " "; print_hex(data.c_str());
                } catch (...) {
                    if (fallback != nullptr) { fclose(fallback); }
                    throw;
                }
                if (fallback != nullptr) { fclose(fallback); }
                std::remove("argument-file");
            } else if (op == "find") {
                auto value = find_argument(name.c_str(), count, argv.data());
                if (value == nullptr) { std::cout << "missing\n"; }
                else { std::cout << "text "; print_hex(value); }
            } else if (op == "require") {
                auto value = require_find_argument(name.c_str(), count, argv.data());
                std::cout << "text "; print_hex(value);
            } else if (op == "bool") {
                auto value = find_bool_argument(name.c_str(), count, argv.data());
                std::cout << "bool " << value << '\n';
            } else if (op == "int") {
                auto value = find_int64_argument(name.c_str(), std::stoll(def), std::stoll(low), std::stoll(high), count, argv.data());
                std::cout << "int " << value << '\n';
            } else if (op == "float") {
                auto value = find_float_argument(name.c_str(), std::strtof(def.c_str(), nullptr), std::strtof(low.c_str(), nullptr), std::strtof(high.c_str(), nullptr), count, argv.data());
                std::cout << "float " << std::hexfloat << value << std::defaultfloat << '\n';
            } else if (op == "double") {
                auto value = find_double_argument(name.c_str(), std::strtod(def.c_str(), nullptr), std::strtod(low.c_str(), nullptr), std::strtod(high.c_str(), nullptr), count, argv.data());
                std::cout << "float " << std::hexfloat << value << std::defaultfloat << '\n';
            } else if (op == "enum") {
                std::map<std::string_view,int> choices{{"",10},{"test",20},{"rest",30}};
                auto value = find_enum_argument(name.c_str(), def == "NONE" ? nullptr : def.c_str(), choices, count, argv.data());
                std::cout << "int " << value << '\n';
            } else if (op == "unknown") {
                check_for_unknown_arguments({"--a","--b"},{"--old"},"program",def=="NONE"?nullptr:def.c_str(),count,argv.data());
                std::cout << "ok\n";
            } else { throw std::invalid_argument("Unknown oracle operation"); }
        } catch(const std::invalid_argument &error) {
            std::cout << "invalid "; print_hex(error.what());
        } catch(const std::out_of_range &error) {
            std::cout << "range "; print_hex(error.what());
        }
    }
}
