"""Record unchanged C++ polynomial values through a small executable oracle."""

import argparse
import hashlib
import json
import random
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict


class Case(TypedDict):
    operation: str
    left: str
    right: str
    amount: int
    expected: str


_ORACLE = r"""
#include "kickmix/util/gf2_poly.h"
#include <iostream>
#include <sstream>
using namespace kickmix;
std::string decode(std::string text) {
    std::string out;
    for (size_t k = 0; k < text.size(); k += 2)
        out += char(std::stoul(text.substr(k, 2), nullptr, 16));
    return out;
}
int main() {
    std::string op, a, b; size_t n;
    while (std::cin >> op >> a >> b >> n) {
        try {
            auto x = GF2Poly::from_str(decode(a));
            auto y = GF2Poly::from_str(decode(b));
            if (op == "mul") x = GF2Poly::mul(x, y);
            else if (op == "mod") x.imod(y);
            else if (op == "gcd") x = GF2Poly::gcd(x, y);
            else if (op == "inv") x = GF2Poly::invert_mod(x, y);
            else if (op == "left") x <<= n;
            else if (op == "right") x >>= n;
            else if (op == "xor") x ^= y;
            else if (op == "xorshift") x.ixor_shifted(y, n);
            else if (op == "self") x.ixor_shifted(x, n);
            else if (op == "truncate") x.truncate(n);
            else if (op == "fixed") x = GF2Poly::from_fixed_width_int(x.to_fixed_width_int(n));
            else if (op == "set") x.set_bit(n, true);
            else if (op == "clear") x.set_bit(n, false);
            else if (op == "toggle") x.xor_bit(n);
            else if (op == "monomial") x = GF2Poly::monomial(n);
            if (op == "algebraic") { std::cout << x.algebraic_str(n) << '\n'; continue; }
            if (op == "bit") { std::cout << x.bit(n) << '\n'; continue; }
            if (op == "meta") {
                std::cout << x.is_zero() << ',' << x.degree() << ','
                          << x.num_bits_in_use() << ',' << x.weight() << '\n';
                continue;
            }
            std::cout << x.str() << '\n';
        } catch (const std::invalid_argument &) { std::cout << "ERROR\n"; }
    }
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    intake = json.loads(
        (root.parent / "g2-gf-binding-intake-20261122.json").read_text()
    )
    sources = []
    for item in intake["sources"]:
        if item["parent"] not in {"K-0309", "K-0310", "K-0311"}:
            continue
        data = (root.parent / item["path"]).read_bytes()
        digest = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        assert digest == item["blob"]
        sources.append(item)
    cases: list[Case] = []

    def add(operation: str, left: str, right: str = "0", amount: int = 0) -> None:
        cases.append(
            Case(
                operation=operation, left=left, right=right, amount=amount, expected=""
            )
        )

    strings = [
        "",
        "0",
        "1",
        "x",
        "X",
        "x^4 + x + 1",
        "x**4 + 0 + 1 + 0",
        "1+x^0",
        "x+x^1",
        "x^",
        "x^-1",
        "x^1024",
        "x^1023",
        "x^0000000000004",
        "x** 2 + X",
        "x ^ 2",
        "x^２",
        "\v1",
        "\t x^7 + 1\r\n",
        "0x",
        "0x___",
        "0B_10_01_",
        "0X_f_f",
        "0x0" + "0" * 256,
        "0b" + "0" * 1025,
        "2*x",
        "+x",
        "x+",
        "0+0",
        "0b2",
        "0xg",
        "x\t**\r3",
        "0x1+1",
    ]
    for text in strings:
        add("parse", text)
    rng = random.Random(9175)
    for trial in range(300):
        left = hex(rng.getrandbits(rng.randrange(1025)))
        right = hex(rng.getrandbits(rng.randrange(513)))
        for operation in ("mul", "mod", "gcd", "inv", "xor"):
            add(operation, left, right)
    for value in (0, 1, 0x11B, (1 << 1024) - 1, (1 << 1023) | (1 << 511) | 1):
        left = hex(value)
        for operation in ("meta", "parse"):
            add(operation, left)
        for amount in (0, 1, 5, 6, 7, 13, 63, 64, 65, 511, 512, 1023, 1024, 1025):
            for operation in (
                "left",
                "right",
                "xorshift",
                "self",
                "truncate",
                "fixed",
                "set",
                "clear",
                "toggle",
                "monomial",
                "bit",
                "algebraic",
            ):
                add(operation, left, "0xDEADBEEFCAFEBABE", amount)
    for value in range(256):
        add("inv", hex(value), "0x11b")
    with tempfile.TemporaryDirectory(prefix="fq-polynomial-") as directory:
        driver = Path(directory) / "oracle.cc"
        binary = Path(directory) / "oracle"
        driver.write_text(_ORACLE)
        src = root / "src"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O1",
                "-I" + str(src),
                str(driver),
                str(src / "kickmix/util/gf2_poly.cc"),
                str(src / "kickmix/util/fixed_width_int.cc"),
                "-o",
                str(binary),
            ],
            check=True,
            timeout=120,
        )
        # Prefix an ASCII space so even an empty source string has a wire token.
        commands = "".join(
            f"{c['operation']} {(' ' + c['left']).encode().hex()} {c['right'].encode().hex()} {c['amount']}\n"
            for c in cases
        )
        result = subprocess.run(
            [str(binary)],
            input=commands,
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
    outputs = result.stdout.splitlines()
    assert len(outputs) == len(cases)
    for case, expected in zip(cases, outputs, strict=True):
        case["expected"] = expected
    args.output.write_text(
        json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n"
    )
    print(f"Recorded {len(cases)} unchanged-source polynomial observations")


if __name__ == "__main__":
    main()
