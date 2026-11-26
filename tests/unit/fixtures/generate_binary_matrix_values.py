"""Observe the complete frozen binary matrix implementation without rewriting it."""

import argparse
import csv
import hashlib
import json
import random
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict

FTQC_SOURCE_INVENTORY = (
    Path(__file__).resolve().parents[3] / "docs/development/ftqc_source_inventory.csv"
)


class Case(TypedDict):
    rows: list[str]
    columns: int
    other_rows: list[str]
    other_columns: int
    vector: str
    digest: str


_ORACLE = r"""
#include "kickmix/util/gf2_matrix.h"
#include <iostream>
#include <sstream>
using namespace kickmix;
std::string hex_bytes(std::string text) {
    std::string result;
    for (unsigned char c : text) {
        result += "0123456789abcdef"[c >> 4];
        result += "0123456789abcdef"[c & 15];
    }
    return result;
}
GF2Matrix read_matrix() {
    size_t rows, cols; std::cin >> rows >> cols;
    GF2Matrix result(rows, cols);
    for (size_t row = 0; row < rows; row++) {
        std::string value; std::cin >> value;
        auto p = GF2Poly::from_str(value);
        for (size_t col = 0; col < cols; col++) result.set(row, col, p.bit(col));
    }
    return result;
}
std::string packed(const GF2Matrix &m) {
    std::string result = std::to_string(m.rows) + "," + std::to_string(m.cols) + ":";
    for (size_t row = 0; row < m.rows; row++) {
        std::string digits;
        for (size_t k = (m.cols + 3) / 4; k--;) {
            unsigned nibble = 0;
            for (size_t bit = 0; bit < 4; bit++)
                if (k*4+bit < m.cols && m.get(row,k*4+bit)) nibble |= 1 << bit;
            if (nibble || !digits.empty()) digits += "0123456789ABCDEF"[nibble];
        }
        result += (digits.empty() ? "0" : digits) + ",";
    }
    return result;
}
int main() {
    size_t count; std::cin >> count;
    for (size_t trial = 0; trial < count; trial++) {
        GF2Matrix m=read_matrix(), other=read_matrix();
        std::string vector; std::cin >> vector;
        std::cout << m.rank() << '|' << m.row_words << '|' << hex_bytes(m.str())
                  << '|' << packed(m.transposed()) << '|' << packed(m.inverse()) << '|';
        GF2Matrix p,l,u;
        if (m.plu_decompose(p,l,u)) std::cout << packed(p) << ';' << packed(l) << ';' << packed(u);
        else std::cout << "NONE";
        std::cout << '|';
        try { std::cout << packed(m * other); }
        catch (const std::invalid_argument &) { std::cout << "ERROR"; }
        std::cout << '|';
        try { std::cout << (m * GF2Poly::from_str(vector)).str(); }
        catch (const std::invalid_argument &) { std::cout << "ERROR"; }
        std::cout << '|';
        if (m.rows && m.cols) {
            GF2Matrix edit=m;
            edit.set(0,m.cols-1,true); std::cout << packed(edit) << ';';
            edit.set(0,m.cols-1,false); std::cout << packed(edit) << ';';
            edit.xor_bit(m.rows-1,0); std::cout << packed(edit) << ';';
            edit.swap_rows(0,0); std::cout << packed(edit) << ';';
            edit.swap_rows(0,m.rows-1); std::cout << packed(edit) << ';';
            edit.xor_row_into(0,m.rows-1); std::cout << packed(edit) << ';';
            edit.xor_row_into(0,0); std::cout << packed(edit) << ';';
            std::cout << edit.is_row_zero(0);
        }
        std::cout << '|' << (m == m) << '|' << (m != other) << '\n';
    }
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    inventory = FTQC_SOURCE_INVENTORY
    sources = []
    with inventory.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["id"] not in {
                "K-0306",
                "K-0307",
                "K-0308",
                "K-0309",
                "K-0310",
                "K-0298",
                "K-0299",
                "K-0322",
            }:
                continue
            data = (root.parent / row["path"]).read_bytes()
            blob = hashlib.sha1(
                b"blob " + str(len(data)).encode() + b"\0" + data
            ).hexdigest()
            assert blob == row["git_blob"]
            sources.append({"parent": row["id"], "path": row["path"], "blob": blob})
    rng = random.Random(306)
    cases: list[Case] = []

    def add(rows: list[int], columns: int, *, mismatch: bool = False) -> None:
        other_columns = min(columns + 1, 130)
        other = [rng.getrandbits(other_columns) for _ in range(columns + int(mismatch))]
        cases.append(
            Case(
                rows=[hex(row) for row in rows],
                columns=columns,
                other_rows=[hex(row) for row in other],
                other_columns=other_columns,
                vector=hex(rng.getrandbits(min(1024, columns + 5))),
                digest="",
            )
        )

    # Exhaust every square value through three coordinates, including singulars.
    for width in range(4):
        for value in range(1 << (width * width)):
            add(
                [(value >> (row * width)) & ((1 << width) - 1) for row in range(width)],
                width,
            )
    for row_count, columns in (
        (0, 5),
        (5, 0),
        (0, 1025),
        (1025, 0),
        (1, 1024),
        (1, 1025),
        (7, 13),
        (13, 7),
        (17, 100),
        (100, 17),
        (63, 65),
        (65, 63),
        (130, 130),
        (512, 512),
    ):
        add([rng.getrandbits(min(columns, 1024)) for _ in range(row_count)], columns)
    for width in (1, 2, 3, 8, 33, 64, 65, 100, 130, 512):
        rows = [1 << index for index in range(width)]
        # Invertible matrices with nontrivial pivots, independent of native rank.
        for _ in range(width * 3):
            first, second = rng.randrange(width), rng.randrange(width)
            rows[first], rows[second] = rows[second], rows[first]
            if first != second:
                rows[first] ^= rows[second]
        add(rows, width)
        add(rows, width, mismatch=True)
    with tempfile.TemporaryDirectory(prefix="fq-matrix-") as directory:
        driver, binary = Path(directory) / "oracle.cc", Path(directory) / "oracle"
        driver.write_text(_ORACLE)
        src = root / "src"
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-O1",
                "-I" + str(src),
                str(driver),
                str(src / "kickmix/util/gf2_matrix.cc"),
                str(src / "kickmix/util/gf2_poly.cc"),
                str(src / "kickmix/util/fixed_width_int.cc"),
                "-o",
                str(binary),
            ],
            check=True,
            timeout=120,
        )
        inputs = [str(len(cases))]
        for case in cases:
            inputs.extend(
                (
                    f"{len(case['rows'])} {case['columns']} " + " ".join(case["rows"]),
                    f"{len(case['other_rows'])} {case['other_columns']} "
                    + " ".join(case["other_rows"]),
                    case["vector"],
                )
            )
        result = subprocess.run(
            [str(binary)],
            input="\n".join(inputs) + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
    outputs = result.stdout.splitlines()
    assert len(outputs) == len(cases)
    for case, output in zip(cases, outputs, strict=True):
        case["digest"] = hashlib.sha256(output.encode()).hexdigest()
    args.output.write_text(
        json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n"
    )
    print(f"Recorded {len(cases)} complete unchanged-source matrix observations")


if __name__ == "__main__":
    main()
