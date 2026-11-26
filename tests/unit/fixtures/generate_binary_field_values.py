"""Record complete frozen binary field utilities and construction outcomes."""

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
    request: list[str]
    digest: str


_ORACLE = r"""
#include "kickmix/util/gf2_field.h"
#include <iostream>
#include <memory>
using namespace kickmix;
std::string decode(std::string text) {
    std::string out;
    for(size_t k=0;k<text.size();k+=2) out += char(std::stoul(text.substr(k,2),nullptr,16));
    return out;
}
std::string packed(const GF2Matrix &m) {
    std::string result=std::to_string(m.rows)+","+std::to_string(m.cols)+":";
    for(size_t row=0;row<m.rows;row++) {
        std::string digits;
        for(size_t k=(m.cols+3)/4;k--;) {
            unsigned v=0;
            for(size_t b=0;b<4;b++) if(k*4+b<m.cols && m.get(row,k*4+b)) v |= 1<<b;
            if(v || !digits.empty()) digits += "0123456789ABCDEF"[v];
        }
        result += (digits.empty()?"0":digits)+",";
    }
    return result;
}
int main() {
    std::string kind;
    while(std::cin>>kind) {
        if(kind=="factors" || kind=="default") {
            size_t degree;std::cin>>degree;
            try {
                if(kind=="default") std::cout << gf2_default_irreducible_poly(degree).str();
                else for(auto p:gf2_mersenne_prime_factors(degree))
                    std::cout << GF2Poly::from_fixed_width_int(p).str() << ',';
            } catch(const std::invalid_argument &) {std::cout<<"ERROR";}
            std::cout<<'\n';continue;
        }
        if(kind=="irreducible") {
            std::string p;std::cin>>p;
            try { std::cout<<gf2_is_irreducible(GF2Poly::from_str(decode(p))); }
            catch(const std::invalid_argument &) {std::cout<<"ERROR";}
            std::cout<<'\n';continue;
        }
        int mode;size_t degree,count;
        std::string p,g,a,b,e;
        std::cin>>mode>>degree>>p>>g>>a>>b>>e>>count;
        try {
            std::unique_ptr<GF2Field> f;
            auto primitive=g=="-"?std::optional<GF2Poly>{}:GF2Poly::from_str(decode(g));
            if(mode==0) f=std::make_unique<GF2Field>(degree);
            else if(mode==1) f=std::make_unique<GF2Field>(degree,GF2Poly::from_str(decode(p)),primitive);
            else if(mode==2) f=std::make_unique<GF2Field>(GF2Poly::from_str(decode(p)),primitive);
            else f=std::make_unique<GF2Field>(degree,decode(p));
            auto left=GF2Poly::from_str(a),right=GF2Poly::from_str(b);
            std::cout<<f->degree()<<'|'<<f->modulus()<<'|'<<f->primitive_element()
                     <<'|'<<f->zero()<<'|'<<f->one()<<'|'<<f->x()
                     <<'|'<<f->is_element(left)<<'|'<<f->is_primitive_element(left)
                     <<'|'<<f->mod(left)<<'|'<<f->add(left,right)
                     <<'|'<<f->mul(left,right)<<'|'<<f->square(left)
                     <<'|'<<f->frobenius(left,count)<<'|';
            try {std::cout<<f->pow(left,FixedWidthInt::from_str(e));}
            catch(const std::invalid_argument &) {std::cout<<"ERROR";}
            std::cout<<'|'<<f->invert(left)<<'|'<<f->div(left,right)<<'|';
            auto product=GF2Poly::mul(f->mod(left),f->mod(right));
            f->ireduce(product);std::cout<<product<<'|'<<packed(f->frobenius_matrix(count))<<'|';
            try {std::cout<<packed(f->constant_mul_matrix(right));}
            catch(const std::invalid_argument &) {std::cout<<"ERROR";}
            std::cout<<'|'<<packed(f->linear_map_matrix([&](auto v){return f->add(f->square(v),v);}))
                     <<'|'<<(*f==*f)<<'|'<<(*f!=GF2Field(1))<<'\n';
        } catch(const std::invalid_argument &) {std::cout<<"ERROR\n";}
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
                "K-0302",
                "K-0303",
                "K-0304",
                "K-0305",
                "K-0306",
                "K-0307",
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
    cases: list[Case] = []

    def add(*request: str) -> None:
        cases.append(Case(request=list(request), digest=""))

    for degree in range(514):
        add("factors", str(degree))
    for degree in (
        *range(66),
        100,
        127,
        128,
        163,
        233,
        255,
        256,
        283,
        300,
        384,
        409,
        504,
        511,
        512,
        513,
    ):
        add("default", str(degree))
    for polynomial in (*range(1024), 1 << 513, (1 << 513) | 1, (1 << 1023) | 1):
        add("irreducible", hex(polynomial).encode().hex())
    rng = random.Random(302)

    def field_case(
        mode: int,
        degree: int,
        modulus: str = "0",
        primitive: str = "-",
        left: int = 0,
        right: int = 0,
        exponent: int = 0,
        count: int = 1,
    ) -> None:
        add(
            "field",
            str(mode),
            str(degree),
            modulus.encode().hex(),
            primitive.encode().hex() if primitive != "-" else "-",
            hex(left),
            hex(right),
            hex(exponent),
            str(count),
        )

    for degree in range(1, 5):
        for left in range(1 << degree):
            for right in range(1 << degree):
                field_case(
                    0,
                    degree,
                    left=left,
                    right=right,
                    exponent=left + right,
                    count=degree + 1,
                )
    for degree in (5, 8, 12, 17, 32, 64, 65, 100, 128, 163, 256, 504, 512):
        for _ in range(3):
            field_case(
                0,
                degree,
                left=rng.getrandbits(min(1024, degree + 3)),
                right=rng.getrandbits(min(1024, degree + 5)),
                exponent=rng.getrandbits(160),
                count=degree + 3,
            )
    for polynomial in range(1, 64):
        field_case(2, 0, hex(polynomial), left=0x57, right=0x83, exponent=7, count=9)
    for modulus in (
        "0x11b",
        "0x1b",
        "x^8 + x^4 + x^3 + x + 1",
        "x^8+x^8+1",
        "0",
        "0x101",
        "0x25",
        "x",
        "0x11d",
    ):
        for mode in (1, 2, 3):
            field_case(mode, 8, modulus, left=0x57, right=0x83, exponent=19, count=10)
    for primitive in range(257):
        field_case(
            1, 8, "0x11b", hex(primitive), left=primitive, right=0, exponent=0, count=0
        )
    for mode in (0, 1, 3):
        for degree in (0, 513):
            field_case(mode, degree)
    # A known dense irreducible modulus exercises the dense reduction branch.
    field_case(
        1, 32, "0x1f2c32b6d", left=0xDEADBEEF, right=0xFFFFFFFF, exponent=97, count=3
    )
    with tempfile.TemporaryDirectory(prefix="fq-field-") as directory:
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
                str(src / "kickmix/util/gf2_field.cc"),
                str(src / "kickmix/util/gf2_matrix.cc"),
                str(src / "kickmix/util/gf2_poly.cc"),
                str(src / "kickmix/util/fixed_width_int.cc"),
                "-o",
                str(binary),
            ],
            check=True,
            timeout=120,
        )
        result = subprocess.run(
            [str(binary)],
            input="\n".join(" ".join(case["request"]) for case in cases) + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=180,
        )
    outputs = result.stdout.splitlines()
    assert len(outputs) == len(cases)
    for case, output in zip(cases, outputs, strict=True):
        case["digest"] = hashlib.sha256(output.encode()).hexdigest()
    args.output.write_text(
        json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n"
    )
    print(f"Recorded {len(cases)} complete unchanged-source field observations")


if __name__ == "__main__":
    main()
