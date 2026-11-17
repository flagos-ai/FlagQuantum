# Binary-field monomial scale file audit

Kickmix reference commit `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

| Ledger | Source | Frozen blob |
| --- | --- | --- |
| K-0089 | `gen_gf_imul_x.cc` | `efc026a507d520c924034adfa2c42569211cfa8d` |
| K-0090 | `gen_gf_imul_x.h` | `13d37b6a36706a11e59358bf796fe80899f67fec` |
| K-0091 | `gen_gf_imul_x.perf.cc` | `06ac25932ec2687d4b173daa1e08e07e84ae5573` |
| K-0092 | `gen_gf_imul_x.test.cc` | `ec9600d1226c023c27a633cb7357b3cb6c60cd72` |

The owner is `compiler/registers/finite_field/binary_linear_map.py`. The
target is a big-endian polynomial word whose width equals the field degree.
Positive exponents multiply by powers of the monomial. Negative exponents
select the source division direction. A zero exponent or degree one exits
before any call mark. The operation works in place without scratch.

The stepped path accumulates the reduction CX gates and emits the deferred
cyclic rotation in source order. It is used through the integer threshold
`degree² / (non-leading modulus term count)`. Strictly larger exponents
use the standalone linear-map call mark, including an empty map when the
field power reduces to the identity. The standalone physical binding and
Karatsuba stages share the same selection and call boundaries.

All nine original tests passed. The fixture verifies 74 complete source
streams including the threshold on either side, both signed directions,
the unsigned 64-bit exponent limit, and the two benchmark layouts.
It excludes source register declarations and the native transport header.
An independent repeated monomial shift checks every complex amplitude of
a coherent nonconsecutive register with a spectator. Applying the opposite
direction restores the full input.

| Degree-512 workload | CX | SWAP | Mark records | Source mutable generation | Native construction and lowering |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multiply by x | 3 | 511 | 2 | 6.3 μs | 55.09 ms |
| Multiply by x^512 | 1,536 | 0 | 2 | 27 μs | 58.91 ms |

Native timings are medians over seven runs after one warmup while local
checks ran on the same host. The table reports emitted records. Neither
path emits a measurement, reset, or non-Clifford primitive.

Run with `python -m benchmarks.benchmark_physical_field_monomial_scale`.
Regenerate with
`python tests/unit/fixtures/generate_kickmix_field_monomial_scale_file.py --source-root <locked-kickmix-root> --output tests/unit/fixtures/kickmix_field_monomial_scale_file.json`.
