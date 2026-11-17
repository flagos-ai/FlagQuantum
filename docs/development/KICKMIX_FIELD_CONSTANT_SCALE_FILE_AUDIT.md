# Binary-field constant scale file audit

Kickmix reference commit `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

| Ledger | Source | Frozen blob |
| --- | --- | --- |
| K-0085 | `gen_gf_imul_classical.cc` | `7b679a3e90bf211e432c95336599250ffd233d3b` |
| K-0086 | `gen_gf_imul_classical.h` | `faa779ea3eb5b5364501bbc68847b0da5ebf7f18` |
| K-0087 | `gen_gf_imul_classical.perf.cc` | `5863d55574d1ac7a654bd8eb74f945d7e39f7fd5` |
| K-0088 | `gen_gf_imul_classical.test.cc` | `5f0397e8e44d43b3c4b04a648756e86a40c15501` |

The owner is `compiler/registers/finite_field/binary_linear_map.py`.
The two directions consume a big-endian polynomial word whose width is
the field degree. The classical factor is reduced before any circuit
emission. Reduced zero is rejected. Reduced one emits neither gates nor
call marks. Division computes the reciprocal in the field then uses the
same PLU synthesis as multiplication. Both directions run in place and
borrow no quantum or classical workspace.

The physical binding and field-product composition share the constant scale
and nested linear-map call ranges. The append entry keeps the fixed-wire
operation schedule. Native coherent validation compares every amplitude on a
nonconsecutive register entangled with a spectator. A separate carryless
multiplication of the unreduced factor followed by polynomial reduction
supplies the reference permutation. Division checks its inverse permutation.

Seven unchanged original tests passed. The fixture compares 68 admitted
physical streams and 22 source-rejected factors, including nonzero unreduced
multiples of the modulus. For admitted inputs the digest excludes only
source register declarations and the native transport header.

| Degree-256 workload | CX | SWAP | Mark records | Source mutable generation | Native construction and lowering |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant multiply | 25,652 | 42 | 4 | 380 μs | 565.83 ms |
| Constant divide | 28,898 | 54 | 4 | 390 μs | 567.00 ms |

Native times are medians over seven runs after one warmup with concurrent
local checks. Counts are emitted records. Neither path emits a
measurement, reset, or non-Clifford primitive.

Run with `python -m benchmarks.benchmark_physical_field_constant_scale`.
Regenerate with
`python tests/unit/fixtures/generate_kickmix_field_constant_scale_file.py --source-root <locked-kickmix-root> --output tests/unit/fixtures/kickmix_field_constant_scale_file.json`.
