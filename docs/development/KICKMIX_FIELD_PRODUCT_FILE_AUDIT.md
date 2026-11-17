# Binary-field product file audit

Kickmix reference commit `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

| Ledger | Source | Frozen blob |
| --- | --- | --- |
| K-0097 | `gen_gf_mul.cc` | `d4448e4c8216fc5ba82c3c492b94ea18e31e3799` |
| K-0098 | `gen_gf_mul.h` | `52ba16ba27561266c11d105ef810e2b28328a4f3` |
| K-0099 | `gen_gf_mul.perf.cc` | `066e8de6efa864b9d9d1ad484376777fce8b7604` |
| K-0100 | `gen_gf_mul.test.cc` | `479a123269dccb267c30b2eb8f1ada6a0882241d` |

The compiler owners are `polynomial_product.py` and `product_erasure.py`.
The binary polynomial accumulator uses two equal-width big-endian factors
and a disjoint target of width 2n−1. Field multiplication requires
equal-width operands and target, accumulates by XOR, and leaves the inputs
unchanged. Neither uncontrolled accumulator needs work wires. Uncontrolled
erasure requires the target to hold the known product. Controlled
multiplication and erasure both compute into a leased clean word,
conditionally XOR into an arbitrary target, and measure-clean the lease.
The controlled path returns the borrowed wires to zero.

The standalone phase uses a caller-owned bit for each coefficient of the
mask, with repeated physical slots allowed. It preserves phase-correction
order, including repeated toggles when two selected coefficients read the
same slot. Each parity group restores its scratch bit, and the first lease
uses an initially zero bit. Both physical and append paths reuse the
same product schedule. The physical route retains source resets and
nested calls.

All twelve unchanged original tests passed. The frozen fixture covers 45
complete streams including six repeated-mask cases and the three original
workloads. Source register declarations and the native transport header
are excluded from the digests. Native controlled-erase tests check
all small reset and HMR histories with a coherent arbitrary target.
The repeated mask case checks complete branch amplitudes and its parity
sign independently.

| Width-256 workload | Physical records | CX | CCX | SWAP | HMR | CZ | Source mutable generation | Native construction and lowering |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Carryless polynomial multiply | 69,101 | 62,540 | 6,561 | 0 | 0 | 0 | 390 μs | 0.73 s |
| Field multiply | 81,083 | 74,034 | 6,561 | 474 | 0 | 0 | 1.1 ms | 1.40 s |
| Field erase | 68,134 | 0 | 0 | 0 | 256 | 65,536 | 360 μs | 3.57 s |

Physical records also include call marks and classical operations. Field
erasure emits 2,084 BIT_INVERT, 254 BIT_STORE0 and four mark records.
These are primitive counts. Source resource estimates run separately.
Native timings are medians of three runs after one warmup while local
checks were running.

Run with `python -m benchmarks.benchmark_physical_field_product`.
Regenerate with
`python tests/unit/fixtures/generate_kickmix_field_product_file.py --source-root <locked-kickmix-root> --output tests/unit/fixtures/kickmix_field_product_file.json`.
