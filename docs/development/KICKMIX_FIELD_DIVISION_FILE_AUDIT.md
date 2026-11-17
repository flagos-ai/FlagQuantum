# Binary-field division file audit

Kickmix reference commit `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

| Ledger | Source | Frozen blob |
| --- | --- | --- |
| K-0073 | `gen_gf_div.cc` | `55de676044e1b676430c70304d3d03894974311f` |
| K-0074 | `gen_gf_div.h` | `b9db6750a5f069bbcb73a39a1eebbaf43d2a5a30` |
| K-0075 | `gen_gf_div.perf.cc` | `09c4ec65e4c0fec14e9ede063a1b66fef840db14` |
| K-0076 | `gen_gf_div.test.cc` | `612a2319b038d80a46670d05202b4416f72a5c99` |

The division owner is `compiler/registers/finite_field/division_chain.py`.
Equal-width, big-endian dividend and divisor words can share an input
register. The target is disjoint from them and from all work wires. A zero
divisor contributes a zero quotient. Division XORs into the target. The
measured inverse requires the target to hold that quotient. The caller
supplies the inverse word and chain in consecutive big-endian words.
A retained forward call leaves the chain live for retained erasure. The
self-cleaning routes borrow the required prefix of the supplied work.

The physical path reuses the existing inverse and product schedules. It
retains leased-wire reset and reactivation records, the nested source calls
including linear-map stages, and precise measurement feedback. The first
parity scratch lease uses initially zero classical storage. Later leases
emit the source's explicit stores. The fixed-wire append schedules keep
their zero-initialization contract.

The fixture records all four overloads at degrees 1, 2, 4 and 8, for
distinct and shared inputs. It also includes the two degree-128 workloads.
Fingerprints compare all operations, operands, classical feedback and
nested markers after removal of only source register declarations and the
native transport header. All seven unchanged original tests passed. Native
small-field tests exhaust both lifetime styles' reset and HMR histories over
coherent complete input spaces, including zero divisors and returned work.
The previously accepted append tests supply the exhaustive GF(256)
arithmetic and relative-phase evidence.

| Degree-128 workload | Physical records | CX | SWAP | CCX | R | HMR | CZ | BIT_INVERT | BIT_STORE0 | Call record count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Divide | 765,462 | 516,353 | 6,135 | 28,431 | 1,664 | 1,664 | 196,608 | 12,648 | 1,523 | 436 |
| Undivide | 756,576 | 492,181 | 5,925 | 26,244 | 1,664 | 1,792 | 212,992 | 13,702 | 1,650 | 426 |

Both workloads lease 1,536 work wires and use the source default modulus
`0x100000000000000000000000000000087`. Counts are emitted records. The
source cost checks and generation timing are separate measurements.
Native timing covers construction, validation and lowering. Source timing
covers mutable stream generation. Both source cases measured about 11 ms.
The native medians were 12.39 s and 23.15 s across three runs after one
warmup, with the concentrated local checks running on the same host.

Run both native workloads with
`python -m benchmarks.benchmark_physical_field_division`.
Regenerate the record fixture with
`python tests/unit/fixtures/generate_kickmix_field_division_file.py --source-root <locked-kickmix-root> --output tests/unit/fixtures/kickmix_field_division_file.json`.
