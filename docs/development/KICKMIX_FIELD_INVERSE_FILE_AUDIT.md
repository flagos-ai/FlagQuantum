# Binary-field inverse file audit

Kickmix reference commit `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

| Ledger | Source | Frozen blob |
| --- | --- | --- |
| K-0093 | `gen_gf_inverse.cc` | `3988c793db4c65454b555548107c77ab2e3c531f` |
| K-0094 | `gen_gf_inverse.h` | `0a18c75e91b268036209f527bd29ac2d82e4bc58` |
| K-0095 | `gen_gf_inverse.perf.cc` | `1f914982bef1a3ffc898f4616bc92edff2ba5108` |
| K-0096 | `gen_gf_inverse.test.cc` | `485a86dba3c67d5435915d3c77b5724e2e86308c` |

The owner is `compiler/registers/finite_field/inverse_chain.py`. The
source and target are disjoint big-endian field words of equal width.
Forward starts with a clean target, writes the multiplicative inverse
and maps zero to zero. Erasure consumes a known inverse. The caller
supplies the ordered live chain for the retained path. The self-cleaning
path borrows its required workspace prefix and returns those wires to
zero. Excess leased wires are left alone.

The physical export reuses the source-ordered doubling and folding
schedules and includes their nested addition, product, and linear-map
calls. It emits the source workspace resets, the reactivation of the
post-fold scratch word, and destructive measurement feedback. Inversion
and field division share the first parity lease's initially zero
classical bit. The fixed-wire append path retains its clean-input
contract.

The frozen fixture checks all four overloads and both degree-128
workloads. All 34 streams match after removing source register
declarations and the native transport header. Eleven original tests
pass. Small coherent checks cover the inverse of every input including
zero, both workspace ownership modes, and all selected reset and
measurement histories. The retained inverse followed by erasure
restores the full state, including the clean target and chain.

| Degree-128 workload | Workspace | Physical records | CX | SWAP | CCX | R | HMR | CZ | Mark records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Self-cleaned | 1,536 | 735,152 | 488,563 | 5,822 | 26,244 | 1,664 | 1,664 | 196,608 | 416 |
| Caller-retained chain | 1,408 | 414,860 | 384,090 | 4,256 | 26,244 | 0 | 0 | 0 | 270 |

The self-cleaned route also emits 12,648 BIT_INVERT and 1,523 BIT_STORE0
records. These are emitted counts. The source resource-model assertions
run separately. Source mutable generation measured 17 ms and 7.3 ms.
Native construction and lowering medians were 7.55 s and 3.22 s over
three runs after one warmup while local checks were running.

Run with `python -m benchmarks.benchmark_physical_field_inverse --warmup 1 --repeats 3`.
Regenerate with
`python tests/unit/fixtures/generate_kickmix_field_inverse_file.py --source-root <locked-kickmix-root> --output tests/unit/fixtures/kickmix_field_inverse_file.json`.
