# Binary-field Frobenius file audit

Kickmix reference commit `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

| Ledger | Source | Frozen blob |
| --- | --- | --- |
| K-0081 | `gen_gf_ifrobenius.cc` | `b4edd6b7339591e8831db6d1cea4781314f09a94` |
| K-0082 | `gen_gf_ifrobenius.h` | `86d05c96092e472f4388d0401ec6b6c75babf215` |
| K-0083 | `gen_gf_ifrobenius.perf.cc` | `34ddd40c8900e1adbd8b918e560a21bec1340f0b` |
| K-0084 | `gen_gf_ifrobenius.test.cc` | `b8cacee36c2b4d567d8e71f446ff0157bcca2c68` |

The owner is `compiler/registers/finite_field/binary_linear_map.py`.
The call transforms one big-endian polynomial word in place, with the field
degree matching the word width. The forward call applies the selected power
of squaring. The adjoint synthesizes the inverse map through the same PLU
schedule with the negated exponent modulo the degree. Counts that are multiples
of the degree, including zero, produce an empty stream and no block marks.
The call borrows no quantum or classical workspace.

The standalone physical binding and inverse-chain composition use the same
source call spans for Frobenius, its adjoint, and the nested linear map. The
existing register IR keeps its append entry. The new tests verify complete
complex amplitudes on a nonconsecutive register entangled with a spectator.
An independent repeated-squaring calculation checks the output permutation,
and the adjoint returns the full state. The source's seven original tests
passed, including periodic, large-degree, rejection, and inverse cases.

The fixture compares 86 complete streams after excluding source register
declarations and the native transport header. It covers identity, original
control order in CX and SWAP, arbitrary source-size counts through the 64-bit
boundary, and both original degree-256 workloads. Neither workload emits a
measurement, reset, or non-Clifford primitive.

| Width | Squarings | CX | SWAP | Mark records | Source mutable generation | Native construction and lowering |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 397 | 249 | 4 | 260 μs | 16.87 ms |
| 256 | 128 | 32,352 | 127 | 4 | 1.1 ms | 319.37 ms |

Native timings are medians of seven iterations after one warmup with local
checks running on the same host. The table reports emitted records. The
source's separate test for zero CCZ-state weight uses an estimator.

Run with `python -m benchmarks.benchmark_physical_field_frobenius`.
Regenerate the source digests with
`python tests/unit/fixtures/generate_kickmix_field_frobenius_file.py --source-root <locked-kickmix-root> --output tests/unit/fixtures/kickmix_field_frobenius_file.json`.
