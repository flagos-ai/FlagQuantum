# Kickmix binary linear map parity

This audit covers K-0101 through K-0104 at Kickmix revision
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

| Source file | Git blob | Native coverage |
| --- | --- | --- |
| `gen_linear_map.cc` | `33e44b9d7508129c1360a750522cca383f46864a` | In-place PLU synthesis and inverse-matrix synthesis |
| `gen_linear_map.h` | `330b84396c040ca500139d42e17e15895ac2e2d2` | Forward and adjoint physical export, operand order and admission |
| `gen_linear_map.perf.cc` | `d4cc7d7bcd4e221bf4d5bab6c44d8a3aaf50a34f` | Both fixed-wire Frobenius matrix workloads |
| `gen_linear_map.test.cc` | `85e140015224b17a452c584f362983130e1d74ff` | Original suite and independent physical and complex-state evidence |

The target tuple is big-endian. Packed column zero selects the least
significant input bit, and bit zero in a column selects the least significant
output bit. Physical export reuses the compiler's PLU implementation. The
adjoint inverts the input matrix and synthesizes the result in the same PLU
order as the source. The map acts in place, with zero workspace and zero
classical bits. A nonempty identity preserves the source's enclosing block
mark. An empty map produces no operation or mark and uses the transport's
minimum-width placeholder. Native inputs require a square width-matched
matrix of packed integers, distinct valid target wires and a boolean adjoint.

The fixture contains 46 valid source streams, including permutations,
identities, seeded invertible matrices and both workload matrices. Tests check
complete serialized streams, primitive counts, block marks, KMX round trips
and 12 matching singular-matrix refusals. One regeneration reproduced the
committed fixture byte for byte. A strided register entangled with a spectator
checks every complex amplitude against an independent matrix action and
restores all amplitudes after an adjoint applied through `Circuit.append`.
The nine unchanged source tests, seven native focused tests and 5,272
concentrated regressions passed. Formatting, strict typing, architecture
boundaries, internal IR and the public API baseline also passed.

## Physical workload counts and runtime

Both workloads build the exact locked Frobenius(1) matrix outside the timed
region. Source timing measures C++ generation. Native timing is the median of
seven Python synthesis and physical lowering runs after one warmup, collected
while local checks were running. These are machine-specific measurements of
the two execution paths.

| Width | CX | SWAP | Block mark records | Source generation | Native generation and lowering |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 125 | 58 | 2 | 15 μs | 2.282 ms |
| 256 | 397 | 249 | 2 | 300 μs | 10.786 ms |

The table counts emitted operations and serialized marks. A fault-tolerance
cost model was outside this batch.

Regenerate the source fixture from the locked tree with:

```sh
python tests/unit/fixtures/generate_kickmix_binary_linear_map_file.py \
  --source-root <locked-kickmix-root> \
  --output tests/unit/fixtures/kickmix_binary_linear_map_file.json
```
