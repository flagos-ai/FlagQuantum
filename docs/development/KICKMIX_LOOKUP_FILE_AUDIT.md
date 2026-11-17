# Kickmix lookup and unlookup parity

This audit covers K-0109 through K-0112 at Kickmix revision
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

| Source file | Git blob | Native coverage |
| --- | --- | --- |
| `gen_lookup.cc` | `d527eeeddf06c10ffe69ab6408594c60fd248182` | Lookup, unlookup, binary/unary conversion and the known incomplete-table phase gap below |
| `gen_lookup.h` | `248ec9915419e51c5183c1c37f69df80d79b2fcf` | Big-endian views, operand kinds, workspace ownership and slot contracts |
| `gen_lookup.perf.cc` | `fea4812e05e740ac486b0d7fce57ca895f253a37` | Both 8-bit address, 30-bit output, dynamically bound classical-table workloads |
| `gen_lookup.test.cc` | `99cbcc91f13cc0df4c534407f507942d8d502275` | Nine unchanged tests, including four diagrams and five fuzz tests |

The compiler reuses three independently accepted owners:

- [Binary/unary conversion](REGISTER_BINARY_UNARY_CONVERSION.md) provides the
  fixed in-place buffer and measured adjoint (`b265e105`, 56 streams).
- [Recursive lookup](REGISTER_RECURSIVE_LOOKUP.md) provides coherent X/Z lookup
  for complete tables with constant, classical or quantum coefficients
  (`1b5f4195`, 480 streams).
- [Measured unlookup](REGISTER_MEASURED_UNLOOKUP.md) clears a previously
  computed output, composes both conversion directions with a full phase
  lookup and reuses the caller's phase and feedback slots
  (`5054899c`, 783 streams).

The underlying test suites check complete small complex states, reset and
measurement histories, workspace cleanup, append entry points, and source
gate counts. The current lookup-specific rerun passed 412 tests and the concentrated regression
passed 5,413. Strict typing, formatting, scope, architecture, internal IR
and public API checks passed. The two
new workload checks retain the complete opcode, operand, feedback and block
mark order. They check the classical inputs declared at the physical
transport boundary, the exact serialized source digests and decoding back
to the native program. A fresh C++ build against the unchanged generator
reproduced the fixture byte for byte.

## Workloads and counts

Both workloads allocate eight address wires, 30 output wires, ten clean
workspace wires and 7,680 caller-owned classical table bits in source order.
The forward case uses one additional measurement slot. Erasure has 256 phase
slots and one additional measurement slot. Register declarations are
excluded from the comparison. All other serialized records match.

| Workload | Emitted operations and scopes | Mark records | Total records | HMR | CCX | Locked C++ generation | Native Python generation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lookup | 9,465 | 1,022 | 10,487 | 255 | 254 | 40 μs | 22.733 ms |
| Measured unlookup | 8,484 | 68 | 8,552 | 57 | 14 | 80 μs | 22.036 ms |

The committed fixture gives the remaining opcode counts, including
7,680 classical inversions during erasure. These are physical records.
Expanded runtime instructions and fault-tolerance cost models are
separate. Native timing is the median of seven runs after one warmup,
measured while local checks ran. Layout construction is outside the timed
region for both implementations. Native measurement includes physical
program validation; the locked C++ workload times builder generation.

## Open functional boundary

An incomplete controlled table can leave a low-only subtree. In the
locked source, HMR feedback for that subtree can introduce an extra
phase on the parent control in outcome one. The native source-ordered
emitter reproduces this effect. With a one-bit controlled all-false table
and coherent control, the affected output can be orthogonal to ideal
identity, even though workspace clears. Complete tables with explicit
zero rows support ideal coherent lookup. K-0109 remains open for the
incomplete-table functional boundary.

Run the native workloads or regenerate their source evidence:

```sh
python -m benchmarks.benchmark_physical_lookup
python tests/unit/fixtures/generate_kickmix_lookup_workloads.py \
  --source-root <locked-kickmix-root> \
  --output tests/unit/fixtures/kickmix_lookup_workloads.json
```
