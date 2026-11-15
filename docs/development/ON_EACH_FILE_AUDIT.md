# OnEach source family

The unchanged reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Both source files retain the
Google LLC 2023 Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0246 `on_each.py` | `0d8a190709e5465a7305858a1ba895ab284c28b4` | Concrete owned QBit composition, numeric action and symbolic resource evidence complete for their admitted domains. Generic constructor and drawing equivalence remain open. |
| Q-0247 `on_each_test.py` | `7ae02d5013c738e0aaf202c10b3d39b4aec50fb9` | All four original test scenarios now have native counterparts. |

## Ownership and interfaces

The locked source validates one child register named `q` of width one. It
declares the outer `q` as a through-going word with an optional target dtype.
Its concrete body splits the word, calls the child on each indexed bit in
order, and joins the results with the target dtype. The symbolic model keeps
one child edge with weight `n`. The source refuses symbolic body construction.

The native `on_each_word_program` uses the existing typed component builder.
It consumes the input binding through Split, connects each child output to
the corresponding Join input, and returns an owned word with the same dtype.
Bit index zero is the most significant bit. Physical binding accepts the
caller-provided order, including noncontiguous or reversed wires. The operation
introduces no scratch or feedback. Workspace used by a child remains the
binder's responsibility. Width zero keeps Split and Join in the logical body
with zero child calls. A physical plan still requires at least one physical wire;
the empty word can share a plan with a spectator and emits no instructions.

The typed adapter admits a self-contained child with one scalar QBit THRU
port named `q` and a concrete quantum target width. The generic source
constructor admits extra metadata shapes, direction cases and dtype quirks
that the adapter does not expose. The numerical owner retains its previously
verified codecs and special width cases. Source `wire_symbol` appends
`⨂n` to child text and preserves empty text. General forwarding of this
foreign presentation and its fallback for other glyphs remain open, along
with those constructor cases. These gaps keep Q-0246 open.

## Original test coverage

| Original function | Native counterpart |
|---|---|
| `test_valid_bloq` | Ten-bit X body verifies as a typed Program and binds to the supplied reversed wire list. The ten child calls sit between Split and Join. |
| `test_tensor_contract` | Five-bit H appended through the typed binder matches the complete locked 32-by-32 complex matrix. |
| `test_classical_simulation` | Existing codecs and the native X value callback map input zero to 1023 with phase one. The bound numerical path propagates an unavailable H action at the first bit. The source likewise refuses classical H evaluation. |
| `test_call_graph` | Ten-bit X and H selected counts match locked child multiplicity and emitted scans. The original Split/Join-generalized declarations also match the corresponding original body counts. |

The admitted classical path takes an explicit leaf callback. The H check
exercises propagation of the leaf refusal; no automatic foreign action lookup
is implied.

All four unchanged source tests passed. The 658-case native regression and
109-case integration profile passed without skips. Strict typing, formatting,
team scope, architecture, multilevel Program IR and the public API baseline
also passed.

## Executed gates and declared costs

Ten-bit X emits ten X instructions. Five- and ten-bit H emit five and ten H
instructions. The respective source declarations count the same number of
Cliffords. These examples have no measurements or clean work wires. The
three-bit X/Z example runs through the typed body while keeping a spectator
in its four-wire circuit. It emits six Clifford instructions across its
forward path, with source peak width three. The inverse restores the complete
complex state.

```bash
python -m examples.algorithms.classical_on_each_values
```
