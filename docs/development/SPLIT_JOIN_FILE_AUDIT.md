# Split and Join source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Frozen bookkeeping files are
Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0325 | `join.py` | `b194add90de87b8aeeb05dd662747701394f400a` | Open for the full external dtype, wire-symbol and direct object protocols. |
| Q-0326 | `join_test.py` | `8cd06b0d322692c18bd66b659bf85b9b76ddb3cf` | Closed for the runnable original scenarios. |
| Q-0330 | `split.ipynb` | `f7150b192af058a668f5efb0d599fdbe03f658bd` | Closed by the complete native Split and Join tutorial. |
| Q-0331 | `split.py` | `ab02e2e6f41402c732a9f30635e83227a09d2c4a` | Open for the corresponding dtype and external hooks. |
| Q-0333 | `split_test.py` | `8efb0a620f8939ee3c4d494f519afdd2a707e284` | Closed for the runnable original scenarios and notebook. |

The Compiler's existing typed rebinding consumes each owned input exactly
once. The word and bit array both use `reg`, on opposite sides. Split
preserves the physical order and exposes an array of one-qubit words
in big-endian order. Join uses the order of the supplied bits and
declares the output dtype. An output permutation is a wire alias
change. No SWAP gates, workspace, or measurement are emitted by either
operation. The typed ports match the source at widths five and 123,
among the tested widths. Both models report zero gates and logical
peak equal to the declared width.

The original ten-wire Allocate–Split–Join case transfers the live
output lease with all physical wires in caller order. The full 1,024
component complex output is the source zero state. The original
four-wire example flips the second logical bit and returns the value
four. The native circuit emits exactly one X for the child operation.
Existing tests also cover complex spectators, permuted return bindings,
single consumption, and measured downstream callers.

Simulation owns the numerical encode and decode order. Split uses
`np.asarray` without coercing the raw callback result to a fixed NumPy
dtype. The typed bit-array collector separately produces the source
call's `uint8` output. Join passes the list of bits to the declared decoder. The original
255 and overflow cases match for Python integers and NumPy scalars.
The original suite covers NumPy's version-specific scalar-constructor
behavior. Empty wire views are valid, while the original zero-width
integer paths fail. Symbolic Split and Join are refused by the
original dtype constructors.

The unchanged original non-notebook run has twelve passes and six
fixture-mode skips. The original 14-cell and native 16-cell notebooks
passed independent kernels. The native tutorial preserves both
four-bit signatures and demonstrates zero-gate bindings around the
original one-X example through append. General external dtype support,
wire-symbol callbacks, and direct foreign object adapters remain open.
