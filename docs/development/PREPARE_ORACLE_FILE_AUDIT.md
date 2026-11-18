# PREPARE oracle base: whole-source closure

Q-0693 maps the complete `qualtran/bloqs/state_preparation/prepare_base.py`
at Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, Git blob
`6870d7752aeb7da0c251ad1abb3b90519f2d0556`. The original is Apache-2.0,
copyright 2023 Google LLC; the native declaration retains that attribution.

## Complete definition map

| Source definition | Native owner and evidence |
| --- | --- |
| `PrepareOracle.selection_registers` | `algorithms/primitives/prepare_oracle.py`: the same abstract property. Neither base can be instantiated without it. Concrete children retain ordered native register descriptors. |
| `junk_registers` | The same cached empty tuple by default. Overrides preserve shape, dtype and LEFT/RIGHT/THRU sides; auxiliary descriptors are not automatically scratch or through ports. |
| `signature` | The same cached selection-then-junk concatenation. Core owns register/signature data; the base does not flatten or reinterpret it. |
| `l1_norm_of_coeffs` | The same optional scalar-expression property and `NotImplementedError` default. No missing norm is replaced with zero or one. Concrete and symbolic norms pass through packed binding and SELECT/LCU/walk metadata. |
| `_PREPARE_ORACLE_DOC` | Native class documentation and this audit preserve the protocol. The frozen documentation record has no examples or hidden factory; its class/import/empty-example assertions are checked. |
| Inherited SDK gate machinery | Conversion stays in Ecosystem and lowering stays in Compiler. The native base declares metadata; owned emitters and existing bound operations supply `circuit.append(op)`. SDK inheritance is not copied into Algorithms. |

The original teaching contract is

\[
\mathrm{PREPARE}|0\rangle = \sum_l \sqrt{c_l/\lambda}\,
|l\rangle|\mathrm{junk}_l\rangle,\qquad \lambda=\sum_l |c_l|.
\]

This is a declaration of the oracle's meaning. The base contains no circuit
decomposition, numerical simulator, resource estimator or symbolic execution
body. A particular preparation owns those algorithms and their domain promises.
Unresolved width metadata is retained rather than converted to a numerical
circuit. A symbolic normalization with concrete wire widths is metadata and
does not prevent execution of an otherwise concrete preparation.

## Operational replacement and ownership

`CircuitRegisterEmitter` owns native forward and adjoint IR. The optional
`import_register_emitter` converts a concrete external implementation immediately
at the SDK boundary. Existing `BlackBoxPrepare`, SELECT and LCU consumers then
use the same named callback interface; native PREPARE implementations and
captured external implementations can replace one another without changing
those consumers. The original source adjoint is captured independently.

Logical THRU junk remains an arbitrary logical input/output. Explicit RIGHT
outputs occupy persistent preparation-local storage, survive SELECT and are
consumed by the adjoint. Both are distinct from clean temporary scratch.
Source metadata retains the complete directed signature. The physical binding
uses row-major array order and big-endian bits within each word. Source objects
do not become native IR or native callback state.

The complete external replacement evidence is in
[the SELECT/LCU audit](LCU_BLOCK_ENCODING_FILE_AUDIT.md): original Hubbard
named/packed factories, scalar and shaped local outputs, arbitrary through
junk, returned-wire aliases, symbolic normalization, zero selection width,
complete small operators, measurement histories and zero private work. Its
physical primitive inventory and source resource declarations remain separate.
The original LCU wrapper's missing LEFT-junk input is a defect in that wrapper,
retained under Q-0283; the base correctly preserves the descriptor and does not
invent a missing caller input. Closing the base does not close that consumer.

The previous native integrations remain in place: sparse chemistry, THC and
Hubbard share this declaration, and their walks consume its coefficient norm.
Hubbard keeps bounded coordinate descriptors and the two-bit `temp` word;
sparse and THC keep their selectors and QROAM banks in source order. Their
accepted gate, resource, measurement and clear-work evidence is reused. The
distinct packed adapter, including the retained Hubbard norm-32 fixture, stays
under [Q-0691](BLACK_BOX_PREPARE_FILE_AUDIT.md).

The former symbolic execution gap now has an actual binding test. A source
subclass declares symbolic width `n`, rotation angle `theta` and norm `2**n`.
Immutable constructor parameters are explicitly substituted for widths 1/2/3;
the reference body remains the original Hadamard/Rz calls. The corresponding
native declaration is resolved and bound to the owned external emitter through
`PackedPrepareDeclaration.bind`, then appended in both directions. Every full
unitary matches the unchanged resolved source, including absolute complex
phase and native IR JSON roundtrip. Each case emits `n` H and one Rz, without
measurement or workspace; the independent source declares `n` Cliffords, one
rotation and peak `n`. An unresolved width is refused before capture or binding.
This closes both previously recorded gaps, external subclass replacement and
symbolic-parameter binding, using the existing constructors and binding API.

## Verification and scope

`test_prepare_oracle_file.py` verifies the pinned blob, abstract property set,
defaults, optional norm failures, cached signatures, symbolic and directed
metadata, native Hubbard sizes 2/3/4, distinct packed binding and the complete
documentation record. These checks are reused with the complete external
replacement and branch proofs; closure is not inferred from an AST count.
The abstract base has no executable body whose gate count could be zero-priced.
All 14 base-file cases pass without skips. The dependency closure passes
361 concentrated cases, 60 local integration cases and the independent LCU
notebook kernel; all 169 Algorithms and touched adapter files pass strict typing
at the configured Python floor. Formatting, ownership, architecture, multi-level
IR and the protected public API baseline also pass.

```bash
pytest -q tests/unit/test_prepare_oracle_file.py tests/unit/test_lcu_sided_junk.py
```

Q-0693 is closed as a complete declaration file and its binding dependency
closure. No parent ID or capability checkbox was added. Concrete preparations,
consumer defects, inherited general construction protocols and backend-specific
execution retain their own inventory entries and acceptance evidence.
