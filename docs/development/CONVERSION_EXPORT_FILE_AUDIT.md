# Conversion package export whole-file audit

Q-0188 covers the entire four-import initializer
`qualtran/bloqs/arithmetic/conversions/__init__.py` at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. Its frozen Git blob, original
import order and exact identity with all defining modules are checked by
`test_arithmetic_export_file.py`. No executable or resource formula occurs in
this initializer.

| Source export | Native owner | Existing consumer and actual contract |
| --- | --- | --- |
| ToContiguousIndex | Compiler `registers/sorting/contiguous_index.py` | `test_contiguous_index.py` compares every small input, occupied result, measurement history, coherent append and clean work. The original callback and model are checked separately. The source has no decomposition. |
| SignedIntegerToTwosComplement | Compiler `registers/bits/signed_encoding.py` | `test_signed_encoding.py` checks original carry gate order, controls and feedback, both encoding directions, coherent spectator amplitudes, all small histories and scratch reset. The original one's-complement signature and independent model map to `ones_complement_to_twos_ir`. The separate sign-magnitude construction retains its independent paper contract. The original root has no decomposition. |
| SignExtend | Compiler `registers/bits/sign_extension.py` | `test_sign_extension.py` binds the original extension/truncation composite and compares its full native gate stream for two's and one's complement. Appending computes prefix+source and preserves the complete coherent input amplitudes. |
| SignTruncate | Same `sign_extend_ir` owner | The same source fanout clears a correctly redundant prefix. The second append is the actual inverse path and returns prefix to zero. Arbitrary high bits are outside that physical contract. |

## Interface and evidence reuse

Words are MSB-first. Sign conversion owns its explicit scratch and measurement
slots. Contiguous indexing XORs into a bound occupied result with caller-owned
work. Sign extension copies the original sign into a zero caller-owned prefix.
Truncation restores that prefix to zero on the sign-extended subspace. Its
logical output is the retained source word. No separate duplicate truncation
emitter or foreign-class registry is introduced. These physical workflows use
ordinary `circuit.append(op)`.

The retained prefix is physical work for truncation and part of the logical
output for extension. For an extra prefix width e, the actual original fanout
emits 2e-1 CX in each direction, no T gates and no measurements. Independent
source model checks remain separate from measured primitive counts. The
initializer itself emits zero primitives and owns no work or branches.

Q-0188 closes the complete package-export responsibility only. Conversion child
implementation and original-test IDs keep their independent generic foreign
constructor, symbolic physical admission, typed rebinding and original source
boundaries. ToContiguousIndex's missing source decomposition is explicitly tested
and is not replaced by a claimed original gate stream. Existing gate, complex
state, branch, zero-work and resource evidence is rerun rather than re-created.
Logs are `ftqc-reference-env/g2-conversion-export-file-*.log`.

The concentrated batch passed 80 cases without warnings or skips. Formatting,
lint, strict typing of the export test, Compiler scope, architecture, multilevel
IR and protected API checks passed. The preceding arithmetic audit's warning
attribution was corrected from the full log to IPython's temporary-directory
fallback and the notebook's non-interactive figure warning.
