# Identity and exchange library constructor dependencies

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The complete `identity.py` and
`swap.py` constructor group uses existing native register and bit owners.
The new mapping supplies five explicit identifiers to
`resolve_component_program`, alongside fixed-gate and bookkeeping mappings.
Source names remain interchange data. Native construction executes no source
constructor or generic reflection.

## Interfaces and ownership

| Source constructor | Native owner | Signature and execution |
| --- | --- | --- |
| Identity | `registers/primitives/identity.py` | Default width 1, through q, one explicit I per concrete bit |
| TwoBitSwap | `registers/primitives/swap.py` | Scalar through x/y, one SWAP |
| TwoBitCSwap | `registers/primitives/controlled_swap.py` | Scalar through ctrl/x/y, exact original 17-instruction recipe |
| Swap | `registers/primitives/swap_component.py` | Required width, through x/y, ordered MSB bit exchanges |
| CSwap | `registers/primitives/controlled_swap_component.py` | Required width, through ctrl/x/y, ordered exact controlled exchanges |

Serialized signatures must equal the class-specific signature. Omitted
signatures derive from existing native declarations. Unknown parameters,
boolean/float/string widths and missing required widths fail before binding.
Symbolic widths retain signature-only logical declarations. Existing native
owners reject zero, negative and unresolved physical widths. The one-bit
Identity binds the source QBit signature directly. Larger identities retain
opaque word storage through the existing word implementation.

Resolver-assigned root names namespace all helper definitions and calls.
A mixed composite exercises two different Swap widths, CSwap and Identity at
three source serialization depths. Helpers remain independent across libraries.
The mapping composes with lifetime/view and fixed-gate owners through the
existing factory interface. A replacement hierarchical implementation uses the
same resolver, binder and `circuit.append(op)` consumer.

## Gate streams, amplitudes and resources

Tests compare dependency-comparable primitive operands and order against the
unmodified source emitter, complete dense operators against source tensors,
and full complex64/complex128 spectator amplitudes against the source simulator.
Physical ports are reversed. A separate clean wire remains exactly zero. Each
self-inverse operation restores the complete complex state. There are zero
measurement histories and zero private work requirements in these components.
Existing full-source control, tensor, drawing and symbolic evidence is reused
from `IDENTITY_FILE_AUDIT.md` and `SWAP_FILE_AUDIT.md`.

| Account | Identity(width) | Swap(width) | CSwap(width) |
| --- | --- | --- | --- |
| Measured primitive stream | width I | width SWAP | 8 width CX, 2 width H, 4 width T, 3 width TDG |
| Measurements / resets / private work | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Independent source/native QEC model | Zero cost | width Clifford | width Fredkin, default 4 width T-equivalent |

The exact source Fredkin recipe uses seven T/TDG per bit. Its independent QEC
model retains its own synthesis convention. Multiwire source Identity is atomic
and its Cirq decomposition may emit zero primitives. Native execution retains
one explicit I per bit. This historical execution distinction is preserved and
is excluded from claims of identical primitive expansion.

## Remaining boundaries

The complete source zero-width Identity export fails for a missing q argument.
Zero-width Swap/CSwap composite decomposition similarly loses required ports.
Source SymPy Integer physical-construction diagnostics, symbol serialization
assumption loss and nested multi-control failures keep their existing dispositions.
Source declarations are unchanged. Protected storage proposals 116–119 remain
pending. This mapping changes no Core contract or storage admission.

Built-in mappings now cover 18 of 303 source resolver identifiers. Q-0876
remains open for 285 other identifiers and generic reflection. Full source
Identity/Swap behavior was previously accepted. This batch closes their
constructor dependency group without recounting those parent files.

## Batch acceptance

The final concentrated native selection passes 337 cases without skips. The
affected local integration selection passes 145 cases with 18 existing emulation
deprecation warnings. Focused constructor acceptance passes 31 cases. Complete
complex amplitudes, operator equality, source-comparable streams, inverse state
restoration and separate model categories are checked through native consumers.
The changed factory owner has 100% statement and branch coverage. Strict typing
passes all three changed Python files. Black, Ruff, Compiler scope, language,
architecture, IR and protected API checks pass. Dependency policy retains the
identical 720 baseline findings. Five source/resolver/test Git blobs and the
locked Kickmix 0.1 compiled Simulator are verified separately. Previously accepted
source-file evidence is reused. Whole-repository and GPU validation retain their
existing open boundaries.
