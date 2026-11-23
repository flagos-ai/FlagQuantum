# Conjunction library constructor dependency audit

## Source and ownership

The reference is unmodified Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The following complete source,
original tests and catalogue blobs are verified in the runtime receipt.

| File | Git blob |
|---|---|
| `bloqs/mcmt/and_bloq.py` | `2e6efeb53938ac43f6f0330d79140f04138861eb` |
| `bloqs/mcmt/and_bloq_test.py` | `70d853579f933303cbfdd22c09674aec96067fc0` |
| `_infra/adjoint.py` | `0601a6248eec488301b31dd7c800a6ba14a7fb27` |
| `_infra/adjoint_test.py` | `f349c5dafbdc5ed7764d4aa39b0689fd14b9e4f7` |
| `serialization/resolver_dict.py` | `b10adb30b493afda81fcff2e330618415df655a2` |

`compiler/formats/legacy_conjunction_factories.py` owns explicit interchange
constructor mapping. Existing `registers/control/conjunction_program.py` owns
logical ports, ladder calls and clean target lifetimes. The existing And emitter
owns the exact compute and measurement-assisted erase recipes. Existing
Simulation and Runtime owners execute them. Prior complete source protocol
acceptance in `AND_FILE_AUDIT.md` is reused.

| Constructor | Interface and ownership |
|---|---|
| And | Defaults `cv1=cv2=1`, `uncompute=False`. Two ordered scalar controls in the shaped `ctrl` port. Forward produces a clean `target`. erase consumes and frees that known conjunction. Both zero-control polarities retain pre/post X. |
| MultiAnd | Required one-dimensional `cvs`, at least three controls. Ordered `ctrl`, produced `junk[n-2]`, produced scalar `target`. Caller owns produced live outputs until their specialized erase. |
| Adjoint of MultiAnd | Existing native reverse ladder consumes target and junk in reverse dependency order, uses the source measurement feedback and returns controls. Double adjoints restore the forward program. |

Constructor argument references retain the existing `FunctionRecord` contract.
The per-compilation factory catalog remembers MultiAnd recipes keyed by the
complete immutable resolved function. It regenerates the existing directional
owner for an adjoint, without inferring a computational operation from register
shape or a function name. A same-name foreign definition refuses. No global
registry, source SDK constructor import, public API or serialized schema is added.
Create a fresh catalog per compilation to bound retained context. Ambiguous
symbolic declaration identities refuse explicitly.

## Execution and accounting

The focused resolver consumer checks all binary control patterns at widths
2, 3 and 4, all measurement histories, complete unnormalized complex128 branch
amplitudes and zero work, including an entangled spectator. Independent basis
logic checks forward complex64 and complex128 amplitudes at widths 2, 3 and 6.
Normal `circuit.append(op)` and measured DynamicCircuit trajectories exercise
those same widths and both dtypes. The shared forced-branch inspector executes
complex128. it does not establish complex64 branch arithmetic.

Source-comparable operations, operands and classical feedback agree in each
wire's dependency order. Serialized examples at depths 0, 1 and 2 resolve.
Named CZPow and its directly serialized And-containing decomposition now produce
the same dependency-comparable primitive sequence. The existing complete
And/MultiAnd source tensor, classical, signature and resource evidence is reused.

For n controls, forward emits n-1 four-T And blocks. Erase emits n-1 H,
measure, conditional CZ and reset blocks. Every zero control adds two X gates
per direction in the source ladder. Logical junk and target are n-1 live output
wires, with no additional private work. Round-trip primitive counts are
4(n-1) total T/TDG, n-1 measurements and n-1 resets. These inventories are
measured separately from the source QEC model, whose T and measurement
categories are compared independently. All seven native QEC model categories
also agree with the original source in each direction. A complete round trip
emits 17(n-1)+4z primitives for z zero controls. Interface declarations emit
no primitives.

## Retained limits

The frozen serializer fails on symbolic MultiAnd `cvs` with an object-array
pickle refusal and on `HasLength(n)` with an unknown-type refusal. Those failures
are reproduced unchanged. Native explicit symbolic And control declarations
remain logical declarations and refuse physical binding. Native raw constructor
sequences can likewise retain a symbolic declaration, but this does not repair
the source wire format. Nonbinary concrete values and nonintegral physical
controls retain native physical admission refusal.

The generic source Adjoint wrapper supports far more constructors than this
catalog. Only the MultiAnd dependency and its nested adjoints are resolved here.
Explicit `Adjoint(And())` refuses because the source wrapper asks an atomic And
for a decomposition. ordinary source `And.adjoint()` uses the supported
specialized And constructor. Generic Adjoint/Controlled/Power catalogue closure,
other constructor mappings, pending Proposals 116–119, broad repository checks,
GPU and performance qualification remain open. Q-0876 remains open.

## Batch verification

The concentrated resolver/conjunction regression passes 417 cases. Affected local
integration passes 145 cases. The new factory consumer passes 38 cases with
100 percent changed-owner statement and branch coverage. Strict typing covers
the factory and focused test. Black, Ruff, Compiler scope, repository language,
architecture, IR and public API checks pass. The dependency checker retains the
identical 720 baseline findings and exits one. no added finding is accepted.

All 43 unchanged non-notebook And/Adjoint tests pass, with three original atomic
or protocol-mode skips. Both unchanged original notebooks pass separately after
providing the existing dependency paths to their local kernels. Initial notebook
runs failed at sandbox loopback binding and then missing kernel import paths.
these were environment failures resolved by the explicit kernel environment.
No reference source was modified. Five frozen blobs and actual native Kickmix
0.1 `Simulator` availability are verified. GPU, whole-repository and performance
qualification are outside this receipt.

Evidence is retained in `ftqc-reference-env/g2-conjunction-*-20261123` receipts
and logs. The tracked source-parent ledger stays open for the other catalogue
constructors and generic wrapper semantics. Two complete constructor identifiers
bring built-in coverage to 28 of 303. scoped Adjoint is an additional partial
identifier. The 275 remaining identifiers include that generic Adjoint gap.
