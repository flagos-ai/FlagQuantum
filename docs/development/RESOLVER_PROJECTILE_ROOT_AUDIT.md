# Projectile SELECT/PREPARE constructor dependency closure

Reference: unchanged Qualtran v0.7.0 complete
`chemistry/pbc/first_quantization/projectile/select_and_prepare.py`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Classification:
`single_device_fast_path`.

## Interface recorded before implementation

Ecosystem adapts the original finite constructors using existing native
projectile signatures and independently owned T/UV/nu/zeta dependencies.
Compiler layout owns literal 3D reshape/exchange calls; preparation and
selection composition keep separate owners. The existing ComponentProgramBuilder
owns MSB words, row-major arrays and SSA lifetimes. No SDK or Algorithms import
belongs in Compiler; no numerical execution or backend choice is introduced.

Selection words and system/projectile banks stay caller-owned. PREPARE preserves
plus_t because the source Hadamard is commented out. SELECT allocates its
three p words, three q words and rl, and retains the original release edges.
Opaque child bodies must fail physical lowering rather than become identities
or establish a false clean-release guarantee.

The controlled 3D wrapper uses cvs for labels but omits it from the actual
MultiplexedCSwap child. Its original physical activation is all-one. This
source-exact path must stay separate from the existing explicitly corrected
native control_values operation. The three projectile-word suffixes are
caller-owned data and must never be reset or silently discarded.

The source PREPARE coordinate words are two bits while SELECT uses three.
Existing corrected recursive adjoint models and original invalid mu-flag
adjoints remain distinct. Full-file acceptance is not claimed by this initial
interface record.

## Delivered constructor and dependency scope

The complete four-class source file and four direct dependency identifiers bind
through `ecosystem/_projectile_component_library.py`. It reuses authoritative
native signatures and the accepted kinetic/potential/nu/zeta constructors.
Compiler layout, preparation and selection retain distinct owners. No SDK or
Algorithms implementation enters Compiler.

Both original examples and an independent small parameter set preserve the
source signature, root node IDs, named producer/port/array identities and
ordered results at serialized depths 0/1/2. PREPARE has four calls; SELECT has
26, including seven allocation and seven free calls. Both nu axes and four
Hamiltonian flags retain their literal connections. PREPARE plus_t returns the
same SSA value because the original Hadamard is commented out.

The two reshape wrappers retain every original Split/Join and child operand,
including the restoration order of targets before junk. Their exact logical
adjoints use the existing owned multiplexed inverse; measured primitives are
not reversed. One- and two-control physical streams match the accepted original
source-compatible operation per wire, with fresh feedback slots checked before
source-key normalization. Explicit corrected negative-control operations retain
their existing separate semantics.

Coherent complex inputs span every valid selector/control word and all data
and suffix states in the small fixtures. Every applicable measurement history
matches an independent conditional-permutation reference, including absolute
complex phase and zero private work, in both directions. Larger fixtures check
the full stream and metadata without claiming exhaustive large-state execution.
Root constructor resolution also works with Qualtran and Cirq imports blocked.

## Empty controls and source protocol refusal

The original empty cvs array has a valid shallow graph. Its native typed
context executes the unconditional addressed prefix exchange, preserves three
live suffixes and uses no private work. It matches the existing independently
verified ordinary exchange on those nonconsecutive prefix wires and the full
coherent permutation reference.

The original Bloq resource query fails with
`ValueError: Cannot apply_along_axis when any iteration dimensions are 0`
during recursive Cirq-style conversion. The test preserves that failure; no
original complete tensor or QEC estimate is attributed to this case. The new
typed context does not widen the existing flat controlled-operation contract.
Source raw integer/boolean cvs labels survive library parsing; their values
still do not supply the argument omitted from the source child call.

## Executed primitives and declaration models

| Exchange cvs | Actual instructions | T/Tdag | Measurements | Private peak |
| --- | ---: | ---: | ---: | ---: |
| empty | 104 | 42 | 0 | 0 |
| (0,) | 122 | 46 | 1 | 1 |
| (0, 1) | 139 | 50 | 2 | 2 |

All leases return. Both nonempty source QEC models are retained separately;
their logical peaks are 15 and 17. The empty-array model query refuses as above.

| Root parameters p/n/eta/atoms/charge | Root | Toffoli | CSwap | And | Clifford | Model measurements | Model peak |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2/3/2/3/3 | PREPARE | 332 | 0 | 15 | 168 | 15 | 64 |
| 2/3/2/3/3 | SELECT | 301 | 66 | 17 | 95 | 17 | 97 |
| 6/8/10/10/10 | PREPARE | 733 | 0 | 25 | 219 | 25 | 104 |
| 6/8/10/10/10 | SELECT | 707 | 768 | 76 | 348 | 76 | 345 |

Other gate categories are zero at this model boundary. All seven categories and
peaks match the existing native model owners. Whole-root primitive inventories
are unavailable, not zero: the PREPARE module contains nine opaque definitions
and SELECT two. Resource formulas are independent declaration estimates, not an
automatic physical pricing certificate for unknown compiled leaves. Receipt:
`g2-projectile-root-resources-20261127.json`.

## Validation and remaining boundaries

The concentrated batch passes 195 cases without skips in each actual
Cirq 1.7.0/Qiskit 2.5.0 and Cirq 1.6.1/Qiskit 2.0.0 profile. Original non-notebook
tests retain eight passes/four skips with one notebook deselected. Prior tutorial
acceptance and the existing notebook declaration checks are reused; a fresh
notebook-kernel execution is not claimed. Five frozen source/test/dependency
blobs are verified. The launcher asserts Qualtran 0.7.0 and actual Kickmix 0.1
Simulator availability. Five Python files pass strict typing, Black and Ruff;
scopes, language/links, architecture, IR and protected API checks pass.

Opaque source physical leaves, generic root adjoints, the invalid recursive mu
flag inverse, absent PREPARE L1 norm and broader symbolic/invalid physical
domains remain open. The source cvs-display/physical-activation discrepancy and
empty-array conversion refusal are not waived. Whole-root physical binding
explicitly refuses missing bodies. Registration is 227/303; Q-0382, Q-0866 and
Q-0876 remain open. No original source is modified or PR merged.
