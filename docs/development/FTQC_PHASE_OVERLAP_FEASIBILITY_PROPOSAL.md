# FTS-10C: Phase and Overlap Feasibility Before Method Selection

Status: the human explicitly approved the finite M-P prototype and acceptance
on 2026-10-26. The implementation below is under separate review after proposal
#911 actual-main admission. This does not select a mixed-circuit algorithm,
admit a production engine or extend approval beyond M-P. Larger qubits, new
measurements and production integration remain deferred.

## What the next decision must resolve

The [master plan](../roadmap/FTQC_SIMULATION_MASTER_PLAN.md#52-mixed-clifford--non-clifford-simulation)
requires a phase-capable decomposition candidate before FTS-10 can compare it
with existing statevector, MPS and tensor-network execution. The completed
[static pilot](FTQC_MIXED_CPU_COMPLETED_PILOT.md) has 48 complete points across
those three existing routes. It contains no coherent-decomposition timing or
capacity observation. Preserve both immutable campaigns, their original failed
warm-up and the missing per-workload outer-wall field in the continuation.
Do not manufacture a fourth candidate from those measurements.

A-D qualified finite coherent reference algebra; E-H qualified projected
references; I-L qualified current local trajectory Born inputs and continuation.
[I-L implementation #898](https://github.com/flagos-ai/FlagQuantum/pull/898)
completed actual-main admission at
`44fb9c6a6779a746780b5b96fcaaa2cb99ce7f2d`: full CI 40282631917 all 54,
required source/single/two-GPU checks 40282635505, main smoke 40282543404,
real QEC optional installation and retained evidence, fresh isolated package and
public examples. Its source-focused checks passed 118 tests with zero skips;
that developer Torch 2.13/SDK-overlay evidence is distinct from fresh core
Python 3.12.14/Torch 2.14.1. These facts close only the approved finite slice.
They do not supply a compact phase representation, general overlap kernel,
production decomposition sampler or FTS-10 method-selection result.

## Existing authorities and reuse

| Authority inspected on main 4bb6657d | Reusable capability | Boundary for this decision |
| --- | --- | --- |
| [Simulation stabilizer engine](../../flagquantum/simulation/stabilizer/engine.py) and [namespace](../../flagquantum/simulation/stabilizer/__init__.py) | Clifford sampling, Pauli readout, positioned Pauli-noise sampling, private prepared sampling and fault/DEM integration. | No declared phase-carrying pure-state or complex state-overlap interface. Pauli operator phases and a channel's Hilbert-Schmidt classification are different quantities. |
| [Stabilizer execution contract 064](API_CHANGE_PROPOSAL_064_STABILIZER_EXECUTION_MODE.md) | Existing planned sampling/readout and representation-cost reporting. | Do not add T gates, state outputs or another mode to this contract. |
| [Statevector Clifford matching](../../flagquantum/simulation/statevector/clifford_matching.py) | Existing dense Clifford fusion and cached phase/permutation mappings. | It still indexes a dense amplitude space; it is not a compact stabilizer-superposition representation. |
| [A-D independent references](../../tests/integration/test_ftqc_coherent_reference.py) | Ten static descriptions, 50 canonical scalar Clifford branch vectors, direct-T and analytic anchors. | Dense test vectors remain independent oracles, never candidate storage or a capacity claim. |
| [E-H](FTQC_DYNAMIC_COHERENT_REFERENCE_PROPOSAL.md) and [I-L](FTQC_TRAJECTORY_BORN_CONFORMANCE_PROPOSAL.md) | Independent conditional references and current trajectory checks. | No dynamic projection, sampling or reset is added to the proposed static representation slice. |

Independent QEC complete-result batching #904 and its #907/#910 delivery records
must be reused under their own approval history. Their existence neither grants
this proposal's approval nor makes them decomposition evidence. In-flight #908
concerns remote terminal-only normalization/count projection, not local overlap
algebra. Twin and physical-model work are separate contributions. No changes to
those PRs, #731, public exports, class/pickle identity, registries or original
fixtures are requested.

## Representation alternatives and proposed research choice

A signed stabilizer tableau represents a ray; coherent coefficients additionally
need the phase of the vector produced by a literal Clifford word. Attaching an
uncomputed scalar phase to a sampler does not solve its update or overlap rule.
Replaying dense gate evolution would supply a small oracle, not this candidate.

[Bravyi et al., low-rank stabilizer decompositions](https://arxiv.org/abs/1808.00128)
provide the research basis for a compact representation with explicit phase.
The [official Cirq CH-form documentation](https://quantumai.google/reference/python/cirq/StabilizerStateChForm)
shows a mature representation that retains overall phase and supplies basis
amplitudes. This supports investigating that design family; it does not establish
FlagQuantum correctness, cost or supply-chain admission. No paper performance
number or external default tolerance is adopted.

The proposed first research implementation is one private Simulation-owned
CH-form-style Clifford state/overlap prototype, derived from the mathematical
representation and documented update/overlap rules. Before code review, record
those rules and their source, bit order, conjugation convention and workspace
bounds in its internal documentation. Keep all numerical state inside Simulation;
Runtime, Compiler, Core and QEC receive no new objects. No public adapter, backend,
planner route, package skeleton or production import is added. It is a research
candidate for later selection, not FTS-12 production delivery or a fourth
cross-domain core. If a native implementation needs a different representation,
new dependency or wider primitive set, stop and revise this proposal first.
Do not copy Cirq source or introduce Cirq/Stim as a new dependency of this slice.

The proposed primitive domain is computational zero/basis preparation and
literal H, S, X and CX Clifford words on one to three qubits, plus a unit scalar
phase applied to a captured state for a discriminating checker negative. T is
expanded externally with A-D's existing coefficients, never accepted as a
Clifford update. There are no measurements, projectors, noise, gradients or
approximate states in this slice. Existing sampling-engine behavior is unchanged.

## Proposed private phase and overlap contract

The initial vector is exactly computational zero with coefficient +1. H, S, X
and CX use A-D's canonical matrices and ascending-wire computational basis.
The candidate must reproduce the complex amplitude of the literal gate word,
not independently choose a new global phase after each update. Its compact
state stores no length-2**n amplitude vector, no full unitary and no cache of
oracle answers. Polynomial representation/workspace structure is a code-review
obligation; passing n<=3 values cannot prove its asymptotic complexity.

For two normalized states, the overlap means bra-left/ket-right
`G[i,j] = <phi_i|phi_j>`, including complex phase. Require diagonal one,
Hermitian conjugacy and agreement with independently computed scalar inner
products. Magnitude-only agreement is insufficient. Rephasing state i by r
changes its bra overlaps by conjugate(r) and ket overlaps by r. In a coherent
sum its coefficient must change by 1/r. Public execution results retain their
existing global-phase-invariant contract.

For coefficients c, check the unmodified weighted norm `c^dagger G c` before
normalization. Check computational-basis amplitudes from overlaps with basis
states, sum the complex terms before squaring, and compare the resulting full
probabilities/density with A-D's independent direct-T/analytic vectors. No
normalization, dropped cross term or squared-coefficient mixture may hide error.
A vanishing coherent sum is an invalid normalized-state request, not a new ray.

## Approved decision: finite groups M-P only

The specifically approved CPU complex128 amplitudes/overlaps/densities and
float64 real probabilities/norm residuals use atol=1e-12, rtol=0. The human
explicitly admitted these representation/overlap boundaries; earlier equal
values and proposal merge did not admit them. Integer invariants remain exact.

| Group | Finite work per module verification | Independent check |
| --- | --- | --- |
| M literal states | Build exactly the 50 A-D branch descriptions once and 14 ordered basis states: 2 at n=1, 4 at n=2, 8 at n=3. At most 64 retained candidate states. | Reuse A-D's 50 scalar vectors once; basis vectors are literal unit vectors. No new circuit family. |
| N complex overlaps | Capture 562 ordered within-case branch overlaps and 288 ordered basis-bra/branch-ket overlaps once, at most 850 candidate queries. | Scalar conjugate inner products, complex values, diagonal and Hermitian checks. No cross-case Cartesian expansion. |
| O coherent reconstruction | Reuse captured values and coefficients for ten weighted norms and ten full probability/density comparisons. At most ten existing unexpanded public CPU executions, one per A-D case/anchor. | Independent direct-T/analytic answers and public phase-invariant densities/probabilities, with separate norm checks. |
| P discriminating negatives | Reuse captured values only: independent branch rephase without coefficient compensation, magnitude-only overlap, swapped conjugation, missing cross terms, invalid shape/dtype/nonfinite data, zero coherent norm and out-of-domain descriptions. | Require each corrupted value to be rejected or distinguished; no extra candidate queries or public executions. |

The counts are explicit: each n=2/3 matrix has branch counts 1,2,4,16. Its
ordered Gram queries total `2*(1+4+16+256)=554`; the two two-branch anchors add
8, yielding 562. Basis queries total `(4+8)*(1+2+4+16)=276`, plus HTH's 4 and
Bell-T-HH's 8, yielding 288. Basis-bra orientation is fixed. Reuse conjugacy for
reverse orientation instead of extra calls. Build/capture once, then reuse in
all parametrized assertions. Reuse the CPU-thread fixture and restore prior
thread settings even on failure. No shots, seed, repeated timing, benchmark
observer, process controller or new collection is included.

The state domain is bounded before construction; refuse non-integer/out-of-range
wires, unsupported instructions, unequal overlap widths and non-unit phase
factors before mutation/allocation. Never truncate a requested representation,
call dense evolution as fallback, or alter the original A-D cases to fit the
candidate. Refusals are private research errors, not new public exceptions.
Only supported finite gate words may reach the prototype. The ten public calls
are existing current execution, not a new public phase API.

## Selection gates after this slice

A passing M-P result would make the phase/overlap candidate eligible for a later
measurement proposal. It would not select it. FTS-10 still needs same-task
construction/update/overlap/reconstruction cost, term count and memory evidence,
explicit target W4 tasks and refusal limits, and a predeclared comparison rule.
Any such measurements need a new resource protocol and explicit budget approval;
never reuse the consumed 193-attempt budget or fill missing raw fields.

Only after a reviewed comparison could FTS-10 admit one primary representation
and the necessary reference path. FTS-12 production integration and FTS-14 dynamic
integration then require their own support/resource/accuracy contracts. Dense
public probability reconstruction on n<=3 does not imply scalable full outputs,
efficient sampling, arbitrary T density, or a latency/capacity benefit. No
1.5x/2x target, domestic execution, experimental data or approximation is enabled.
R04/R05, FTS-02/09/10/11 and M2 remain partial.

## Runnable current public user example

This code exercises existing exact public execution, not the proposed prototype.
Requirements: supported Python, core FlagQuantum and CPU; no optional SDK, GPU,
credentials or dataset. Save as `/tmp/ftqc_phase_overlap_example.py`.
Installed command: `python /tmp/ftqc_phase_overlap_example.py`.
From the source checkout: `PYTHONPATH=. python /tmp/ftqc_phase_overlap_example.py`.

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1).t(0).h(0).h(1)
for mode in ("statevector", "mps", "tensor_network"):
    probabilities = fq.run(
        circuit,
        outputs=fq.probabilities([0, 1]),
        options=fq.ExecutionOptions(
            mode=mode, device="cpu", precision="complex128",
            allow_approximate=False, allow_backend_fallback=False,
        ),
    ).probabilities
    assert probabilities is not None and probabilities.shape == (1, 4)
    print(mode, *[f"{float(p):.8f}" for p in probabilities[0]])
```

Expected stdout:

```text
statevector 0.42677670 0.07322330 0.07322330 0.42677670
mps 0.42677670 0.07322330 0.07322330 0.42677670
tensor_network 0.42677670 0.07322330 0.07322330 0.42677670
```

The PR must retain exact reviewed-head execution evidence for this code, its
conditions and command. It is an engineering example, not a new study point,
prototype implementation, phase contract, wheel certification or performance
measurement. Proposal merge and CI do not approve M-P implementation.

## Approved implementation rules and ownership

The human explicitly approved the finite M-P prototype and acceptance on
2026-10-26. Proposal #911 completed actual-main admission at
`53fce96e2ac3052d3f0645b11165673884fa2010`: all 54 CI jobs, required source,
single/two GPU, smoke, real optional QEC installations/evidence retention, fresh
842-member package/core and installed public examples passed. This decision does
not admit another matrix or a production integration.

The private Simulation implementation follows the CH data in Bravyi et al.,
Section 4.1, equations 42-56: binary row masks F/G/M, modulo-four gamma,
Hadamard mask v, basis mask s and complex scalar omega. Wire zero is the most
significant bit. Its C-layer fixes zero with phase +1; inverse conjugation rows
encode X-before-Z Pauli products. Left S/CX and right S/CX/CZ updates use integer
XOR and Pauli commutation signs. X is the exact H S S H identity. Hadamard uses
the two computational strings obtained by commuting X and Z through the C/H
layers, merges their difference to one pivot with C-type gates, and resolves
only that single-qubit scalar pair. That constant-size algebra never expands
the n-qubit state. No Cirq source is copied and no dependency is introduced.

A retained candidate additionally keeps its validated literal Clifford word and
an optional unit scalar phase. Overlap copies only the ket's CH masks, applies
the inverse bra word in reverse order (S inverse is three S updates), reads the
zero amplitude, then applies the conjugate bra scalar phase. For the zero
amplitude, a nonzero un-Hadamarded s bit means zero; otherwise the result is
omega times 2**(-popcount(v)/2). This preserves complex bra/ket phase without a
magnitude-only inner product or enumeration of computational states. It is a
polynomial replay implementation, not a claim of an optimal overlap kernel.

Retained storage is three n-row masks, gamma, scalar/masks and the word:
O(n**2 + g) bits/word entries apart from Python overhead. Overlap scratch has
O(n**2) integer-mask data; it replays at most three updates per bra instruction
and does not retain intermediate states or cache query answers. H uses O(n**2)
bit-level work; overall word-replay work is polynomial in n and g. These are
structural review bounds, not measured time/RSS or capacity conclusions.
The approved input words have at most 22 instructions, derived from the literal
three-qubit A-D branch. Reject a longer word, invalid width/wire/gate or non-unit
scalar before constructing state. Internal CZ is only the C-layer Hadamard
factorization rule, not a newly accepted input instruction. The accepted prototype input set
and 64 retained states/850 queries/ten executions remain as proposed above.

The finite implementation reuses the approved scalar fixtures once, captures
64 retained states and 850 successful overlap results once, and makes ten
existing public executions once per module verification. The unequal-width
negative refuses before scratch allocation and produces no overlap result.
Captured-data negatives exercise conjugation/magnitude-only/cross-term/phase
and normalization errors without further candidate queries. No production
module imports or registry entry points expose the private prototype.

Local implementation acceptance: 41 new M-P checks passed. Combined with the
existing A-D/E-H/I-L, stable dynamic and original retirement contracts, 159
checks passed with zero skips in 2.92 seconds, using developer Python 3.12.14,
Torch 2.13 and the explicit existing SDK overlay. This is not a fresh optional
installation or a benchmark. The declared minimum Ruff 0.15.0, Black 24.1.0 and
Mypy 2.0.0 passed on both changed modules (Mypy Python 3.12); this is targeted,
not whole-version typing certification. Simulation/Integration ownership,
architecture, English, links, source-of-truth and tracked hygiene passed.
Initial validation called the existing two-argument Circuit.gate incorrectly;
correcting only the test call completed all ten public executions. Initial
complex-scalar test assignments were made explicit typed tensors for Mypy;
no production gate, checker, threshold, fixture or skip was changed. Full
exact-head CI and required GPUs subsequently passed. Actual main
`82143b09dea21bf565c51a2e7f3f3c1496ec9dbf` completed all 54 CI jobs, required
source/single/two-GPU checks, main smoke, real optional QEC installations and
installed-evidence retention. Fresh isolated sdist/wheel membership matched all
845 Python members; Python 3.12.14/Torch 2.14.1 core checks and the installed
three-route public example passed outside the checkout without PYTHONPATH or
an SDK overlay. Actual-main focused contracts passed the same 159 checks with
zero skips on the developer profile. Records are `pr922-main-delivery.json`,
`pr922-main-install-acceptance.json` and `pr922-main-public-example.json`.

This closes finite M-P correctness only. The separate
[cost-study proposal](FTQC_PHASE_OVERLAP_COST_STUDY_PROPOSAL.md) requires a new
explicit observation budget; no measurement, wider domain or production
integration is admitted by this acceptance.
