# Modular addition and subtraction with exact source adjoints

The executable reference is Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`bloqs/mod_arithmetic/mod_addition.py:ModAdd` and
`bloqs/mod_arithmetic/mod_subtraction.py:ModSub`.
Google copyright and Apache-2.0 notices are retained. No algorithm or reference
patch is used. These concrete compiler emitters compose existing register
primitives; no new executor, runtime allocator or public root API is introduced.

## Ownership and call chains

`compiler/registers/modular_addition.py` owns `modular_add_ir` and the shared
validated workspace layout. With equal big-endian n-bit operands it implements
`(x,y) -> (x,(x+y)%p)` on the promised domain `0 <= x <= p`, `0 <= y < p`,
where `1 <= p < 2**n`. The x=p extension is needed by ModSub when its original
left operand is zero. Quantum input amplitudes must have no support outside
the promised domain; the compiler does not inspect unknown quantum values.

The unmodified component order is extended Add, AddK(-p), sign-controlled
AddK(p) on the low n bits, unsigned LinearDepthGreaterThan, then X(sign).
All components retain their original measured And cleanup and gate ordering.

`compiler/registers/modular_subtraction.py` owns `modular_subtract_ir` on
`0 <= x,y < p`. It preserves the original sequence:
BitwiseNot(x), AddK(p+1), ModAdd, **AddK(p+1).adjoint()**, BitwiseNot(x).
The newly shared literal constant-adder adjoint is used directly; relowering
AddK(-p-1) is not a gate-equivalent replacement. The source load requires
p+1 < 2**n, so the all-ones modulus is rejected by this concrete entry.
Unsigned and Montgomery bit representations are checked separately upstream.

## Explicit workspace lifetimes

`modular_add_requirements(n)` returns `(2*n+2, n)`: clean quantum work and
reused classical slots, for either modular operation. The workspace is:

- w[0]: extended-left zero prefix in the first Add, then the constant MSB.
- w[1]: persistent sign/correction bit.
- w[2:n+2]: remaining n constant wires.
- w[n+2:]: n carry wires.

The first Add preserves its extended left input, hence the zero prefix is dead
and zero before its reuse as the constant MSB. Every constant/carry phase clears
its work before the next phase. Comparison reuses the full constant tuple and
the last carry (the one-bit comparison uses just the first constant wire).
This preserves each measured physical work wire's classical-slot binding across
roles. All 2n+2 work wires end in zero on the promised domains.

Qualtran's QubitCount model and the native bound are compared on the same basis:
including 2n operand wires, both are **4n+2** for the tested sources. This is a
comparison with that model, not a surface-code hardware-qubit or universal
minimum-allocation claim. Classical slots retain their most recent outcomes;
measurement event counts include all overwritten events.

## Runnable user example

```sh
python -m examples.algorithms.register_modular_arithmetic
```

The user writes `circuit.append(ModularArithmetic("add"))` or the subtraction
operation. The three-bit modulo-three example prepares all nine valid residue
pairs with distinct complex amplitudes, checks both operations and eight work
wires returning to zero, then checks addition followed by exact source modular
subtraction restoring the state. It uses eight trajectories, not an exhaustive
claim for that example's measurement outcomes. No reference package or NumPy
is required at runtime.

## Measured primitive scans versus independent source models

| Path | n, p | Measured CNOT | Measured T/TDG | Source QEC T | Events / source measurements | Quantum work | Native total / source QubitCount | Reused classical slots |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ModAdd | 1, 1 | 28 | 12 | 12 | 3 / 3 | 4 | 6 / 6 | 1 |
| ModAdd | 3, 3 | 126 | 44 | 44 | 11 / 11 | 8 | 14 / 14 | 3 |
| ModSub | 3, 3 | 168 | 60 | 60 | 15 / 15 | 8 | 14 / 14 | 3 |
| ModAdd | 8, 251 | 376 | 124 | 124 | 31 / 31 | 18 | 34 / 34 | 8 |
| ModSub | 8, 251 | 538 | 180 | 180 | 45 / 45 | 18 | 34 / 34 | 8 |

Exact T/measurement formulas are 16n-4 / 4n-1 for ModAdd and 24n-12 / 6n-3
for supported ModSub. Source prose's approximate Toffoli summaries do not replace
its executable QEC model. Expanded X/H/S/CNOT/T/TDG, measurement, conditional CZ
and reset counts are scanned independently. Raw Clifford estimates, including
the pre-existing one-bit Add model anomaly, are not equated to primitive totals.

## Source boundaries retained as evidence

The source ModAdd classical action accepts y=p, but its executable decomposition
cannot clean its sign there. The literal example n=2,p=3,x=0,y=3 yields classical
y=0 while decomposition fails its Free check. Primitive-IR postselection also
checks that the sign is one, rather than patching it to zero. That target domain
is not advertised by the native clean-work contract; x=p with y<p is verified.

Source ModSub cannot load p+1 when p=2**n-1, despite its classical subtraction
being defined. Widths 1,2,3 retain the original ValueError from source flattening;
the native emitter rejects that boundary explicitly. There is no silent fallback
to a different modular-subtraction network.

## Verification scope

There are 34 new cases. The add suite has eight literal source width/modulus
pairs through eight bits, four complete small coherent-domain cases with an
entangled spectator and every measurement event, validation and the original
y=p counterexample. The subtraction suite has twelve source width/dtype pairs,
two complete small all-event cases plus forward restoration, one nonbinary
modulus case with complete coherent amplitudes in three selected branches,
three source load-boundary cases, exact-adjoint composition and the user example.

Per-wire primitive dependencies, operands, feedback and source output identities
match separately from resource models. Source temporary allocations are bound by
component lifetime without changing source gates. On widths one/two, independent
events are enumerated even when slots are overwritten; each unnormalized branch
equals the ideal complex state divided by sqrt(2**events), checking branch
probabilities, phase and all work. Subtraction and subsequent forward restoration
are each checked for every small branch; the nonbinary three-bit case explicitly
does not claim enumeration of all 32,768 outcomes.

Controlled modular variants, symbolic widths/moduli, other out-of-domain states
and final integration CI/GPU/review remain open.

Batch acceptance: all 34 new and 1,182 concentrated regression cases pass,
including pinned Qualtran/Kickmix references, with no skips. Both modular
modules have 100% statement coverage. Strict mypy passes on 946 source files;
minimum/current Black, Ruff, architecture and multi-level IR checks pass locally.
