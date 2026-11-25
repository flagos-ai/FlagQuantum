# Molecular QPE with power-specific product formulas

Run from the repository root:

```sh
python -m examples.synthesis.applications.chemistry.qpe_for_molecules.main
python -m examples.synthesis.applications.chemistry.qpe_for_molecules.comparisons.circuit
```

The native program retains the six-state-bit LiH model's 49 ordered Pauli terms,
including its identity coefficient, and seven phase bits. For p=2**i the source
uses time=-2*pi*p and ceil(0.05*p**1.5) second-order repetitions, specifically
[1,1,1,2,4,10,26]. Repeating a fixed approximate one-step unitary is not equivalent
to these separately specialized formulas. Input state preparation is explicit;
the default remains the original all-zero state.

A generic classical partition groups only consecutive mutually commuting terms.
The retained groups have sizes [10,4,4,6,6,1,7,11]. Each Strang center combines two
half steps of the final group, and neighboring repetitions combine the first
group's half steps. This is exact regrouping of the source formula and preserves
all noncommuting order. It fits the existing 4096-effect construction limit
without raising a cap, changing production code or declaring a new optimizer.

The phase output is a raw seven-bit array. Interpret its unsigned word w as
(w if w<64 else w-128)/128, preserving the source's signed seven-fractional-bit
format. The executable prints that decoded phase distribution. State bits and
phase bits remain separate named outputs; no physical wire decoding is required
for the native quantum program.

Independent Circuit validation expands the ungrouped original Strang schedule
using Pauli basis changes, parity CNOT chains and controlled RZ gates. Identity
terms apply conditional scalar phases. An analytic inverse Fourier matrix
checks the QFT convention independently. Acceptance compares all 8,192 complex
amplitudes against the frozen source and native program, adds a nonzero signed
input and changed time scale, and tests coherent controls, global-phase
interference, inverse restoration, model/group/step identity, alias rejection
and budgets. Compiler workspace is zero.

Native default resources are 13 physical bits, 17,957 operations and depth
16,199; frozen source resources are 13 bits, 16,931 operations and depth 15,488.
The source performs additional commuting compaction; remaining native overhead
is an explicit optimization gap. This verifies the quantum program rather than
the complete Notebook workflow or a performance advantage.
