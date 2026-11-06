# Modular inverse arithmetic and scaling layers

The compiler modules port unchanged `_KaliskiIterationStep5/6` in Qualtran v0.7.0
[`mod_division.py`](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py),
commit `bcb17bb25360ed68ce3bc084645db3e929799389`. Google copyright and Apache-2.0
attribution are retained. They emit IR through shared primitives; no runtime,
device policy or stable public API is introduced. They are iteration layers,
not a complete modular inverse.

## Arithmetic interface and domains

`inverse_arithmetic_ir` takes four equal-width positive big-endian words, branch
and active flags, `2n` clean work wires and `2n-1` reusable classical slots.
It emits the original And(active,branch;1,0), BitwiseNot(v), CAdd(u,v),
BitwiseNot(v), CAdd(r,s), And erase sequence. The loaded-word/carry pool is reused
between additions; the outer control lives until both complete. There are
`4n-1` measurements. The source adjoint uses original CAdd adjoints and reverses
the two stages, preserving the inactive branch's original conjugation gates.

The gate program performs `v-=u`, `s+=r` modulo `2**n` when active and not
branch. The source high-level classical model omits wrapping and is compared
on its no-underflow/no-overflow domain. A direct unmodified-source witness at
n=2 gives classical v=-1,s=4 while original primitive gates and native IR give
v=3,s=0. This is an explicit domain distinction, not a patched reference. The
one-bit source Clifford model is -1 (an inherited Add cost exception); measured
Clifford counts are never inferred from that estimate.

## Scaling interface and physical output roles

`inverse_scaling_ir` takes four words, branch/parity/marker/active, fitting positive
odd modulus, `2n+5` clean work wires and `n+1` classical slots. It retains:

1. marker and parity XORs into branch;
2. active-controlled cyclic right rotation of v using original exact Fredkins;
3. original ModDbl of r;
4. parity-controlled exchange of u/v and r/s;
5. negative-control s-LSB XOR into parity.

The inverse reverses original components and consumes the complete actual
forward output. `inverse_scaling_outputs` is the layer's physical output contract,
consumed by the IR emitter and subsequent composition: coefficient becomes
`(*r[1:], work[0])`, while clean work becomes `(old_r[0], *work[1:])`.
The inverse returns the original roles through shared modular-half output
resolution. No physical SWAP is added to conceal those aliases. All remaining
word and flag wire tuples retain their physical identities.

Forward coefficient r is a valid residue. For agreement with the source
classical division, v must be even when active: the **gate** extension on odd
v is a cyclic rotation, while high-level classical v>>1 discards the LSB. This
boundary is tested directly. Outside the residue promise, n2/p1/r2 gives
high-level r0, child-classical r2, and original primitive/native r3. These three
contracts are preserved and distinguished. No universal off-domain numeric
or clean-work guarantee is claimed.

## Original graph and channel validation

The shared original-Soquet reader binds And/CAdd/BitwiseNot or ModDbl/CSwap/
TwoBitCSwap/basic-gate leaves. Atomic source And is exported through its original
Cirq generator. Exact source TwoBitCSwap uses its original Clifford+T circuit.
Split/Join bookkeepers call their original as_cirq_op, verify no emitted operation
and check ordered physical wire identities;
widths and all desired output bindings are checked. Reference algorithms and
strict primitive dependency/operand/feedback assertions are unchanged.

Forward and adjoint source streams, physical ports, QEC costs and peak qubits
are compared through eight bits. Complete small arithmetic input columns and
all erasure histories are certified through n3; n1/n2 additionally run every
raw 8/128 measurement history on full complex control/word/spectator states.
Scaling certifies every u/v/s/flag input with r a valid residue for n<=3 and
named larger boundaries, including s=p and the cyclic extension on odd active v. The other coefficient s
is an arbitrary n-bit word; reachable full-iteration states include s=p.
Full complex/spectator states run every raw 8/32 history for n1/n2, checking
unnormalized probability, phase, returned-work zeros and original inverse
restoration. The original classical model is independently checked on its
even/residue domain. Larger histories use a compositional primitive certificate,
not an unreported dense execution of all trajectories.

| Eight-bit layer | Measured T | CX | X / H | Measurements / resets | Clean work | Total qubits | Classical slots |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Arithmetic | 124 | 264 | 20 / 93 | 31 / 31 | 16 | 50 | 15 |
| Scaling, p=251 | 229 | 400 | 12 / 97 | 17 / 17 | 21 | 57 | 9 |

Separately, source arithmetic QEC estimates 124 T/31 measurements and QubitCount
50. Scaling source **default four-T CSwap** estimates 160 T; **exact seven-T
CSwap** estimates 229 T, matching literal decomposition. Both scaling models
have 17 measurements and QubitCount 57. Primitive and logical Clifford counts
are separate; a default model number is not a measured implementation.

Run the native user example without NumPy or Qualtran:

```bash
python -m examples.algorithms.register_inverse_updates
```

It uses `circuit.append(op)` for complete one-bit conditional ring arithmetic,
then a nontrivial modulo-three scaling superposition and original inverses.
Scaling explicitly checks returned physical work roles. That example uses 21
qubits, 32 MiB per state and one trajectory, and prints measured primitives and
source/default/exact costs separately. It does not claim all histories from a
single trajectory; exhaustive raw-branch evidence is in the tests above.

Local acceptance: 51 new tests and 417 directly affected dependency regressions
pass without skips, including the full s word domain (s=p included). The two
production modules have 100% statement coverage. After the source reader was
strengthened to invoke original Split/Join lowering on actual Cirq qubits,
130 final source/consumer cases pass (the two expensive dense branch cases were
already covered by the concentrated run); that reader independently has 100%
coverage. Strict mypy passes on 968 production files and separately on the
example and reader. Minimum/current Black, Ruff, scope, architecture and
multi-level IR checks pass. Five original source Git blobs match the immutable
inventory. The native no-NumPy example completes its nontrivial modulo-three
forward/inverse check with explicit returned-work ownership.
