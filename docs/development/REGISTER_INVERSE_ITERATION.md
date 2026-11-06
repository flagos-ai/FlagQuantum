# Original modular inverse iteration

`compiler/registers/inverse_iteration.py` composes the six shared register
layers in the unchanged Qualtran v0.7.0 `_KaliskiIteration` order. It owns
composition and physical register roles, not arithmetic kernels or execution.
Source: [pinned modular division](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py),
Apache-2.0, Google LLC; copyright and provenance remain in the adapted module.

## Interface and ownership

`inverse_iteration_ir(u, v, r, s, marker, active, terminal, modulus, workspace,
classical_bits, adjoint=False)` accepts four equal-width big-endian words with
n >= 2, a fitting positive odd modulus, 2n+7 clean work wires and
max(2n-1,n+1) reusable classical slots. The work pool starts with the source's
internal parity `a` and branch `b`, followed by the shared 2n+5-wire scratch
pool. All six stages reuse scratch only at their clean boundaries.

The coefficient output physically changes to `(*r[1:], workspace[2])`.
`inverse_iteration_outputs` returns that word and the clean work roles,
`(workspace[0], workspace[1], old_r[0], *workspace[3:])`. Consumers and the exact
inverse use those roles. There are no materializing SWAPs or logical resets.
The reverse starts with the original scaling adjoint, then reverses arithmetic,
exchange, ordering, parity and termination with their exact measured adjoints.

The forward clean-work guarantee is for algorithm-reachable inputs, with fresh
zero marker and terminal bits. The inverse takes the entire forward image.
The source's high-level signature alone does not encode all these promises.
For example, n=2, p=3, `(u,v,r,s,m,f,t)=(1,0,0,0,0,0,0)` leaves internal `a=1`
in the literal circuit, despite the top-level classical method returning r=0.
This is preserved and tested; `Free` is not translated into a reset that hides
invalid inputs. Reachable terminal states with s=p remain supported.

## Source and channel evidence

The test oracle walks the unmodified whole component graph. It identifies the
internal a/b allocations by their original edges into Step2 (or reverse Step6),
not the incidental ordering of independent allocation nodes. Each stage uses its
existing original-source reader, and every quantum operand, output alias and
classical feedback slot is remapped explicitly. Full forward and adjoint streams
and ports match at n=2,3,5,8.

Reachable traces start from every x in [0,p) at `(u,v,r,s,f)=(p,x,0,1,1)` and
run 2n original iterations with fresh history outputs. Tests certify the full
complex linear channel for every erasure history on all those states:

| n, modulus | Distinct reachable input states | Measurement events per iteration |
| --- | ---: | ---: |
| 2, 3 | 7 | 21 |
| 3, 5 | 15 | 29 |
| 3, 7 | Checked over all inputs | 29 |
| 4, 13 | Checked over all inputs | 37 |
| 8, 251 | 1,557 | 69 |

The independent compositional certificate accepts only literal proven primitive
And/erase and exact CSwap blocks plus X/CX. It checks each zero-target/erasure
invariant and all work bits, establishing the same map divided by
`sqrt(2**events)` for every complete history. The reverse restores all columns
and original roles. This is not a dense enumeration of 2**69 histories.

Raw 22-qubit complex statevectors additionally check all seven n=2 reachable
states in one superposition for three complete histories: all-zero, all-one and
alternating outcomes, both forward and reverse. These are sampled histories,
not an exhaustive raw branch claim. The existing individual stages have their
own smaller exhaustive raw branch tests.

## Measured gates and independent estimates

Both directions have 8n+5 measurements and resets. T counts below include T and
T-dagger; the default source model weights CSwap at four T, whereas the unchanged
literal source decomposition uses seven T. Those are different cost assumptions.

| n, modulus | Measured T | Measured CX | Measured X | Measured H | Measured M/reset | Work | Total qubits | Classical slots | Default model T | Exact-swap model T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2, 3 | 147 | 271 | 60 | 81 | 21 | 11 | 22 | 3 | 120 | 147 |
| 3, 5 | 214 | 393 | 74 | 115 | 29 | 13 | 28 | 5 | 172 | 214 |
| 8, 251 | 549 | 1,013 | 134 | 285 | 69 | 23 | 58 | 15 | 432 | 549 |

There are also one S and one conditioned CZ per measured erase. The independent
source models give default T=52n+16, exact-swap T=67n+13, M=8n+5 and Q=6n+10.
The explicit clean-pool allocation matches source peak qubits. Model values are
never labeled as measured primitive streams.

## Local acceptance

The batch adds 23 tests; 440 directly affected dependency regressions pass
without skips. The iteration module, scaling dependency and shared source reader
have 100% statement coverage. Strict types pass on 969 production files and the
example/reader; minimum and current Black, Ruff, compiler scope, architecture and
multi-level IR checks pass. The pinned source blob is
`947f40eb5d17101ad91ff4d4432ac5287f867bd5`, unchanged from the source ledger.
This scoped regression does not claim a rerun of every historical G2 test.

## Runnable native path

```bash
python -m examples.algorithms.register_inverse_iteration
```

The user writes `circuit.append(InverseIteration())`, followed optionally by its
adjoint. The example uses 22 qubits (64 MiB per state) and one trajectory for
each check, comparing complete complex vectors on a superposition that includes
zero-input termination and both arithmetic branches. It runs without NumPy or
Qualtran installed and prints measured gates and independent model assumptions.

This delivers a complete single iteration. The complete multi-iteration inverse,
its scaling convention and initialization/final cleanup remain separate G2 work.
