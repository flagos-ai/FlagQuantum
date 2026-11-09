# Locked demolition Fourier measurement

`qft_measurement_operations` ports `gen_qft_demolition_measure.cc`
from Kickmix commit `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.
The unchanged source blob is `59c942ad6d1a4e40e3776166afb99220bbf93f6b`.

## Register contract and source decomposition

Target and classical output tuples are equally wide big-endian words. Quantum
ports are distinct and output slots are distinct nonnegative integers. The
source's low-first arrays are obtained by reversing both tuples. It first
swaps the source word ends, then visits each source qubit from low to high.
Every previous output controls one exact `Z_POW` of half-turn exponent
`2**(previous-current)`, followed by HMR into the next output slot. Inverse
mode negates those exponents; it selects the opposite Fourier measurement
basis, not an inverse of an already destructive channel.

Every measured target qubit ends in zero. The operation allocates no extra
quantum work and no extra classical feedback slots. Logical output slots are
also the feedback slots. `qft_measurement_program` embeds the sequence
in a caller-supplied positive ambient width and marks holes between output
slots as unowned classical inputs. Binding preserves those holes and
initializes only the selected outputs. The caller may also apply the existing
classical offset at binding.

An empty logical word returns an empty sequence and embeds in an existing
positive-width circuit as an identity on its spectators. This preserves the
source's zero-arity operation without changing the protected Core circuit
width contract. Ambient spectator qubits are not algorithmic workspace.

The source's 128-bit full-turn grid makes phase terms with exponent at most
-128 zero. These `Z_POW` nodes are retained, as the original builder emits them.
The raw exact angle survives KMX round trips. Core `phase(theta)` is created
only at the explicit centered binary64 binding boundary.

## Gate-flow and branch acceptance

A native C++ oracle compiled against the unchanged source matches 176 entire
primitive streams. Logical widths range from zero through 129, with both
directions, strided physical port indices, and classical offsets.
Only source declarations/tags and the compiler metadata header are excluded.
No phase, measurement or zero-angle instruction is optimized away.

For logical widths zero through four, every raw measurement history is
postselected from the actual bound primitive IR. The reference is the independent
dense Fourier projection on the input target, leaving an entangled spectator:

`branch_y[s] = sum_x exp(sign*2*pi*i*x*y/2**n) * psi[x,s] / sqrt(2**n)`.

The complete unnormalized complex branch vector, including global phase, agrees.
Target qubits are zero and branch probabilities sum to one. Contiguous and
nonmonotonic/interleaved layouts are tested. The shared branch helper delegates
classical XOR assignments to Core's existing evaluator before postselection.

Local trajectory and batched execution additionally match each observed
conditional spectator state and the Born probabilities. Preexisting classical
holes remain unchanged. The native compressed simulator independently confirms
exact Fraction-valued phases on basis inputs through 128 bits, with randomized
large input words. Large-word checks are source-phase and gate-flow evidence;
they do not claim a 128-qubit dense statevector run.

## Primitive resources

| Logical word bits | Source SWAP | Source Z_POW | Source HMR | Exact zero-angle Z_POW |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 0 |
| 3 | 1 | 3 | 3 | 0 |
| 4 | 2 | 6 | 4 | 0 |
| 128 | 64 | 8128 | 128 | 0 |
| 129 | 64 | 8256 | 129 | 1 |
| 130 | 65 | 8385 | 130 | 3 |

The bound four-bit example has two SWAP, six conditional `phase`, four H,
four measurements and four conditional X, plus four classical output
initializations. There are no quantum ancillas or classical snapshot slots.
These are emitted primitive counts, not executed branch counts or synthesized
T counts. The source routine supplies no independent T-resource model; phase
precision and a synthesis model must be specified before quoting such a cost.

## Runnable user example

```sh
python -m examples.algorithms.fourier_measurement --shots 64
python -m examples.algorithms.fourier_measurement --inverse-qft --shots 64
```

Each run prepares Fourier word three and uses the existing
`circuit.append(operation)` interface. It reports `{3: 64}` and confirms the
measured target is zero. It requires FlagQuantum and PyTorch, with no Qualtran
or Kickmix runtime dependency.

The compiler primitive, codec and numerical binding remain separate authorities.
No stable public API or Core instruction schema changes are introduced.

Local acceptance: 64 new tests; 1,279 concentrated regressions pass with four
optional backend skips. The regression includes existing controlled add/subtract
consumers of the shared branch projector. Strict typing, formatting, scope and
architecture checks pass. Both runnable example directions return word three.
