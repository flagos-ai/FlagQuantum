# iQuHack state-preparation workshop

Open [walkthrough.ipynb](walkthrough.ipynb) or execute the workshop locally.

```bash
python -m examples.synthesis.community.hackathons.iquhack_2025.workshop.overview.main
```

The runnable reference exercises retain all 64 terms of the coherent
arithmetic example, the seeded 32-entry probability array and the five-bit
exponential warm-up. The signed exercise computes `5 + (-7.5) = -2.5`.
It uses a 14-qubit Circuit with sign extension and controlled increments
in units of one half. The visible native qfunc now also runs at that fourteen-bit
width. The Circuit exercise checks all 256 input pairs in one coherent witness.

The November 21, 2026 native preflight retains the same input formats:
`QNum[3]`, `QNum[5, True, 1]` and result `QNum[6, True, 1]`. Immutable input-bit
borrowing first reduced retained lowering from 36 work bits / 286 instructions
to 25 / 264, with recomputation changing from 25 / 304 to 25 / 264. Those
historical measurements remain unchanged.

The earlier equal-width modular-sum fusion emitted the terminal XOR result directly,
while read-only sign aliases avoid copying an extended signed word. Both
schedules then used **11 work bits and 49 arithmetic instructions**. Preparation
added four instructions: **25 total qubits, 53 instructions and depth 24**.
Those are historical compile-only measurements. Subsequent immutable shift
borrowing reduced the default to **20 total qubits, six work bits, 47 instructions
and depth 23**; that fewer-instruction default remains unchanged.

With `max_qubits=14, max_workspace=0`, a bounded Fourier XOR writer instead
borrows and restores a root input word and uses the arbitrary output high bit
to retain overflow. The Notebook executes this **14-qubit, zero-extra-work,
89-instruction, depth-51** route, including the four preparation instructions.
This is a width/operation trade-off, not a runtime or hardware-gate advantage.
The current route accepts native logical targets; basis conversion and routing
refuse until absolute-phase preservation is admitted for those targets.

```python
import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, qfunc


@qfunc
def signed_addition(
    x: Output[QNum[3]],
    y: Output[QNum[5, True, 1]],
    result: Output[QNum[6, True, 1]],
) -> None:
    x |= 5
    y |= -7.5
    result |= x + y


program = synth.synthesize(
    signed_addition,
    max_qubits=14,
    max_workspace=0,
    max_operations=89,
    memory_limit_bytes=1024**2,
)
result = fq.run(program)
print(dict(program.numeric_probabilities(result, registers=("x", "y", "result"))))
# The sole outcome is (5, -7.5, -2.5), with probability approximately one.
```

Compiler checks cover all 256 inputs and 64 arbitrary XOR targets, inverses,
shared negatives, unequal moduli, and smaller complete coherent witnesses with
absolute phase and clean workspace. This is a general modular arithmetic rule,
not an optimization keyed to this Notebook or these constants. The work-free
writer additionally passes all 16,384 original complex unitary columns, two
complete random complex states, smaller signed/unsigned and shifted sums, and
a native compute/phase/uncompute program with clean local storage. Exact QFT
angles are retained; no angle truncation or clean-target assumption is used.

The amplitude exercise uses native indexed rotations to load `x²` on the
indicator, with `x = word / 32`. It draws the 2,048 shots observed in the
source transcript and derives both joint and marginal counts from them.
The original scalar parameters marked `a` and `b` are unused by this
exercise. The warm-up applies the positive-sign QFT to a probability grid
that decays as `exp(-0.1 * word)` and executes one seeded bonus draw.
All local checks compare complete complex states with independent
arithmetic, probability or Fourier references.
