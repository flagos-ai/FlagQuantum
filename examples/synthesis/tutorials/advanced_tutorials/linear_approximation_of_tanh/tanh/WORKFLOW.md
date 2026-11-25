# Coherent piecewise-linear tanh approximation

Run from the repository root:

```sh
python -m examples.synthesis.tutorials.advanced_tutorials.linear_approximation_of_tanh.tanh.main
python -m examples.synthesis.tutorials.advanced_tutorials.linear_approximation_of_tanh.tanh.comparisons.circuit
```

`main.py` prepares all sixteen inputs coherently and computes four linear
segments using a Python conditional expression inside a reversible XOR update.
`Annotated[QNum, (4, False, 4)]` declares an unsigned four-bit number with four
fractional bits. The program uses named registers without physical wire numbers.
The conditions select quantum branches coherently; the Python function body is
captured rather than executed as a classical function.

The slopes are `(1, 15/16, 13/16, 5/8)` and intercepts are
`(0, 1/32, 1/16, 7/32)`. Segment boundaries are `1/4`, `1/2` and `3/4`.
For raw input `x`, select segment `x >> 2`, evaluate exactly at `x/16`,
multiply by sixteen, floor toward negative infinity and mask to four bits.
The update XORs these bits into the existing output. This is a quantized
piecewise approximation, not an exact mathematical tanh evaluation.

```python
import flagquantum as fq
from examples.synthesis.tutorials.advanced_tutorials.linear_approximation_of_tanh.tanh.main import build_program

program = build_program(initial=5)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
print(program.report["resources"])
```

The default zero-target program uses ten physical qubits, including two clean
work qubits, sixty operations and depth forty-eight. Repeating the XOR update
twice restores any initial output. Integer-literal `^= 5` remains a raw bit mask;
`^= 5/16` expresses the fixed-point value with the same stored bits here.

Acceptance compares original, native and independent Circuit full complex
statevectors, including absolute phase, logical bit order and every work
amplitude. Tests cover output words 0/1/5/10/15, two additional nonzero models,
negative coefficients, segment boundaries, inverse updates and budget refusals.
The independent Circuit constructs its permutation directly from rational
arithmetic and does not call synthesis quantization or truth-table helpers.
Only exact dyadic coefficients and bounded unsigned scalar profiles are admitted;
unsupported precision, signed words and overlapping reads/targets refuse.
