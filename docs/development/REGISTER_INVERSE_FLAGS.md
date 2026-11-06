# Modular inverse flag layers

These compiler components port the unchanged Qualtran v0.7.0
[`_KaliskiIterationStep1` and `_KaliskiIterationStep2`](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py),
commit `bcb17bb25360ed68ce3bc084645db3e929799389`.
Google copyright and Apache-2.0 attribution are retained in each module.
They emit native IR through the existing shared multi-control/And primitives;
they introduce no simulator, scheduler, hidden register allocation or stable API.
This is a pair of register primitives, not a completed modular inverse.

## Termination ownership and interface

`inverse_termination.py::inverse_termination_ir` accepts a big-endian word `v`
and logical flag wires `marker`, `active`, `terminal`. Width `n>=2` requires
`n` caller-owned clean work wires and `n` distinct classical slots. The original
program updates marker by `active & (v==0)`, then XORs the updated marker into
active and terminal. It preserves `v` and returns every work bit to zero.
Terminal is logical output, not disposable workspace. The original classical
contract requires terminal initially zero; the reversible gate extension on a
dirty terminal is separately tested, not claimed as source classical support.

The adjoint restores the two flag updates before applying the original clean
multi-control detection again. It reverses abstract components before expansion,
not the dynamic primitive stream. The source one-bit entry constructs a
prohibited two-control MultiAnd and fails before decomposition. The native
source-parity entry rejects width one explicitly; the reference is not patched.

## Parity ownership and interface

`inverse_parity.py::inverse_parity_ir` preserves two equal-width positive
big-endian words `u`, `v` and updates logical flags in original order:

1. `parity ^= active & (u_LSB==0)`.
2. `marker ^= active & (v_LSB==0) & (parity==0)`, using the updated parity.
3. `branch ^= parity ^ marker`, using the updated flags.

The least significant bit is the last physical word wire. All logical flags
may be arbitrary coherent inputs. Two clean work wires and two classical slots
are reused at the clean boundary between the two multi-controlled Xs. Three
measurements occur; two slots do not imply two measurements. The original
adjoint restores branch, then marker, then parity, with the same clean work pool.

## Source and channel checks

Literal source forward/adjoint streams, operands, output ports and feedback are
compared at termination widths 2/3/5/8 and parity widths 1/2/3/5/8. Source
QEC and peak QubitCount are evaluated independently. Termination widths 2/3/5/8
and parity widths 1/2/3/5 check every logical input and all histories through a
compositional primitive phase certificate. Small widths through three additionally
execute every raw measurement branch on complete complex superpositions with
an entangled spectator, retaining probabilities and phases, and reverse exactly.
The original terminal-zero classical promise and width-one failure are separate
from those complete reversible gate-column checks.

| Eight-bit layer | Measured T | Measured CX | Measured X | Measurements / resets | Clean work | Total qubits | Classical slots |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Termination | 32 | 51 | 32 | 8 / 8 | 8 | 19 | 8 |
| Parity/branch | 12 | 22 | 12 | 3 / 3 | 2 | 22 | 2 |

Separately, source QEC estimates 32/12 T and 8/3 measurements, and QubitCount
estimates 19/22 qubits. Source logical component counts and expanded CX/H/S/CZ
counts are distinct. The first layer uses an eight-And prefix chain and three
logical CNOTs; the second uses three Ands and four logical CNOTs. No classical
flag or retained logical output is counted as cleaned quantum work.

Run the native append workflow without NumPy or Qualtran:

```bash
python -m examples.algorithms.register_inverse_flags
```

`circuit.append(InverseFlags())` composes the two original two-bit layers with
one shared two-work/two-slot pool, checks complete complex amplitudes and logical
outputs, then appends the original reverse layers to restore the input. It
reports measured primitives separately from 20 T/five measurements/eleven peak
qubits in the source component model.

Local acceptance: 37 new and 2,140 combined regressions pass without skips;
both production modules have 100% statement coverage. Strict mypy passes on
964 production files and the example. Minimum/current Black, Ruff, scope,
architecture and multi-level IR checks pass. The original mod_division.py Git
blob matches the immutable inventory. The native example also runs without
NumPy or Qualtran installed.
