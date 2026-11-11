# Dense unitary numerical inspection

Simulation owns these two independent NumPy numerical routines under
`flagquantum/simulation/numerics/`. Compiler does not call them to synthesize
instructions. They consume caller-supplied dense arrays and do not import
Qualtran, Cirq, Runtime, or Compiler. No public exports or protected APIs change.

## Runnable local path

```python
import numpy as np
import torch
import flagquantum as fq
from flagquantum.simulation.numerics.unitary_polynomial import (
    evaluate_polynomial_of_unitary_matrix,
)
from flagquantum.simulation.numerics.unitary_distance import (
    unitary_distance_ignoring_global_phase,
)

circuit = fq.Circuit(1, dtype=torch.complex128).h(0)
state = circuit.state().detach().numpy().reshape(-1)
hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
result = evaluate_polynomial_of_unitary_matrix((0, 0, 1), hadamard)
np.testing.assert_allclose(result, np.eye(2), atol=1e-14)
np.testing.assert_allclose(hadamard[:, 0], state, atol=1e-14)
assert unitary_distance_ignoring_global_phase(hadamard, -hadamard) < 1e-7
assert not np.allclose(hadamard, -hadamard)
```

Normal operation insertion remains `circuit.append(op)`; the numerical functions
postprocess dense inspection data. The append regression uses an existing bound
operation with H and Rz, checks the original instruction stream and an idle clean
wire, then compares full complex polynomial data to the source. It does not
pretend the polynomial itself is an appendable circuit or a normalized state.

## Source semantics

Both functions adapt unchanged Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, preserving source notices.

| Owner | Source | Retained semantics |
| --- | --- | --- |
| `unitary_polynomial.py` | `linalg/polynomial/basic.py` | Ascending coefficients; square-shape assertion; integer matrix power; negative offset uses adjoint powers; sequential accumulation into the input dtype; empty coefficients yield zero |
| `unitary_distance.py` | `linalg/matrix.py` distance function only | Same-shape then square-shape ValueError; trace overlap; clamp squared distance to zero before square root |

Negative powers assume unitarity. For a nonunitary input the source uses an
adjoint rather than an inverse; the native function does the same. Neither
routine validates unitarity, promotes real storage to accommodate complex
coefficients, or mutates input arrays. Native typing covers float32/float64/complex64/complex128 arrays;
NumPy still determines operation and casting support. Accumulation uses
`np.add(..., out=result)`, the same in-place ufunc as source `+=`, so real
storage still rejects complex additions. The distance's NumPy
scalar is converted to a Python float without changing its numerical value.

The distance intentionally removes global phase. It must not replace exact
complex-amplitude checks: U and exp(i phi) U can have zero distance while their
controlled forms differ. Tests explicitly retain that counterexample. Empty
matrix/nonfinite numerical policy remains NumPy/source behavior, not a new
physical validation contract. Random Hermitian generation in `matrix.py` is
outside this slice; its parent remains open. Polynomial tests cover the complete
single source function, not all linalg or G2 acceptance.

## Verification and resource interpretation

`tests/unit/test_unitary_polynomial.py` compares complex64/complex128 data against
the unchanged source for dimensions 1, 2, 4 and offsets -3, 0, 2. Independent
spectral evaluation checks the result. Real/empty coefficients, invalid shapes,
complex-to-real casting refusal and nonunitary adjoint behavior are separate
cases. `tests/unit/test_unitary_distance.py` checks global and relative phases,
shape errors, clamp behavior and the controlled-phase counterexample. Original
upstream tests are run separately without editing their code.

| Quantity | This numerical batch |
| --- | --- |
| Added physical primitive instructions | 0 |
| Added work wires / measurement branches | 0 / none |
| Source QEC gate/resource model | Not supplied by either numerical function |
| Dense numerical storage | O(N**2), multiple matrices; not a measured peak |
| Polynomial matrix products | One per coefficient plus initial matrix power, retaining source order |
| Distance matrix products | One trace-overlap product |

These are dense inspection utilities, with no scalability, timing, differentiation
or synthesized gate-count claim. The existing circuit decomposition and exact
measurement-branch conformance are unaffected.
