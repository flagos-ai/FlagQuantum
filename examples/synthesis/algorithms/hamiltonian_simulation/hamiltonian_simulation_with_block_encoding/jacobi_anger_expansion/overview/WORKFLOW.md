# Native workflow acceptance

Retain evolution time 22, approximation target 1e-7, all 100 points from zero to one and both approximation/error panels. This is a classical polynomial workflow, not a quantum circuit. Native Torch cosine projection and Clenshaw evaluation replace external polynomial helpers. A conservative Bernstein-ellipse bound selects degree 49; the Classiq degree-selection policy is not reproduced. The exact-coefficient truncation bound and measured floating-point/quadrature error are reported separately. A 75-digit independent Decimal Bessel series checks coefficients, and 4097 points check actual error on [-1,1].

Every source code cell is reviewed and hash-bound in provenance. Examples use
native Python, public FlagQuantum APIs and Torch. There is no source-language
loader or new production dependency.

VQE acceptance compares every optimizer evaluation's full complex state with an
independent RY Circuit, and its energy with cos(theta) + 0.1 sin(theta). The final
energy must agree with -sqrt(1.01) within 1e-10 and start at angle zero. Grover
acceptance checks all five stages, original register sizes/order, both empty
drafts, the independent arithmetic phase oracle/reflection, and zero workspace.
No global phase alignment is allowed in the quantum comparisons.

Jacobi acceptance independently computes Bessel coefficients with a 75-digit
Decimal series, checks parity and the conservative exact-coefficient tail bound,
and evaluates errors at 4097 points separately from the original 100 plot points.
The 14 by 5 inch plot retains four approximation curves and two error curves.
Plot tests skip if optional Matplotlib is absent; numerical tests still run.
Real Jupyter execution complements headless tests. Checked-in Notebooks contain
no execution outputs.

Notebook acceptance is separate from the frozen 216-program quantum corpus.
The classical Jacobi workflow does not add a quantum program to that corpus.
No hardware, scaling, optimizer-equivalence or resource-optimality claim is made.

```sh
python -m pytest tests/integration/test_synthesis_vqe_grover_jacobi_notebooks.py -q
```
