# Choose a synthesis objective

The same native quantum function prepares a uniform three-bit input and computes
`y |= x**2 + 1`. `build_program("depth")` and `build_program("width")` select the
existing synthesis objectives without changing the user's computation.

The reviewed Notebook synthesizes under both objectives, opens both circuits,
and prints their measured width and depth. The native `main.py` performs all
these computational steps. Local text drawing replaces the cloud viewer;
The public `estimate_resources(program.circuit)` measures allocated width and
gate depth using the same canonical estimator as the synthesis report.
The example additionally executes each result and reads named probabilities.

An objective selects among the compiler's available candidates. It is not a
guarantee of global optimality, and the two objectives may choose identical
resource counts for this expression. The example prints the measurements rather
than claiming an improvement. The original resource baseline remains unchanged.

The recorded local result for both objectives is ten allocated qubits, sixteen
operations and depth twelve, including one clean workspace qubit. This matches
the frozen quantum-source resource row. These are measurements of this example,
not an expected advantage of one objective over the other.

Acceptance checks both objective choices against an independent reversible
Circuit over all complex amplitudes, absolute phase, register order and clean
hidden workspace. Each result has the eight pairs `(x, x*x + 1)` with probability
1/8. The actual runnable script is also checked for both reports and readouts.

The [provenance record](provenance.json) pins the Notebook and every reviewed
code cell. Run the commands in [README](README.md).
