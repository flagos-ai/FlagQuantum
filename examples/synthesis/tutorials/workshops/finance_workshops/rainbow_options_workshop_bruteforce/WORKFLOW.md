# Native Rainbow payoff amplification

The application physics module preserves both identical four-outcome asset
distributions and the eight original rotation angles. It computes a finite
joint-selector model from `max(x1, (x1+x2)/2 - 3/2)` using exact rational
arithmetic. Multiplying this result by two recovers the original three-bit,
one-fraction-bit raw selector. Reinterpreting those bits at another scale
in the source does not change the raw rotation-table index.

`effective_angles()` evaluates this formula at each of the 16 possible joint
asset words and checks exact representability before selecting the original
angle. This is classical model construction, not host execution of quantum
data. It retains the formula instead of hardcoding its accidental simplified
value on this small domain. The quantum program coherently loads both assets,
applies the derived Y-rotation table and performs bounded amplification using
ordinary qfunc composition, adjoint, within/apply and phase control. The local
scaled-payoff register need not exist because it is used only as a terminal
rotation selector, exactly as in the source adapter's checked fusion.

The native asset array uses its first two logical bits for x1 and its next
two for x2. The joint table index is x1+4*x2. Named indicator output is read
without physical wire numbers or bitstring decoding. Dropped asset storage
remains retained in the complete state, rather than being falsely declared
zero workspace. Both originals compare full complex states against an
independent Circuit for k=0/1/2 without phase alignment. Additional checks
cover every joint selector with indicator0/1, two nonzero coherent loading
vectors, exact scale indices and actual width/operation/depth/memory refusal.

At k=1 the native program uses5physical bits,zero work,398operations and
depth343;the source uses5bits,354operations and depth306. No optimization
claim is made. Iteration counts0..16 are bounded before synthesis; compiler
construction and resource budgets still apply. See the [shared amplification
guide](../../../../amplification_guide.md) for full phase and lifecycle semantics. This accepts
the quantum program, not a complete Notebook, pricing estimator or financial
result.
