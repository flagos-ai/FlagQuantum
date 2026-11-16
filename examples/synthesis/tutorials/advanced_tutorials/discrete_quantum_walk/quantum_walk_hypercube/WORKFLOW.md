# Native hypercube quantum walk

The original four-bit position and uniform two-bit coin are retained. Each step
reflects the coin about its uniform state, then flips exactly the logical
position bit selected by the coherent coin value. The helper `move(direction,
coin,x)` expresses one selected move with a bounded comparison predicate and X.
Its classical direction is bound before capture; the user does not design a
multi-controlled decomposition or choose physical wire numbers.

The native main binds real position/coin amplitudes and a step count0..16.
Defaults preserve the original zero position, uniform coin and t=2. Compiler
limits still apply to the unrolled program. Uniform diffusion composes the
original preparation inverse, block-zero phase and global minus sign using
adjoint and within/apply. Predicate inputs in borrowed helpers must cover a
complete raw register in logical bit order; partial numeric views are refused.

Tests compare unchanged-source/native/independent Circuit full complex states
for t=0/1/2/3. Two additional nonzero coherent position/coin models use those
same step counts. The independent reference constructs a classical hypercube
permutation and the exact coin reflection. All named position/coin marginals
and every workspace amplitude are checked without phase alignment. Actual
width/work/operations/depth/memory budgets refuse infeasible requests.

At t=2 the native program uses14physical bits(6logical+8work),418operations and
depth337; the source uses6bits,zero work,36operations/depth26. This gap is
recorded without optimization claims. Named probabilities require no bitstring
decoding. This accepts the quantum program, not a complete Notebook workflow.
