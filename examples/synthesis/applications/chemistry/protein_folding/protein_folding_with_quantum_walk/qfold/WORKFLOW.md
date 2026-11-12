# Native QFold coined step

The example preserves the two source angles(4.994422698e-6 and
1.372174780775247), uniform logical phi/psi and the exact coin/shift/inverse-coin
ordering followed by the coin/move zero-state phase oracle. Logical source
registers C/M are named coin_bit/move_bit in Python. The two conditional X
moves and conditional real Y rotations are ordinary borrowed quantum helpers.
The main composes them with within/apply and reads named phi/psi probabilities.

The two local coin/move bits are dropped from the public register interface but
remain part of the complete quantum environment. They need not be zero. Only
compiler predicate workspace is required to return to zero. Tests explicitly
check this distinction instead of padding the retained environment with zeros.
User code declares the quantum storage, angles and state; the compiler handles
predicate computation, controlled gates, inverse scratch identity and cleanup.

The unchanged source, native default and independent dense rotation/permutation
reference agree in complete complex amplitudes. Two further nonzero angle/data
models and all16columns of the entire step,including nonzero coin/move inputs,
verify XOR,bit order,global phase and borrowed-name resolution. Generic predicate
checks also cover complex inputs,partial-view refusal,Const/init/alias/Output
safety and inverse composition. Actual width/work/operations/depth/memory limits
are enforced. The native program uses11physical bits(4retained data/environment
+7work),399operations/depth315; the source uses4bits,zero work,87operations/
depth69. No resource parity or optimization claim is made. This accepts the
quantum program,not a complete protein-folding optimizer or Notebook workflow.
