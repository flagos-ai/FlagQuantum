# Discrete quantum walk verification

The circle retains the source's seven signed position bits, a single Hadamard
coin, and 50 steps for each initialization. The first starts from a zero coin.
The second prepares the balanced complex coin with H then S. Each step applies
the same clockwise or counterclockwise modular shift. The hypercube retains all
four position bits, the uniform two-bit coin, and four Grover diffusion steps.
Independent permutation Circuits check complete complex states and every
compiler work bit. Sampling uses independent local seeds.

The classical circle comparison starts each independent 50-step trial at zero.
The source's classical hypercube code instead records one continuous, uniformly
directed trajectory of 10,000 moves. Its opposite-corner visit frequency
therefore describes a different observation from the four-step quantum run.

The benchmark uses the source's ten position bits and one coin. The built-in
arithmetic and explicit multi-controlled X cascade are both native qfuncs.
Each width limit visits both width and depth compiler objectives. Each resulting
circuit is converted exactly to CX/U3 and counted in that same basis. Selection
uses the actual CX count, with depth and physical width as tie-breakers, and
retains any tighter-budget candidate that already fits. These are the best
circuits among the visited candidates. The CX objective is applied in this
example's search over measured circuits. The benchmark does not assert a global
minimum or a matching trajectory to the hosted source optimizer.

Selected circuits at the two endpoints undergo an additional complete-state
reference check after CX/U3 conversion. The two source styles can yield the
same resource curve because their lowering shares multi-controlled operations.
