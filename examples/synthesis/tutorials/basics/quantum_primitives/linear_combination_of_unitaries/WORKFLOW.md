# Coherent Fourier linear combination

The model retains the original controller probabilities(0.5,0.25,0.25,0).
Controller0 selects identity,1 selects QFT,2 selects inverse QFT and3 selects
identity. `within` loads the controller distribution, `apply` selects the
coherent operations, and leaving the region unprepares the controller. The
controller remains part of the quantum output; the example does not silently
postselect it or renormalize the desired LCU block.

The native qfunc uses existing high-level QFT, control and adjoint operations.
The conjunction controller>=j and controller<=j expresses equality with the
existing typed quantum comparison API. The compiler forms a closed temporary
predicate computation, inverts only the controlled QFT and then uncomputes and
releases that predicate. This does not admit arbitrary allocations or escaping
locals inside inverse regions. No user physical wire numbers, Fourier gate
sequences or bitstring decoding are required.

Default data is zero, matching the original. `build_program` also accepts four
real data amplitudes and four classical controller probabilities. Compiler
validation remains authoritative for finite values, normalization contracts
and resources. Tests compare the unchanged source, native program and an
independent dense Fourier SELECT matrix in complete complex amplitudes. Two
further nonzero weight/data models and all four initial data basis columns
exercise the inverse branch and coherent interference. Controller/data named
marginals and all workspace amplitudes are checked without global-phase
alignment. Additional compiler tests cover nested regions, input stability,
SSA freshness, relative/global phase and ownership/overlap refusal.

The default native program uses12physical bits(4data/output+8work),146operations
and depth128. The source adapter uses4bits,zero work,264operations/depth228.
A direct single comparison predicate probe uses6work bits; the public example's
composite range predicates use8. These are distinct resource tradeoffs, not an
optimization claim. Width, workspace, operations, depth and memory constraints
are enforced on the actual synthesized program. Read the main module's named
controller/psi probabilities using the runnable command in README. This accepts
the quantum program, not a full Notebook workflow or a successful LCU postselection.
