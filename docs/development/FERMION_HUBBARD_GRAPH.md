# Spinful Hubbard model on an explicit graph

## Scope before implementation

This proposed Algorithms-owned extension starts from accepted main
`4b1fbd7edcb2f24f00a74ff5ab1e921b1bcf89cb`. The initial native migration and
H2 encoding replacement are complete; this is a separate model extension.

Add the experimental concrete function `fermi_hubbard_graph(n_sites, edges,
*, t=1.0, u=0.0, mu=0.0)` in the existing `algorithms.fermion.hubbard` module.
It returns the existing `FermionOperator`. Sites retain interleaved up/down
modes, uniform minus-t hopping, unshifted onsite U and minus-mu population.
`edges` is a finite sequence of endpoint pairs describing an undirected simple
graph. Each edge is counted once; either orientation and any order give the same
operator. Refuse self edges, duplicate/reversed edges, out-of-range endpoints,
noninteger endpoints and malformed pairs. Empty graphs and isolated sites are
valid. Coefficients remain finite static real numbers.

Model assembly owns no graph library, solver, encoding, circuit preparation or
runtime policy. Do not change the existing chain/ring signatures or validation,
Stable Core exports, dependency manifests, CI or capability registry. Reuse their
existing authoritative algebra and scalar/count validation. A later Integration
change may register the new experimental boundary after implementation evidence;
existing chain/ring capability entries must not claim arbitrary graphs.

## Acceptance before implementation

- Compare independent occupation-bit matrices on branching and disconnected
  graphs, including isolated sites, Hermiticity and separate spin conservation.
- Replace chain/ring producers with explicit path/cycle edge lists without
  changing their encoding and measurement consumers. Check all three encodings.
- Check edge orientation/order invariance, zero-width retention, empty graphs,
  onsite terms and failures before encoding or execution.
- Use an explicit spectator on a non-neighbor hop to check the fermionic sign
  independently; ordinary native expectations, angle gradients and an SGD
  update must match an analytic two-determinant reference.
- Supply a small executable CPU example with no dense eigenstate preparation
  or external science SDK; include its CLI in the existing example inventory.

This is bounded CPU `single_device_fast_path` correctness, not a performance,
accelerator, distributed or scalability claim. Weighted/complex edges, flux,
spin-dependent coefficients, trainable Hamiltonian parameters and general
encoded ansatz transformation remain outside this scope.
