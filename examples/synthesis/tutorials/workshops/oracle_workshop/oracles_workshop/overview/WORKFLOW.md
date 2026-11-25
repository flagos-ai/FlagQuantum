# Oracle Workshop semantic boundaries

This is a `single_device_fast_path` local workflow. `ansatz.py` owns fixed native
quantum functions. `workflow.py` owns independent references,phase tables,actual
sampling and classical classification. The Notebook and `main.py` compose them.
Only public FlagQuantum APIs are used; there is no source importer or provider job.

The bit oracle XORs a target with2*x+y==4. Independent tests must cover nonzero marks
and superposed marks, not only a zero-initialized output. Phase kickback prepares
an auxiliary with X then H,uses the same XOR,and reverses preparation automatically.
It is already zero after within/apply; adding a further X would leave it dirty.
Clean actual auxiliary amplitudes are verified independently before lifetime ends.

Deutsch-Jozsa keeps the three-bit balanced predicate x%2==0. Classification operates
on actual positive-count observed words: a single zero means constant,a single
nonzero word means balanced,and multiple observed words are undetermined.

Grover keeps x/y as separate unsigned two-bit words. The comparison x-y==2 uses the
integer difference,so negative differences are never mistaken for modular positives.
The root allocates one auxiliary,borrowed across the two phase-helper calls. Diffusion
is2|s><s|-I,including its global sign. The exercise's placeholder5 repetitions becomes
the source solution's2. No known solution is injected into sample counts.

The source struct and generic built-in callback applications have explicit native
counterparts for these fixed examples. The workflow does not advertise generic
higher-order Python qfunc or a new qperm/phase_oracle/grover_search API. These limits
are documented rather than hidden behind a source import or a Circuit-only wrapper.

Independent Circuit references enumerate XOR permutations,phase diagonals and the
Grover reflection. They share no native compiler lowering. Actual full statevectors
are reordered by declared logical register mappings before hidden wires are checked.
Global phases are compared directly. Source-cell hashes and revision remain in
provenance without copying original source into the user examples.
