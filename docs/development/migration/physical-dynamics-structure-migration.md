# Physical dynamics source ownership

The Rydberg Hamiltonian assembly, presence segments and Ising braid actions
move from the Simulation root to its existing Dynamics domain. Runtime calls
these numerical owners directly. All three previous internal modules are
deleted; no compatibility files, module aliases or forwarding functions remain.
The modules define functions rather than public record types. Existing public
program, plan and result records keep their owning modules and serialization.

The three source files are unchanged by the incoming #1084 base-to-head diff
through audited head `dcd7a8cf37436416492595a5c1061a624cf562fe`.
The complete 4,376 changed-text audit and subsequent eight-file Cirq increment
contain no old numerical module path consumers. #1084 remains a separate
integration gate for Core physical records and construction; these source
moves do not integrate or replace that functionality. Recheck its head before
creating the final integration patch.

Validation must include the existing physical action, sampled/unread branch,
Runtime and gradient tests, normalized function AST comparison, source ledger
and retired-import checks, public API/type checks, documentation and wheel
installation. Existing mathematical assertions and frozen evidence stay intact.
Temporary preview results do not qualify a pushed commit; its full CI must pass.

The next verified incoming increment, from `dcd7a8cf` through
`7bbffeed70da51252e6b9f36fd05b1d85509cd6f`, adds six commits and thirteen
format-factory, test and audit-document files. All thirteen blobs are
SHA-verified; none intersects the migration sources or targets or contains an
old operational path reference. The actual retired-import guard also passes
on these files, including relative and parent-package imports.
