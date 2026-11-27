# Exact clean-release analysis for literal field circuits

This internal integration contract replaces a conservative degree-based literal
inverse refusal with analysis of the actual bound primitive circuit. It changes
no Stable Core export, IR schema, codec, generic binder or public API contract.

## Ownership and interface

Compiler retains the literal source Program, including all original releases,
and binds ordinary wire leases. It does not execute numerical kernels or select
backends. Mathematical field operations remain separate from literal recipes.

Simulation owns an internal exact Boolean-polynomial analysis over Core
`CircuitIR`. Its inputs are arbitrary quantum input wire IDs and clean-release
pairs `(instruction_offset, wire_tuple)`. All other wires start zero. Offsets
refer to the instant before the instruction at that position. A successful
return proves every specified release is zero for every input, including
arbitrary complex superpositions entangled with an external spectator. A
nonzero algebraic normal form raises `ValueError`; unsupported gates, guarded
instructions or exhausted analysis resources never count as success.

The analysis accepts only unconditional X/CX/CCX permutations. It models each
wire by its exact multilinear polynomial over GF(2), using `x*x=x` and XOR
cancellation. A backward dependency slice excludes operations unable to affect
any requested release. Resource bounds apply to symbolic expansion, not to
accepted quantum amplitudes. No numerical field callback, source correction,
measurement-branch sampling, reset or basis-only positive test substitutes for
this proof. No new certificate, registry or result type is introduced.

Examples compose the Compiler and Simulation owners: `FieldInverse.bind`
constructs and binds the literal source, validates every clean release through
Simulation, then exposes the existing bound record. User code remains
`circuit.append(op)`. The default mathematical profile retains its existing
implementation. Literal catalog graphs retain original invalid paths for
inspection without granting execution admission.

## Acceptance

* Compare the actual source and native primitive streams and release points.
* Reproduce original dirty frees, including cases outside degrees four/eight.
* Validate all admitted positive profiles by exact polynomials, not a width
  whitelist. Exercise the same analysis with literal product private padding.
* Cross-check polynomial zero/nonzero results against complete small truth
  tables and complex operators. Preserve unused-work and entangled-spectator
  checks and exact native inverse restoration.
* Reject guarded/non-permutation instructions, invalid offsets or wire layouts,
  and symbolic expansion beyond the explicit bounded budget.
* Replace the analysis provider in the append scenario without changing its
  consumer; unsupported or failing analysis must prevent composition.
* Run local source/dependency regression, strict typing and repository checks.

Original invalid clean releases remain source defects under the approved
faithful-migration policy. Actual symbolic-analysis resource limits are separate
performance boundaries. Unimplemented valid-source paths are not waived.
