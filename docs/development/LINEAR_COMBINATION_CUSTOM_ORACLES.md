# Supplied linear-combination oracles

G2-BE08 extends the native linear-combination composition with the original
Qualtran v0.7.0 supplied PREPARE/SELECT paths. Locked commit:
`bcb17bb25360ed68ce3bc084645db3e929799389`; parent blob
`a3d8aa7fafa7a6f3fcaf095826751209427cd255`. Original source stays unchanged.

The existing LinearCombinationBlockEncoding accepts prepare_override and/or
select_override, using existing BlackBoxPrepare and SelectBlockEncoding
contracts. Either supplied oracle can replace its default independently. Packed
selection widths must agree; the supplied SELECT must have exactly one target
named system, packing the constituent system and maximum ancilla/resource words.
Inner PREPARE and SELECT port names may differ, as the source black boxes expose
packed outer registers. Both overrides may enlarge the selection domain beyond
the original term count. Junk can be empty or nonempty.

Source structural validation does not verify whether supplied oracles implement
the advertised coefficient magnitudes and signs. Their numerical behavior is
owned by the caller, and the native path preserves it unchanged. In particular,
a supplied SELECT owns its sign phases: default signed children are not appended
or charged again. PREPARE's supplied normalization does not replace alpha's
original sum(abs(lambda_i)*alpha_i) formula. Tests deliberately distinguish them.
The runnable example supplies a mathematically consistent pair for (X+Z)/2.

A supplied controlled SELECT uses the existing ControlledSelectEmitter contract.
It receives one positive control. Preparation stays unconditional. A missing
controlled emitter fails before preparation emission. Supplied SELECT owns its
workspace; default child factories/work groups must be empty. Existing emitter
binding checks restrict quantum access to each oracle's own registers and work.
The complete root may reuse scratch across sequential PREPARE and SELECT calls.
No generic control is added to measured primitive IR.

The normalization/padding arithmetic is shared independently of coefficient bit
precision. This enables symbolic precision when PREPARE is supplied, whether
SELECT is supplied or default, without building the default alias table. Even
unused zero/negative precision values pass through that original supplied
PREPARE path; they are not spuriously validated as a default alias request.
The default probability factory still
rejects symbolic precision, matching the original unsupported condition. This
is not symbolic amplitude synthesis. Symbolic block-alpha probability
construction and further control specifications remain separate work.

Thirty-two original source configurations cover PREPARE-only, SELECT-only,
both, and both with zero preparation junk; mixed constituent auxiliary widths;
forward/adjoint; and external control present/absent. Primitive dependencies,
operands, measurement feedback, metadata, default QEC categories and peak width
match. Sixteen full source matrices act on arbitrary complex states across all
logical ports, including junk, bystanders and controls. Every small measurement
history is compared as a complete unnormalized vector with clean private work,
without phase alignment or partial traces. Eight more source gate/model cases
verify symbolic precision and explicitly forbid default factory access. Four
further source gate cases verify unused zero/negative precision passthrough.

Five original/native constructor failures cover mismatched selection widths and
invalid SELECT target width/name. Controlled-emitter ownership failures and the
original default symbolic factory refusal are separate tests. The original
zero-total-ancilla parent exposes metadata but its decomposition and QEC query
raise TypeError for a missing ancilla parameter. Native binding and resource
composition explicitly reject that same unsupported route; they do not invent
a working replacement and report it as original parity.

Run:

```sh
python -m examples.algorithms.linear_combination_custom_oracles
```

The example supplies a two-bit selection preparation with a junk tag and a
four-row SELECT. User execution is circuit.append(op). It checks the projected
(X+Z)/2 block, private scratch zero and full complex inverse restoration.
The unchanged shared indexed traversal emits and models 8 T/T-dagger,
2 measurements and 5 peak logical qubits. Counts remain separate fields;
no physical resource measurement is inferred from the model.

All work stays in algorithms/primitives. Compiler traversal, arithmetic,
partitioning and resource accounting remain authoritative. No new oracle
protocol, compatibility layer, IR, simulator or Stable Core export is added.
