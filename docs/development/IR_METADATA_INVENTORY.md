# FlagQuantum IR Metadata Inventory

Status: Phase 0 static inventory and importer-relevant typed destinations established.
Machine inventory: `tests/fixtures/internal_ir/metadata_inventory.json`

## 1. Method and Scope

Static AST scanning found 55 keys read/written through string literals in
metadata-like mappings. The scan covers `get/pop/setdefault/subscript` in
`flagquantum/**/*.py`, with tests detecting drift.

This does not mean CircuitIR has 55 public fields. Metadata containers with the
same name occur in Instruction, MeasurementNode, CircuitIR, ExecutionResult,
runtime evidence, and DeploymentPackage. The inventory prevents importers from
confusing these scopes.

Static scanning is not complete semantic analysis. Dict merges, variable keys,
external payloads, and keys produced but never read need manual review.
`logical_state_shape` is currently recorded separately.

## 2. Importer-Relevant Categories

- **Instruction semantics**: channel, dynamic/condition, classical bits, global
  phase, and diagonal/MPO/Pauli/Schmidt hints.
- **Execution request**: seed, postselection, measurement format/name, and resource limits.
- **Constraints**: batch size, runtime configuration, and manually recorded logical
  state shape.
- **Compiler evidence**: routing, routing strategy, and dependency scheduling evidence.
- **Interop pending review**: circuit name, interop, num_clbits, and Qiskit labels.
- **Outside source CircuitIR**: deployment/provider payloads, runtime reverse
  records, claimability, and platform evidence.

The machine inventory is authoritative for complete keys and classifications.

## 3. Phase 1 Rules

1. Identify the metadata's owning object before classifying semantics.
2. Keys affecting execution, legality, results, gradients, or identity need typed destinations.
3. Routing/provider/runtime evidence does not enter program semantic hashes.
4. Unclassified keys cannot be silently dropped or automatically treated as provenance.
5. Dynamic/condition metadata is rejected structurally in `circuit_ir_v1_static`.
6. When drift tests fail, classify new keys; do not simply update expected sets.
7. This inventory neither freezes public metadata APIs nor promotes internal keys
   to compatibility commitments.

## 4. Completion Checklist

- [x] Repository-wide static consumed-key inventory.
- [x] Initial separation of importer-relevant and external metadata.
- [x] Input format established for automated drift tests.
- [x] Manual secondary audit of variable keys, dict merges, and provider payloads.
- [x] Typed destinations for every importer-relevant key.
- [x] Semantic/provenance decisions for the four Interop keys pending review above.
- [ ] Compiler/runtime owners approve final classification.
