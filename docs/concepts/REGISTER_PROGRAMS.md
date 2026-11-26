<!-- Adapted navigation roles from Qualtran v0.7.0 docs/bloq_infra.rst; Google LLC, Apache-2.0. -->

# Register programs and resource reasoning

Register operations describe reusable algorithms over typed logical words.
Core owns types and program semantics, Compiler owns decomposition and binding,
Simulation owns numerical execution, and Drawer owns presentation. Compose an
operation through the existing `circuit.append(op)` workflow; advanced program
metadata stays behind that entry point.

Follow the tutorials in order, then choose a concept or specialist workflow.
Notebook links download the existing native teaching assets for local execution.
They do not execute code during documentation rendering. Install optional SDKs
only for workflows that require them. Each notebook or owning guide records its
supported inputs and evidence; a navigation entry does not certify every
parameter domain or make a resource estimate an emitted gate count.

## Tutorial

- {download}`Wire registers <../../examples/notebooks/register_wiring.ipynb>`
- {download}`Build reusable components <../../examples/notebooks/register_components.ipynb>`
- {download}`Address individual bits <../../examples/notebooks/register_bit_addressing.ipynb>`

## Concepts

- {download}`Component protocols <../../examples/notebooks/component_protocols.ipynb>`
- {download}`Register data types <../../examples/notebooks/register_data_types.ipynb>`
- {download}`Classical evaluation <../../examples/notebooks/classical_simulation.ipynb>`
- {download}`Tensor simulation <../../examples/notebooks/tensor_simulation.ipynb>`
- {download}`Measurement-assisted uncomputation <../../examples/notebooks/measurement_uncomputation.ipynb>`
- {download}`Superoperator simulation <../../examples/notebooks/superoperator_simulation.ipynb>`
- {download}`Call graphs <../../examples/notebooks/call_graph_protocol.ipynb>`
- {download}`Qubit counts <../../examples/notebooks/qubit_count_protocol.ipynb>`
- {download}`Exact adjoints <../../examples/notebooks/register_adjoints.ipynb>`
- {download}`Quantum controls <../../examples/notebooks/register_controls.ipynb>`

## Quantum computer architectures

- {download}`Physical cost model <../../examples/notebooks/physical_cost_model.ipynb>`
- {download}`Beverland resource model <../../examples/notebooks/beverland_resource_model.ipynb>`
- {download}`THC physical compilation <../../examples/notebooks/thc_physical_compilation.ipynb>`
- {download}`External physical-resource estimates <../../examples/notebooks/physical_resource_interop.ipynb>`

## Advanced topics

- {download}`Component infrastructure <../../examples/notebooks/register_component_infrastructure.ipynb>`
- {download}`Composite construction <../../examples/notebooks/component_construction.ipynb>`
- {download}`Cirq register interoperability <../../examples/notebooks/cirq_register_interop.ipynb>`
- {download}`Cirq T-complexity estimates <../../examples/notebooks/cirq_t_complexity.ipynb>`
- {download}`Resource interchange <../../examples/notebooks/resource_interchange.ipynb>`
- [Register-gate interfaces](../development/REGISTER_GATE_SOURCE_AUDIT.md)
- {download}`Specialized controls <../../examples/notebooks/specialized_control.ipynb>`
- {download}`Register graphs <../../examples/notebooks/register_graphs.ipynb>`
- {download}`Musical scores <../../examples/notebooks/register_musical_score.ipynb>`
- {download}`Resource call-graph diagrams <../../examples/notebooks/resource_call_graphs.ipynb>`
- {download}`Classical/tensor cross-checks <../../examples/notebooks/classical_tensor_crosscheck.ipynb>`
- [Documenting components](../development/COMPONENT_DOCUMENTATION.md)

Measurement-assisted uncomputation and documentation example factories retain
their recorded source limitations. Resource and physical-cost notebooks consume
models; they do not by themselves establish full state preparation, clean
workspace or hardware scalability. Optional interoperability stays outside Core.
The API reference and generated component catalog have separate acceptance
boundaries from this fundamentals navigation.
