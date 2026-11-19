# Control descriptor metadata and lifetime audit

This audit completes the metadata dependency review for Q-0158 and Q-0159 at
Qualtran v0.7.0 revision `bcb17bb25360ed68ce3bc084645db3e929799389`.
The parent files remain open for their other outstanding execution dependencies.
The frozen source remains unchanged. Native ownership is distributed across
existing Register, RegisterType, Signature, ControlRegister, SymbolicArrayShape
and numerical pattern functions.

## Complete metadata member disposition

| Frozen member | Native owner and evidence |
| --- | --- |
| Constructor dtype grouping | Explicit native Register tuples preserve dtype order. RegisterType constructors are independently audited by existing register metadata/type tests. Source implicit dtype grouping is foreign object construction. |
| `_cvs_convert` | Simulation normalize_control_patterns preserves the full converter, including scalar/list/grouped arrays, iterator consumption, symbolic shape and ndarray identity. Existing test_classical_control_patterns.py covers malformed and unusual Python inputs. |
| `__attrs_post_init__` | Source asserts matching dtype/pattern counts. Native consumers validate explicit descriptor/pattern counts before binding. The default selector, dense tensor owner and physical predicate owner retain their domain-specific errors. |
| `num_ctrl_reg` | Length of the ordered Register tuple. |
| `shapes` | Register.shape_symbolic, matched against normalized activation shapes. |
| `concrete_shapes` | Register.shape for native concrete admission. The source symbolic-tuple leak is documented below. |
| `num_bits`, `num_qubits`, `num_cbits` | Existing Signature n_bits/n_qubits/n_cbits over THRU control ports and Register total widths. Mixed metadata counts remain independent of execution admission. |
| `is_symbolic` | Existing is_symbolic_value over Register descriptors and normalized patterns. Unknown values are symbolic even with concrete dimensions. |
| `activation_function_dtypes` | Ordered Register dtype/shape pairs, passed to existing typed consumers. |
| `__cvs_tuple`, `__eq__`, `__hash__` | Existing immutable Register and ControlRegister tuples retain shape/dtype/integer activation identity. SymbolicArrayShape retains unknown activation identity. Source numerical hash values belong to foreign objects and are not serialized. |
| `is_active` | Existing classical_controls_active, with the explicit Register descriptors supplied as symbolic_parameters. |

No additional metadata class, graph, key format or public export is introduced.
Native register names label graph connections and are retained. Source CtrlSpec
has no names. Equality comparisons therefore use corresponding names and
existing tuple records at the domain boundary. Snapshot construction happens
explicitly when an array is bound to immutable ControlRegister activation words.

`test_control_descriptor_metadata.py` compares scalar/array/empty-array shapes,
64-bit signed values, unsigned and classical words, zero-width descriptors,
all three aggregate counts, mixed quantum/classical metadata and symbolic
widths/shapes. Its equality matrix distinguishes QBit, one-bit unsigned words,
signed encodings, classical storage, scalar versus singleton shapes and
multi-register grouping. Equal native/source descriptors remain valid keys in
their respective dictionaries or sets. Symbolic identities use the existing
shape records. Existing source-derived converter and register-type tests cover
the generic constructor dependencies.

## Array mutation and source cache inconsistency

Source CtrlSpec retains an externally mutable ndarray. Hashing or comparing it
caches a flattened tuple of values. Mutating a scalar array from one to zero
after that cache has been populated leaves the old hash and default equality
intact. Its current is_active and direct _tensor_data methods read the live zero.
The default dispatcher still chooses Controlled, whose stored decomposition is
a positive CX. Its decomposition-based tensor is also positive CX. Its direct
_tensor_data instead describes negative CX. This is a source inconsistency.

The complete native witness retains array identity through
normalize_control_patterns and verifies its current numerical activation and
direct controlled tensor against the source live-array methods. A previously
bound ControlRegister retains its positive immutable snapshot. A freshly bound
snapshot is negative. Both run through circuit.append(op) on a complex state
entangled with a spectator. Each agrees with its corresponding original source
path. Positive emitted CX and negative X/CX/X have exact gates and operands.
This explicitly separates the two activation lifetimes. It does not infer one
consistent operation from the contradictory upstream paths.

The reproducible source result is saved in
`ftqc-reference-env/g2-control-metadata-mutation-boundary.json`. The regression
checks the frozen controlled.py blob
`6fabe760e793f1b39f825d76f80e92b6e5fa5c28` before exercising this behavior.

## Symbolic concrete-shape leak

Frozen CtrlSpec.concrete_shapes passes each shape tuple as an opaque argument to
is_symbolic, which does not recurse into tuples. Shaped((n, 2)) consequently
returns ((n, 2),) from this supposedly concrete property. Register.shape tests
individual dimensions and rejects the same symbolic dimensions. Register's own
behavior agrees with the frozen upstream Register. Native aggregate symbolic
bit counts still match CtrlSpec exactly. The test preserves this discrepancy
without admitting symbolic dimensions to the physical binder.

## Execution and resource evidence

The new lifetime witness uses no workspace or measurement. Its positive snapshot
emits one CX. Its negative snapshot emits two X and one CX. The direct dense
inspection path emits zero instructions. Previous
`g2-default-control-resources.json` records the corresponding source analytical
models of one and three Clifford gates, separately from observed primitives.
Existing concentrated regressions retain the two-control measured branches,
full complex amplitudes and zero predicate work leakage. Native Kickmix and
unchanged-source version checks are reused from that batch and the regression.

## Remaining parent dependencies

Symbolic typed execution, additional physical encoding admission and generic
classical stored-child graph control remain open. Source cache mutation and
concrete-shape leakage are explicit diagnosed boundaries. They are not treated
as successful physical equivalence. The complete controlled source and test
parents remain unaccepted until their remaining dependencies are closed.

All 4086 concentrated cases pass with one existing source deprecation warning.
Strict typing passes the new complete test file. Black, Ruff, Simulation team
scope, architecture and multi-level IR checks pass. The dependency scan reports
687 existing whole-repository findings and zero batch-path findings. Production
code is unchanged, so the previous 96-case SDK integration result is reused.
