# Extension protocol

`flagquantum.ecosystem.extensions` is the public boundary for optional
third-party extensions. It owns extension manifests, capability negotiation,
task-local registration, lifecycle isolation, and conformance checks.

It does not own compiler passes, numerical kernels, execution scheduling, or
provider implementations. An extension implements those domain contracts; the
extension package only validates and contains its lifecycle.

Installed extensions register zero-argument factories in the
`flagquantum.extensions` Python entry-point group. Discovery is explicit and
kind-specific, so importing FlagQuantum never loads third-party packages. See
`docs/guides/COMPILER_PLUGINS.md` for the circuit-compiler path.

Start with `sdk.py` for the protocol, `conformance.py` for executable checks,
and `examples/extensions/reference_extensions.py` or
`examples/extensions/reference_compiler_extension.py` for the shortest working
implementations. Run:

```bash
python -m pytest tests/unit/test_extension_sdk.py \
  tests/api_contract/test_extension_protocol_semantics.py -q
```
