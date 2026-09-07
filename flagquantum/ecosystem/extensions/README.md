# Extension protocol

`flagquantum.ecosystem.extensions` is the public boundary for optional
third-party extensions. It owns extension manifests, capability negotiation,
task-local registration, lifecycle isolation, and conformance checks.

It does not own compiler passes, numerical kernels, execution scheduling, or
provider implementations. An extension implements those domain contracts; the
extension package only validates and contains its lifecycle.

Start with `sdk.py` for the protocol and `conformance.py` for executable checks.
The shortest working example is `examples/extensions/reference_extensions.py`.
Run:

```bash
python -m pytest tests/unit/test_issue080_extension_sdk.py \
  tests/api_contract/test_extension_protocol_semantics.py -q
```
