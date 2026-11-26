# Compiler artifacts

Owns compiled-program routing evidence and its validation. Runtime prepares execution and Remote validates submitted packages; neither owns a second routing implementation. Depends on Compiler topology and existing Core artifact contracts. Published entry points retain their existing deployment imports.

## Ten-minute contributor path

Read the existing implementation and its scenario test, make one focused change,
then run from the repository root:

```sh
python -m pytest tests/unit/test_deployment_routing_evidence.py -q
python tools/check_architecture.py
```

Keep existing public contracts and serialized identities. Physical accelerators
and scalability claims retain their separate acceptance gates.
