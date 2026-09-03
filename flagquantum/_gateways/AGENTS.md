# Agent Gateway Transitional Boundary

Gateway code translates protocols into Agent Services calls. It must not invoke
Compiler internals, Runtime internals, kernels, devices, or providers directly.
Production MCP/REST/gRPC lifecycle remains owned by the external Compute Service.
