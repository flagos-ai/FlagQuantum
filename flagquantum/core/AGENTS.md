# Core Team Boundary

Core owns backend-neutral semantics and versioned domain types. It must not
import Runtime, Compiler implementations, Simulation, Provider SDKs, ecosystem
frameworks, or gateways. Changes to protected artifacts, IR, public schemas, or
Stable Core APIs require an integration contract/ADR change before implementation.
