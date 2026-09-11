# Services boundary

Keep this package limited to reusable application-level composition. A function
belongs here only when it combines multiple stable APIs into one coherent
workflow with additional validation, policy, or structured failure handling.

Do not add pass-through wrappers, protocol request models, MCP/REST/CLI code,
LLM logic, vendor SDK calls, numerical kernels, persistence, or a general service
manager. Simple compile, plan, run, and result operations use their owning public
APIs directly.
