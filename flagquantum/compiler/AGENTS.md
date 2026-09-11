# Compiler Team Boundary

This directory is the authoritative home for stable program transformation:
validation at compiler entry, optimization, scheduling, target lowering, and
topology routing. It must not select execution backends, manage devices,
schedule jobs, execute numerical kernels, or own Runtime result types.

Do not copy implementations from `_compiler` or `compilation`. A migration is
complete only when the former authority and all of its consumers are removed.
