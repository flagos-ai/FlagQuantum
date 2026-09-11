# Ecosystem Team Boundary

Interop converts external framework and format objects at the boundary. Convert
immediately to FlagQuantum-owned representations; never let external objects
become canonical IR or leak into Core, Compiler, Runtime, or Simulation.

The `extensions/` package owns the optional third-party extension protocol and
lifecycle. Keep it independent from the framework-adapter registry; do not add
another registry or compatibility namespace.
