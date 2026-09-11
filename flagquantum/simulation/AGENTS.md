# Simulation Team Boundary

Simulation owns statevector, MPS, tensor-network, noise, and differentiation
algorithms. It consumes Core-owned contracts and must not own cluster policy,
credentials, durable jobs, concrete vendor selection, or protocol gateways.
