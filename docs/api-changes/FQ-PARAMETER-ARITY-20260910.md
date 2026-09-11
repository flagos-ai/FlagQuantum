# Parameter expression operand validation

Parameter expressions support two operands for `add`, `sub`, and `mul`, and
one operand for `neg`. Binding validates the operation and operand count before
resolving parameter values. Malformed expressions raise `ValueError` rather
than ignoring extra operands or exposing an incidental `IndexError` or missing
parameter error. Unsupported operations continue to raise `ValueError`.

Public signatures, valid expression arithmetic, parameter-object key precedence,
tensor identity, and gradient propagation remain unchanged. This is a Core
correctness repair within the authorized repository remediation.

Verification must cover invalid operand counts, unsupported operations, valid
arithmetic, nested binding, missing parameters, and tensor gradients. Existing
snapshot ownership tests must continue to pass. Mapping key typing remediation
is a separate change and is not claimed by this repair.
