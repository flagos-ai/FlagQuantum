"""Parse the bounded declarative VQE portion of a native source file."""

import ast
import re
from dataclasses import dataclass

from ...compiler._hybrid.model import SourceLocation
from .syntax import MAX_BYTES, Token, reject

_CSCOPE = re.compile(r"^cscope[ \t]+```[ \t]*\r?\n([^`]*)\n[ \t]*```[ \t\r\n]*\Z", re.M)
_OPTIONS = {
    "hamiltonian",
    "maximize",
    "initial_point",
    "optimizer",
    "max_iteration",
    "tolerance",
    "step_size",
    "skip_compute_variance",
    "alpha_cvar",
}


@dataclass(frozen=True)
class VqeSettings:
    location: SourceLocation
    operator: str
    initial_point: tuple[float, ...]
    maximize: bool
    max_iteration: int
    tolerance: float
    step_size: float
    skip_compute_variance: bool
    alpha_cvar: float
    saved: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class QaoaWorkflow:
    settings: VqeSettings
    parameter: str
    output: str
    qubits: int
    diagonal: tuple[tuple[int, float], ...]


def split_cscope(source: str, filename: str) -> tuple[str, VqeSettings | None]:
    """Return only checked settings and source text, without executing Python."""
    match = _CSCOPE.search(source) if isinstance(source, str) else None
    if match is None:
        return source, None
    token = Token(
        "cscope", SourceLocation(filename, source.count("\n", 0, match.start()) + 1, 1)
    )
    if len(source.encode("utf-8")) > MAX_BYTES:
        reject(token, "budget.source", f"source exceeds {MAX_BYTES} UTF-8 bytes")
    body = match[1]
    if len(body) > 16384:
        reject(token, "cscope", "classical workflow exceeds the syntax budget")
    try:
        statements = ast.parse(body).body
    except (SyntaxError, ValueError, RecursionError):
        reject(token, "cscope", "malformed classical workflow")
    if len(statements) != 2:
        reject(
            token, "cscope", "workflow requires one VQE assignment and one save call"
        )
    assignment, save = statements
    if (
        not isinstance(assignment, ast.Assign)
        or len(assignment.targets) != 1
        or not isinstance(assignment.targets[0], ast.Name)
        or not isinstance(assignment.value, ast.Call)
        or not isinstance(assignment.value.func, ast.Name)
        or assignment.value.func.id != "vqe"
        or assignment.value.args
    ):
        reject(
            token,
            "cscope",
            "workflow requires a named VQE result with explicit options",
        )
    result = assignment.targets[0].id
    fields = {field.arg: field.value for field in assignment.value.keywords}
    if fields.keys() != _OPTIONS or len(assignment.value.keywords) != len(_OPTIONS):
        reject(token, "cscope", "VQE options must be explicit and unique")
    operator, optimizer = fields["hamiltonian"], fields["optimizer"]
    if (
        not isinstance(operator, ast.Name)
        or result == operator.id
        or not isinstance(optimizer, ast.Attribute)
        or optimizer.attr != "COBYLA"
        or not isinstance(optimizer.value, ast.Name)
        or optimizer.value.id != "Optimizer"
    ):
        reject(
            token,
            "cscope",
            "workflow requires a named Hamiltonian and Optimizer.COBYLA",
        )

    def number(value: ast.AST) -> float:
        negative = isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub)
        if negative:
            assert isinstance(value, ast.UnaryOp)
            value = value.operand
        if (
            not isinstance(value, ast.Constant)
            or not isinstance(value.value, (float, int))
            or isinstance(value.value, bool)
        ):
            reject(token, "cscope", "VQE numeric options require real literals")
        if not -(2**53) <= value.value <= 2**53:
            reject(token, "cscope", "VQE numeric options must be finite and bounded")
        return float(-value.value if negative else value.value)

    def boolean(name: str) -> bool:
        value = fields[name]
        if not isinstance(value, ast.Constant) or not isinstance(value.value, bool):
            reject(token, "cscope", f"{name} must be a Boolean literal")
        return value.value

    start = fields["initial_point"]
    iterations = fields["max_iteration"]
    if (
        not isinstance(start, ast.List)
        or not 2 <= len(start.elts) <= 256
        or len(start.elts) % 2
    ):
        reject(token, "cscope", "initial_point requires 1..128 angle pairs")
    if (
        not isinstance(iterations, ast.Constant)
        or type(iterations.value) is not int
        or not 1 <= iterations.value <= 4096
    ):
        reject(token, "cscope", "max_iteration requires an integer in 1..4096")
    tolerance = number(fields["tolerance"])
    step_size = number(fields["step_size"])
    alpha = number(fields["alpha_cvar"])
    if not (0 < alpha <= 1 and tolerance >= 0 and step_size >= 0):
        reject(
            token,
            "cscope",
            "require alpha_cvar in (0,1] and nonnegative optimizer options",
        )
    if (
        not isinstance(save, ast.Expr)
        or not isinstance(save.value, ast.Call)
        or not isinstance(save.value.func, ast.Name)
        or save.value.func.id != "save"
        or save.value.keywords
        or len(save.value.args) != 1
        or not isinstance(save.value.args[0], ast.Dict)
    ):
        reject(token, "cscope", "save requires one literal result mapping")
    mapping = save.value.args[0]
    saved = []
    for key, value in zip(mapping.keys, mapping.values, strict=True):
        if (
            not isinstance(key, ast.Constant)
            or not isinstance(key.value, str)
            or not key.value
            or not isinstance(value, ast.Name)
            or value.id not in {result, operator.id}
        ):
            reject(
                token,
                "cscope",
                "save may reference only the VQE result and Hamiltonian",
            )
        saved.append((key.value, "result" if value.id == result else "hamiltonian"))
    if (
        len(saved) != 2
        or len(dict(saved)) != 2
        or {kind for _, kind in saved} != {"result", "hamiltonian"}
    ):
        reject(
            token,
            "cscope",
            "save requires distinct keys for the VQE result and Hamiltonian",
        )

    return source[: match.start()], VqeSettings(
        token.location,
        operator.id,
        tuple(number(value) for value in start.elts),
        boolean("maximize"),
        iterations.value,
        tolerance,
        step_size,
        boolean("skip_compute_variance"),
        alpha,
        tuple(saved),
    )
