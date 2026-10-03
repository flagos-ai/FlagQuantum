"""
Text mode circuit drawer
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from .ir_adapter import to_drawable_circuit


@dataclass
class _CurrentTotals:
    """Accumulated circuit strings"""

    finished_lines: list[str]  # Completed lines (used when wrapping)
    qubit_totals: list[str]  # Accumulated quantum wire strings
    bit_totals: list[str]  # Accumulated classical bit strings


@dataclass
class _Config:
    """Drawing configuration"""

    qubit_map: dict[Any, int]  # Wire label -> display position
    qubit_order: list[Any]  # Wire order (top to bottom)
    num_op_layers: int  # Number of operation layers
    cur_layer: int = -1  # Current layer index
    decimals: Optional[int] = None  # Parameter precision
    show_qubit_labels: bool = True  # Whether to show wire labels

    @property
    def qubit_filler(self) -> str:
        """Filler character: '─' for operation layers, space for measurement layers"""
        return "─" if self.cur_layer < self.num_op_layers else " "

    @property
    def n_qubits(self) -> int:
        return len(self.qubit_map)


class TextDrawer:
    """
    Text circuit drawer

    Core mechanism:
    1. Maintain totals.qubit_totals as accumulated strings
    2. Each layer is connected via filler.join([t, s])
    3. _left_justify ensures all qubit rows have equal length
    """

    # Gate name to symbol mapping
    GATE_SYMBOLS: ClassVar[dict[str, str]] = {
        # Single-qubit gates
        "rx": "RX",
        "ry": "RY",
        "rz": "RZ",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "h": "H",
        "hadamard": "H",
        "s": "S",
        "t": "T",
        "sx": "SX",
        "sdg": "S†",
        "tdg": "T†",
        "sxdg": "SX†",
        "p": "P",
        "phase": "P",
        "u1": "U1",
        "u2": "U2",
        "u3": "U3",
        # Two-qubit gates
        "cx": "●",
        "cnot": "●",
        "cy": "●",
        "cz": "●",
        "swap": "SWAP",
        "cphase": "P",
        "crx": "CRX",
        "cry": "CRY",
        "crz": "CRZ",
        "rxx": "RXX",
        "ryy": "RYY",
        "rzz": "RZZ",
        # Three-qubit gates
        "ccx": "Toffoli",
        "toffoli": "Toffoli",
        "cswap": "CSWAP",
        "fredkin": "CSWAP",
        # Measurements
        "measure": "Meas",
        "measurement": "M",
        "measure_allz": "MZ",
    }

    def __init__(
        self,
        qdev: object,
        qubit_order: Sequence[int | str] | None = None,
        show_all_qubits: bool = False,
        decimals: int | None = 3,
        max_length: int = 100,
        show_initial_state: bool = False,
    ) -> None:
        qdev = to_drawable_circuit(qdev)
        self.qdev = qdev
        self.op_history: Sequence[dict[str, Any]] = getattr(qdev, "op_history", ())
        self.n_qubits: int = (
            qdev.n_qubits if hasattr(qdev, "n_qubits") else self._detect_n_qubits()
        )
        self.decimals = decimals
        self.max_length = max_length
        self.show_all_qubits = show_all_qubits
        self.show_initial_state = show_initial_state

        # Determine qubit order
        self.qubit_order = self._create_qubit_order(qubit_order)
        self.qubit_map = {qubit: idx for idx, qubit in enumerate(self.qubit_order)}
        self.reverse_qubit_map = {idx: qubit for qubit, idx in self.qubit_map.items()}

        # Create layers
        self.layers = self._create_layers()
        self.num_op_layers = len(self.layers)

    def _detect_n_qubits(self) -> int:
        """Automatically detect the number of qubits"""
        max_qubit = -1
        for op in self.op_history:
            qubits = op.get("qubits", [])
            if isinstance(qubits, int):
                qubits = [qubits]
            for w in qubits:
                if isinstance(w, int) and w > max_qubit:
                    max_qubit = w
        return max_qubit + 1 if max_qubit >= 0 else 0

    def _create_qubit_order(
        self, wire_order: Sequence[int | str] | None
    ) -> list[int | str]:
        """Create qubit order"""
        if wire_order is None:
            wire_order = list(range(self.n_qubits))

        if not self.show_all_qubits:
            used_qubits = set()
            for op in self.op_history:
                qubits = op.get("qubits", [])
                if isinstance(qubits, int):
                    qubits = [qubits]
                used_qubits.update(qubits)
            wire_order = [w for w in wire_order if w in used_qubits]

        return list(wire_order)

    def _is_parameterized_gate(self, name: str) -> bool:
        """Check if the gate has trainable parameters"""
        return name in [
            "rx",
            "ry",
            "rz",
            "p",
            "phase",
            "u1",
            "u2",
            "u3",
            "crx",
            "cry",
            "crz",
            "cphase",
            "rxx",
            "ryy",
            "rzz",
        ]

    def _create_layers(self) -> list[list[dict[str, Any]]]:
        """
        Layering algorithm

        Multi-qubit gates occupy all qubits in between to prevent overlap with other operations
        """
        last_layer: dict[int, int] = {}  # wire -> last_layer_index
        layers: list[list[dict[str, Any]]] = []

        for op in self.op_history:
            qubits = op.get("qubits", [])
            if isinstance(qubits, int):
                qubits = [qubits]

            if not qubits:
                # Operations without qubits (e.g., global operations) occupy all qubits
                qubits = list(range(self.n_qubits))

            # Get all qubits occupied by this operation (including intermediate qubits)
            min_qubit = min(qubits)
            max_qubit = max(qubits)
            occupied_qubits = set(range(min_qubit, max_qubit + 1))

            # Map to display qubit indices
            mapped_occupied = set()
            for w in occupied_qubits:
                if w in self.qubit_map:
                    mapped_occupied.add(self.qubit_map[w])

            if not mapped_occupied:
                continue

            # Find the maximum layer index among the last layers of these qubits
            max_layer = -1
            for w in mapped_occupied:
                if w in last_layer:
                    max_layer = max(max_layer, last_layer[w])

            new_layer = max_layer + 1

            # Ensure the layer list is long enough
            while len(layers) <= new_layer:
                layers.append([])

            layers[new_layer].append(op)

            # Update the last layer for these qubits
            for w in mapped_occupied:
                last_layer[w] = new_layer

        return layers

    def _get_param_str(self, params: object) -> str:
        """Format parameter string (fixed decimal places, preserve trailing zeros)"""
        if self.decimals is None or params is None:
            return ""

        if isinstance(params, list):
            if params == []:
                return ""
            p = params[0]
        else:
            p = params

        if isinstance(p, (int, float)):
            # Fixed decimal places, preserve trailing zeros
            return f"{p:.{self.decimals}f}"
        if p is not None:
            return f"{p}"
        return ""

    def _render_single_gate(self, name: str, params: object) -> str:
        """Render single-qubit gate: ─RX(0.31)─"""
        symbol = self.GATE_SYMBOLS.get(name.lower(), name.upper() if name else "?")
        param_str = self._get_param_str(params)
        if param_str:
            return f"─{symbol}({param_str})─"
        return f"─{symbol}─"

    def _render_toffoli(self, wires: list[int]) -> list[tuple[int, str]]:
        """
        Render Toffoli (CCX) gate

        Format:
        Control line 1: ╭●
        Control line 2: ├●
        Target line:    ╰X
        """
        mapped_qubits = [self.qubit_map[w] for w in wires if w in self.qubit_map]
        if not mapped_qubits:
            return []

        min_qubit = min(mapped_qubits)
        max_qubit = max(mapped_qubits)
        n_lines = max_qubit - min_qubit + 1

        lines = []
        for i in range(n_lines):
            abs_idx = min_qubit + i

            if i == 0:
                # First control line
                lines.append((abs_idx, "╭●"))
            elif i == n_lines - 2:
                # Second control line (for 3 lines, this is the middle line)
                lines.append((abs_idx, "├●"))
            elif i == n_lines - 1:
                # Target line
                lines.append((abs_idx, "╰X"))
            else:
                # Other intermediate lines: connector
                lines.append((abs_idx, "│"))

        return lines

    def _render_cswap(self, wires: list[int]) -> list[tuple[int, str]]:
        """
        Render Fredkin (CSWAP) gate

        Format:
        Control line:   ╭●
        Target line 1:  ├╳
        Target line 2:  ╰╳
        """
        mapped_qubits = [self.qubit_map[w] for w in wires if w in self.qubit_map]
        if not mapped_qubits:
            return []

        min_qubit = min(mapped_qubits)
        max_qubit = max(mapped_qubits)
        n_lines = max_qubit - min_qubit + 1

        lines = []
        for i in range(n_lines):
            abs_idx = min_qubit + i

            if i == 0:
                # Control line
                lines.append((abs_idx, "╭●"))
            elif i == n_lines - 2:
                # First target line
                lines.append((abs_idx, "├╳"))
            elif i == n_lines - 1:
                # Second target line
                lines.append((abs_idx, "╰╳"))
            else:
                # Other intermediate lines: connector
                lines.append((abs_idx, "│"))

        return lines

    def _render_swap_gate(self, wires: list[int]) -> list[tuple[int, str]]:
        """
        Render SWAP gate (avoid misleading: draw only ends, connectors in between)

        Adjacent qubits SWAP(qubits=[0,1]):
            0: ╳
            1: ╳

        Non-adjacent qubits SWAP(qubits=[0,2]):
            0: ╳
            1: │   (just a connector)
            2: ╳
        """
        mapped_qubits = [self.qubit_map[w] for w in wires if w in self.qubit_map]
        if not mapped_qubits:
            return []

        min_qubit = min(mapped_qubits)
        max_qubit = max(mapped_qubits)
        n_lines = max_qubit - min_qubit + 1

        lines = []
        for i in range(n_lines):
            abs_idx = min_qubit + i

            if i == 0 or i == n_lines - 1:
                # Ends: draw X
                lines.append((abs_idx, "╳"))
            else:
                # Middle lines: draw only connector
                lines.append((abs_idx, "│"))

        return lines

    def _render_controlled_gate(
        self, wires: Sequence[int], target_symbol: str
    ) -> list[tuple[int, str]]:
        """Render controlled gate (CZ, CY, CX, etc.)"""
        mapped_qubits = [self.qubit_map[w] for w in wires if w in self.qubit_map]
        if not mapped_qubits:
            return []

        min_qubit = min(mapped_qubits)
        max_qubit = max(mapped_qubits)
        n_lines = max_qubit - min_qubit + 1

        lines = []
        for i in range(n_lines):
            abs_idx = min_qubit + i

            if i == 0:
                # Top: control line
                lines.append((abs_idx, "╭●"))
            elif i == n_lines - 1:
                # Bottom: target line (display Z, X, or Y)
                lines.append((abs_idx, f"╰{target_symbol}"))
            else:
                # Middle lines: connector
                lines.append((abs_idx, "│"))

        return lines

    def _render_controlled_gate_with_param(
        self, wires: Sequence[int], gate_name: str, params: object
    ) -> list[tuple[int, str]]:
        """Render parameterized controlled gate (CRX, CRY, CRZ)"""
        mapped_qubits = [self.qubit_map[w] for w in wires if w in self.qubit_map]
        if not mapped_qubits:
            return []

        min_qubit = min(mapped_qubits)
        max_qubit = max(mapped_qubits)
        n_lines = max_qubit - min_qubit + 1

        param_str = self._get_param_str(params)
        if param_str:
            label = f"{gate_name}({param_str})"
        else:
            label = gate_name

        lines = []
        for i in range(n_lines):
            abs_idx = min_qubit + i

            if i == 0:
                lines.append((abs_idx, "╭●"))
            elif i == n_lines - 1:
                lines.append((abs_idx, f"╰{label}"))
            else:
                lines.append((abs_idx, "│"))

        return lines

    def _render_ising_gate(
        self, wires: Sequence[int], gate_name: str, params: object
    ) -> list[tuple[int, str]]:
        """Render Ising gate (RXX, RYY, RZZ)"""
        mapped_qubits = [self.qubit_map[w] for w in wires if w in self.qubit_map]
        if not mapped_qubits:
            return []

        min_qubit = min(mapped_qubits)
        max_qubit = max(mapped_qubits)
        n_lines = max_qubit - min_qubit + 1

        param_str = self._get_param_str(params)
        if param_str:
            label = f"{gate_name}({param_str})"
        else:
            label = gate_name

        lines = []
        for i in range(n_lines):
            abs_idx = min_qubit + i

            if i == 0:
                lines.append((abs_idx, f"╭{label}"))
            elif i == n_lines - 1:
                lines.append((abs_idx, f"╰{label}"))
            else:
                lines.append((abs_idx, f"├{label}"))

        return lines

    def _render_multi_gate(
        self, name: str, wires: list[int], params: object
    ) -> list[tuple[int, str]]:
        """
        Render multi-qubit gate (e.g., QFT)

        Format:
        Top:    ╭QFT─
        Middle: ├QFT─
        Bottom: ╰QFT─
        """
        symbol = self.GATE_SYMBOLS.get(name.lower(), name.upper() if name else "?")
        param_str = self._get_param_str(params)

        if param_str:
            label = f"{symbol}({param_str})"
        else:
            label = symbol

        mapped_qubits = [self.qubit_map[w] for w in wires if w in self.qubit_map]
        if not mapped_qubits:
            return []

        min_qubit = min(mapped_qubits)
        max_qubit = max(mapped_qubits)
        n_lines = max_qubit - min_qubit + 1

        lines = []
        for i in range(n_lines):
            abs_idx = min_qubit + i

            if i == 0:
                lines.append((abs_idx, f"╭{label}"))
            elif i == n_lines - 1:
                lines.append((abs_idx, f"╰{label}"))
            else:
                lines.append((abs_idx, f"├{label}"))

        return lines

    def _render_op(self, op: dict[str, Any]) -> list[tuple[int, str]]:
        """
        Render a single operation, returning [(qubit_index, string), ...]

        Note: Only the content to be added is returned here, not including filler characters
        Filler characters are handled by _initialize_layer_str and _left_justify
        """
        name = op.get("name_or_mat", "").lower()
        qubits = op.get("qubits", [])
        params = op.get("params", [])

        if isinstance(qubits, int):
            qubits = [qubits]

        # Skip measurement operations (handled uniformly in _finalize_layers)
        if name == "measure_allz":
            return []

        if not qubits:
            return []

        # Filter existing qubits
        valid_qubits = [w for w in qubits if w in self.qubit_map]
        if not valid_qubits:
            return []

        # Single-qubit gate
        if len(valid_qubits) == 1:
            qubit_idx = self.qubit_map[valid_qubits[0]]
            return [(qubit_idx, self._render_single_gate(name, params))]

        # Toffoli (CCX) - 3 qubits
        if name in ["ccx", "toffoli"] and len(valid_qubits) == 3:
            return self._render_toffoli(valid_qubits)

        # Fredkin (CSWAP) - 3 qubits
        if name in ["cswap", "fredkin"] and len(valid_qubits) == 3:
            return self._render_cswap(valid_qubits)

        # CZ gate (controlled-Z)
        if name == "cz":
            return self._render_controlled_gate(valid_qubits, "Z")

        # CNOT/CX
        if name in ["cx", "cnot"]:
            return self._render_controlled_gate(valid_qubits, "X")

        # CY gate (controlled-Y)
        if name == "cy":
            return self._render_controlled_gate(valid_qubits, "Y")

        # CPhase gate
        if name in ["cphase", "controlledphase"]:
            return self._render_controlled_gate_with_param(valid_qubits, "CP", params)

        # CRX, CRY, CRZ controlled rotation gates
        if name == "crx":
            return self._render_controlled_gate_with_param(valid_qubits, "CRX", params)
        if name == "cry":
            return self._render_controlled_gate_with_param(valid_qubits, "CRY", params)
        if name == "crz":
            return self._render_controlled_gate_with_param(valid_qubits, "CRZ", params)

        # Ising gates (RXX, RYY, RZZ)
        if name in ["rxx", "ryy", "rzz"]:
            return self._render_ising_gate(valid_qubits, name.upper(), params)

        # SWAP gate
        if name == "swap":
            return self._render_swap_gate(valid_qubits)

        # Default multi-qubit gate
        return self._render_multi_gate(name, valid_qubits, params)

    def _initialize_layer_str(self, config: _Config) -> list[str]:
        """Initialize the string array for a new layer"""
        return [config.qubit_filler] * config.n_qubits

    def _left_justify(self, layer_str: list[str], config: _Config) -> list[str]:
        """Pad all qubits in this layer to the same length"""
        if not layer_str:
            return layer_str

        max_label_len = max(len(s) for s in layer_str)

        for w in range(config.n_qubits):
            layer_str[w] = layer_str[w].ljust(max_label_len, config.qubit_filler)

        return layer_str

    def _add_layer_str_to_totals(
        self, totals: _CurrentTotals, layer_str: list[str], config: _Config
    ) -> _CurrentTotals:
        """Merge the current layer into accumulated strings"""
        totals.qubit_totals = [
            config.qubit_filler.join([t, s])
            for t, s in zip(
                totals.qubit_totals, layer_str[: config.n_qubits], strict=True
            )
        ]

        return totals

    def _add_to_finished_lines(
        self, totals: _CurrentTotals, config: _Config, add_measurement: bool = False
    ) -> _CurrentTotals:
        """Save current line to finished_lines when exceeding max_length and start a new line"""
        suffix = " ···"

        saved_lines = [line + suffix for line in totals.qubit_totals]

        totals.finished_lines += saved_lines
        totals.finished_lines[-1] += "\n"

        # Reset totals (new line)
        prefix = "··· "

        if config.show_qubit_labels:
            totals.qubit_totals = [
                f"{qubit}: " + prefix for qubit in config.qubit_order
            ]
            line_length = max(len(s) for s in totals.qubit_totals)
            totals.qubit_totals = [
                s.rjust(line_length, " ") for s in totals.qubit_totals
            ]
        else:
            totals.qubit_totals = [prefix] * config.n_qubits

        return totals

    def _finalize_layers(
        self, totals: _CurrentTotals, config: _Config
    ) -> _CurrentTotals:
        """Add end-of-line markers (measurement symbols)"""

        # Check if there is a measure_allZ operation
        has_all_measure = any(
            op.get("name_or_mat", "").lower() == "measure_allz"
            for op in self.op_history
        )

        if has_all_measure:
            # All qubits are measured
            for i in range(len(totals.qubit_totals)):
                totals.qubit_totals[i] = f"{totals.qubit_totals[i]}─┤  <Z>"
        else:
            # Check measurement per qubit
            for i, qubit in enumerate(config.qubit_order):
                has_measure = any(
                    op.get("name_or_mat", "").lower()
                    in ["measure", "measurement", "measurez"]
                    and (
                        qubit in op.get("qubits", [])
                        if isinstance(op.get("qubits"), list)
                        else op.get("qubits") == qubit
                    )
                    for op in self.op_history
                )
                if has_measure:
                    totals.qubit_totals[i] = f"{totals.qubit_totals[i]}─┤  <Z>"
                else:
                    totals.qubit_totals[i] = f"{totals.qubit_totals[i]}─┤"

        return totals

    def _initialize_qubit_totals(self, config: _Config) -> list[str]:
        """Initialize qubit_totals (include qubit labels, optionally show initial state)"""
        if config.show_qubit_labels:
            # Check whether to show initial state (can be controlled via attribute)
            show_initial_state = getattr(self, "show_initial_state", False)

            if show_initial_state:
                qubit_totals = [f"{qubit}: |0⟩ " for qubit in config.qubit_order]
            else:
                qubit_totals = [f"{qubit}: " for qubit in config.qubit_order]

            # Right-align qubit labels (make all qubit labels equal width)
            line_length = max(len(s) for s in qubit_totals)
            qubit_totals = [s.rjust(line_length, " ") for s in qubit_totals]
        else:
            qubit_totals = [""] * config.n_qubits

        return qubit_totals

    def draw(self) -> str:
        """Draw the circuit diagram (supports automatic line wrapping)"""
        if not self.op_history:
            return "Empty circuit"

        if not self.qubit_order:
            return "No qubits"

        config = _Config(
            qubit_map=self.qubit_map,
            qubit_order=self.qubit_order,
            num_op_layers=self.num_op_layers,
            decimals=self.decimals,
            show_qubit_labels=True,
        )

        # Initialize accumulator (include qubit labels)
        qubit_totals = self._initialize_qubit_totals(config)

        totals = _CurrentTotals(
            finished_lines=[], qubit_totals=qubit_totals, bit_totals=[]
        )

        len_suffix = 4  # Length of " ···"

        # Process layer by layer
        for layer_idx, layer in enumerate(self.layers):
            config.cur_layer = layer_idx

            # Initialize this layer
            layer_str = self._initialize_layer_str(config)

            # Add all operations in this layer
            for op in layer:
                rendered = self._render_op(op)
                for qubit_idx, s in rendered:
                    layer_str[qubit_idx] += s

            # Left justify (pad to same length)
            layer_str = self._left_justify(layer_str, config)

            # Check if line wrap is needed
            is_last_layer = layer_idx == len(self.layers) - 1
            cur_max_length = (
                self.max_length - len_suffix if not is_last_layer else self.max_length
            )

            if (
                totals.qubit_totals
                and len(totals.qubit_totals[0]) + len(layer_str[0]) > cur_max_length - 1
            ):
                totals = self._add_to_finished_lines(
                    totals, config, add_measurement=False
                )

            # Merge into accumulated strings
            totals = self._add_layer_str_to_totals(totals, layer_str, config)

        # After processing all layers, add measurement markers at the end
        totals = self._finalize_layers(totals, config)

        # Merge final results
        result_lines = totals.finished_lines + totals.qubit_totals + totals.bit_totals

        # Filter empty lines
        result_lines = [line for line in result_lines if line.strip() or line == "\n"]

        return "\n".join(result_lines)


def draw_text(
    qdev: object,
    qubit_order: Sequence[int | str] | None = None,
    show_all_qubits: bool = False,
    decimals: int | None = 3,
    max_length: int = 100,
    show_initial_state: bool = False,
) -> str:
    """
    Draw a text circuit diagram

    Args:
        qdev: FlagQuantum Circuit, CircuitIR, or device object
        qubit_order: Qubit order (top to bottom), e.g., [0, 1, 2, 3] or ["q0", "q1"]
        show_all_qubits: Whether to show all qubits (including unused ones)
        decimals: Precision for parameter display
        max_length: Maximum width per line
        show_initial_state: Whether to show initial state (e.g., "0: |0⟩")

    Returns:
        str: Circuit diagram string
    """
    drawer = TextDrawer(
        qdev, qubit_order, show_all_qubits, decimals, max_length, show_initial_state
    )
    return drawer.draw()
