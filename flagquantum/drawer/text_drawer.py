"""
Text mode circuit drawer
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .ir_adapter import to_drawable_circuit


@dataclass
class _CurrentTotals:
    """Accumulated circuit strings"""

    finished_lines: List[str]  # Completed lines (used when wrapping)
    wire_totals: List[str]  # Accumulated quantum wire strings
    bit_totals: List[str]  # Accumulated classical bit strings


@dataclass
class _Config:
    """Drawing configuration"""

    wire_map: Dict[Any, int]  # Wire label -> display position
    wire_order: List[Any]  # Wire order (top to bottom)
    num_op_layers: int  # Number of operation layers
    cur_layer: int = -1  # Current layer index
    decimals: Optional[int] = None  # Parameter precision
    show_wire_labels: bool = True  # Whether to show wire labels

    @property
    def wire_filler(self) -> str:
        """Filler character: '─' for operation layers, space for measurement layers"""
        return "─" if self.cur_layer < self.num_op_layers else " "

    @property
    def n_wires(self) -> int:
        return len(self.wire_map)


class TextDrawer:
    """
    Text circuit drawer

    Core mechanism:
    1. Maintain totals.wire_totals as accumulated strings
    2. Each layer is connected via filler.join([t, s])
    3. _left_justify ensures all wires have equal length
    """

    # Gate name to symbol mapping
    GATE_SYMBOLS = {
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
        wire_order: Sequence[int | str] | None = None,
        show_all_wires: bool = False,
        decimals: int | None = 3,
        max_length: int = 100,
        show_initial_state: bool = False,
    ) -> None:
        qdev = to_drawable_circuit(qdev)
        self.qdev = qdev
        self.op_history: Sequence[dict[str, Any]] = getattr(qdev, "op_history", ())
        self.n_wires: int = (
            qdev.n_wires if hasattr(qdev, "n_wires") else self._detect_n_wires()
        )
        self.decimals = decimals
        self.max_length = max_length
        self.show_all_wires = show_all_wires
        self.show_initial_state = show_initial_state

        # Determine wire order
        self.wire_order = self._create_wire_order(wire_order)
        self.wire_map = {wire: idx for idx, wire in enumerate(self.wire_order)}
        self.reverse_wire_map = {idx: wire for wire, idx in self.wire_map.items()}

        # Create layers
        self.layers = self._create_layers()
        self.num_op_layers = len(self.layers)

    def _detect_n_wires(self) -> int:
        """Automatically detect the number of qubits"""
        max_wire = -1
        for op in self.op_history:
            wires = op.get("wires", [])
            if isinstance(wires, int):
                wires = [wires]
            for w in wires:
                if isinstance(w, int) and w > max_wire:
                    max_wire = w
        return max_wire + 1 if max_wire >= 0 else 0

    def _create_wire_order(
        self, wire_order: Sequence[int | str] | None
    ) -> list[int | str]:
        """Create wire order"""
        if wire_order is None:
            wire_order = list(range(self.n_wires))

        if not self.show_all_wires:
            used_wires = set()
            for op in self.op_history:
                wires = op.get("wires", [])
                if isinstance(wires, int):
                    wires = [wires]
                used_wires.update(wires)
            wire_order = [w for w in wire_order if w in used_wires]

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

        Multi-qubit gates occupy all wires in between to prevent overlap with other operations
        """
        last_layer: dict[int, int] = {}  # wire -> last_layer_index
        layers: list[list[dict[str, Any]]] = []

        for op in self.op_history:
            wires = op.get("wires", [])
            if isinstance(wires, int):
                wires = [wires]

            if not wires:
                # Operations without wires (e.g., global operations) occupy all wires
                wires = list(range(self.n_wires))

            # Get all wires occupied by this operation (including intermediate wires)
            min_wire = min(wires)
            max_wire = max(wires)
            occupied_wires = set(range(min_wire, max_wire + 1))

            # Map to display wire indices
            mapped_occupied = set()
            for w in occupied_wires:
                if w in self.wire_map:
                    mapped_occupied.add(self.wire_map[w])

            if not mapped_occupied:
                continue

            # Find the maximum layer index among the last layers of these wires
            max_layer = -1
            for w in mapped_occupied:
                if w in last_layer:
                    max_layer = max(max_layer, last_layer[w])

            new_layer = max_layer + 1

            # Ensure the layer list is long enough
            while len(layers) <= new_layer:
                layers.append([])

            layers[new_layer].append(op)

            # Update the last layer for these wires
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
        elif p is not None:
            return f"{p}"
        return ""

    def _render_single_gate(self, name: str, params: object) -> str:
        """Render single-qubit gate: ─RX(0.31)─"""
        symbol = self.GATE_SYMBOLS.get(name.lower(), name.upper() if name else "?")
        param_str = self._get_param_str(params)
        if param_str:
            return f"─{symbol}({param_str})─"
        return f"─{symbol}─"

    def _render_toffoli(self, wires: List[int]) -> list[tuple[int, str]]:
        """
        Render Toffoli (CCX) gate

        Format:
        Control line 1: ╭●
        Control line 2: ├●
        Target line:    ╰X
        """
        mapped_wires = [self.wire_map[w] for w in wires if w in self.wire_map]
        if not mapped_wires:
            return []

        min_wire = min(mapped_wires)
        max_wire = max(mapped_wires)
        n_lines = max_wire - min_wire + 1

        lines = []
        for i in range(n_lines):
            abs_idx = min_wire + i

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

    def _render_cswap(self, wires: List[int]) -> list[tuple[int, str]]:
        """
        Render Fredkin (CSWAP) gate

        Format:
        Control line:   ╭●
        Target line 1:  ├╳
        Target line 2:  ╰╳
        """
        mapped_wires = [self.wire_map[w] for w in wires if w in self.wire_map]
        if not mapped_wires:
            return []

        min_wire = min(mapped_wires)
        max_wire = max(mapped_wires)
        n_lines = max_wire - min_wire + 1

        lines = []
        for i in range(n_lines):
            abs_idx = min_wire + i

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

    def _render_swap_gate(self, wires: List[int]) -> list[tuple[int, str]]:
        """
        Render SWAP gate (avoid misleading: draw only ends, connectors in between)

        Adjacent wires SWAP(wires=[0,1]):
            0: ╳
            1: ╳

        Non-adjacent wires SWAP(wires=[0,2]):
            0: ╳
            1: │   (just a connector)
            2: ╳
        """
        mapped_wires = [self.wire_map[w] for w in wires if w in self.wire_map]
        if not mapped_wires:
            return []

        min_wire = min(mapped_wires)
        max_wire = max(mapped_wires)
        n_lines = max_wire - min_wire + 1

        lines = []
        for i in range(n_lines):
            abs_idx = min_wire + i

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
        mapped_wires = [self.wire_map[w] for w in wires if w in self.wire_map]
        if not mapped_wires:
            return []

        min_wire = min(mapped_wires)
        max_wire = max(mapped_wires)
        n_lines = max_wire - min_wire + 1

        lines = []
        for i in range(n_lines):
            abs_idx = min_wire + i

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
        mapped_wires = [self.wire_map[w] for w in wires if w in self.wire_map]
        if not mapped_wires:
            return []

        min_wire = min(mapped_wires)
        max_wire = max(mapped_wires)
        n_lines = max_wire - min_wire + 1

        param_str = self._get_param_str(params)
        if param_str:
            label = f"{gate_name}({param_str})"
        else:
            label = gate_name

        lines = []
        for i in range(n_lines):
            abs_idx = min_wire + i

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
        mapped_wires = [self.wire_map[w] for w in wires if w in self.wire_map]
        if not mapped_wires:
            return []

        min_wire = min(mapped_wires)
        max_wire = max(mapped_wires)
        n_lines = max_wire - min_wire + 1

        param_str = self._get_param_str(params)
        if param_str:
            label = f"{gate_name}({param_str})"
        else:
            label = gate_name

        lines = []
        for i in range(n_lines):
            abs_idx = min_wire + i

            if i == 0:
                lines.append((abs_idx, f"╭{label}"))
            elif i == n_lines - 1:
                lines.append((abs_idx, f"╰{label}"))
            else:
                lines.append((abs_idx, f"├{label}"))

        return lines

    def _render_multi_gate(
        self, name: str, wires: List[int], params: object
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

        mapped_wires = [self.wire_map[w] for w in wires if w in self.wire_map]
        if not mapped_wires:
            return []

        min_wire = min(mapped_wires)
        max_wire = max(mapped_wires)
        n_lines = max_wire - min_wire + 1

        lines = []
        for i in range(n_lines):
            abs_idx = min_wire + i

            if i == 0:
                lines.append((abs_idx, f"╭{label}"))
            elif i == n_lines - 1:
                lines.append((abs_idx, f"╰{label}"))
            else:
                lines.append((abs_idx, f"├{label}"))

        return lines

    def _render_op(self, op: dict[str, Any]) -> list[tuple[int, str]]:
        """
        Render a single operation, returning [(wire_index, string), ...]

        Note: Only the content to be added is returned here, not including filler characters
        Filler characters are handled by _initialize_layer_str and _left_justify
        """
        name = op.get("name_or_mat", "").lower()
        wires = op.get("wires", [])
        params = op.get("params", [])

        if isinstance(wires, int):
            wires = [wires]

        # Skip measurement operations (handled uniformly in _finalize_layers)
        if name == "measure_allz":
            return []

        if not wires:
            return []

        # Filter existing wires
        valid_wires = [w for w in wires if w in self.wire_map]
        if not valid_wires:
            return []

        # Single-qubit gate
        if len(valid_wires) == 1:
            wire_idx = self.wire_map[valid_wires[0]]
            return [(wire_idx, self._render_single_gate(name, params))]

        # Toffoli (CCX) - 3 qubits
        if name in ["ccx", "toffoli"] and len(valid_wires) == 3:
            return self._render_toffoli(valid_wires)

        # Fredkin (CSWAP) - 3 qubits
        if name in ["cswap", "fredkin"] and len(valid_wires) == 3:
            return self._render_cswap(valid_wires)

        # CZ gate (controlled-Z)
        if name == "cz":
            return self._render_controlled_gate(valid_wires, "Z")

        # CNOT/CX
        if name in ["cx", "cnot"]:
            return self._render_controlled_gate(valid_wires, "X")

        # CY gate (controlled-Y)
        if name == "cy":
            return self._render_controlled_gate(valid_wires, "Y")

        # CPhase gate
        if name in ["cphase", "controlledphase"]:
            return self._render_controlled_gate_with_param(valid_wires, "CP", params)

        # CRX, CRY, CRZ controlled rotation gates
        if name == "crx":
            return self._render_controlled_gate_with_param(valid_wires, "CRX", params)
        if name == "cry":
            return self._render_controlled_gate_with_param(valid_wires, "CRY", params)
        if name == "crz":
            return self._render_controlled_gate_with_param(valid_wires, "CRZ", params)

        # Ising gates (RXX, RYY, RZZ)
        if name in ["rxx", "ryy", "rzz"]:
            return self._render_ising_gate(valid_wires, name.upper(), params)

        # SWAP gate
        if name == "swap":
            return self._render_swap_gate(valid_wires)

        # Default multi-qubit gate
        return self._render_multi_gate(name, valid_wires, params)

    def _initialize_layer_str(self, config: _Config) -> List[str]:
        """Initialize the string array for a new layer"""
        return [config.wire_filler] * config.n_wires

    def _left_justify(self, layer_str: List[str], config: _Config) -> List[str]:
        """Pad all wires in this layer to the same length"""
        if not layer_str:
            return layer_str

        max_label_len = max(len(s) for s in layer_str)

        for w in range(config.n_wires):
            layer_str[w] = layer_str[w].ljust(max_label_len, config.wire_filler)

        return layer_str

    def _add_layer_str_to_totals(
        self, totals: _CurrentTotals, layer_str: List[str], config: _Config
    ) -> _CurrentTotals:
        """Merge the current layer into accumulated strings"""
        totals.wire_totals = [
            config.wire_filler.join([t, s])
            for t, s in zip(totals.wire_totals, layer_str[: config.n_wires])
        ]

        return totals

    def _add_to_finished_lines(
        self, totals: _CurrentTotals, config: _Config, add_measurement: bool = False
    ) -> _CurrentTotals:
        """Save current line to finished_lines when exceeding max_length and start a new line"""
        suffix = " ···"

        saved_lines = [line + suffix for line in totals.wire_totals]

        totals.finished_lines += saved_lines
        totals.finished_lines[-1] += "\n"

        # Reset totals (new line)
        prefix = "··· "

        if config.show_wire_labels:
            totals.wire_totals = [f"{wire}: " + prefix for wire in config.wire_order]
            line_length = max(len(s) for s in totals.wire_totals)
            totals.wire_totals = [s.rjust(line_length, " ") for s in totals.wire_totals]
        else:
            totals.wire_totals = [prefix] * config.n_wires

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
            # All wires are measured
            for i in range(len(totals.wire_totals)):
                totals.wire_totals[i] = f"{totals.wire_totals[i]}─┤  <Z>"
        else:
            # Check measurement per wire
            for i, wire in enumerate(config.wire_order):
                has_measure = any(
                    op.get("name_or_mat", "").lower()
                    in ["measure", "measurement", "measurez"]
                    and (
                        wire in op.get("wires", [])
                        if isinstance(op.get("wires"), list)
                        else op.get("wires") == wire
                    )
                    for op in self.op_history
                )
                if has_measure:
                    totals.wire_totals[i] = f"{totals.wire_totals[i]}─┤  <Z>"
                else:
                    totals.wire_totals[i] = f"{totals.wire_totals[i]}─┤"

        return totals

    def _initialize_wire_totals(self, config: _Config) -> List[str]:
        """Initialize wire_totals (include wire labels, optionally show initial state)"""
        if config.show_wire_labels:
            # Check whether to show initial state (can be controlled via attribute)
            show_initial_state = getattr(self, "show_initial_state", False)

            if show_initial_state:
                wire_totals = [f"{wire}: |0⟩ " for wire in config.wire_order]
            else:
                wire_totals = [f"{wire}: " for wire in config.wire_order]

            # Right-align wire labels (make all wire labels equal width)
            line_length = max(len(s) for s in wire_totals)
            wire_totals = [s.rjust(line_length, " ") for s in wire_totals]
        else:
            wire_totals = [""] * config.n_wires

        return wire_totals

    def draw(self) -> str:
        """Draw the circuit diagram (supports automatic line wrapping)"""
        if not self.op_history:
            return "Empty circuit"

        if not self.wire_order:
            return "No wires"

        config = _Config(
            wire_map=self.wire_map,
            wire_order=self.wire_order,
            num_op_layers=self.num_op_layers,
            decimals=self.decimals,
            show_wire_labels=True,
        )

        # Initialize accumulator (include wire labels)
        wire_totals = self._initialize_wire_totals(config)

        totals = _CurrentTotals(
            finished_lines=[], wire_totals=wire_totals, bit_totals=[]
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
                for wire_idx, s in rendered:
                    layer_str[wire_idx] += s

            # Left justify (pad to same length)
            layer_str = self._left_justify(layer_str, config)

            # Check if line wrap is needed
            is_last_layer = layer_idx == len(self.layers) - 1
            cur_max_length = (
                self.max_length - len_suffix if not is_last_layer else self.max_length
            )

            if (
                totals.wire_totals
                and len(totals.wire_totals[0]) + len(layer_str[0]) > cur_max_length - 1
            ):
                totals = self._add_to_finished_lines(
                    totals, config, add_measurement=False
                )

            # Merge into accumulated strings
            totals = self._add_layer_str_to_totals(totals, layer_str, config)

        # After processing all layers, add measurement markers at the end
        totals = self._finalize_layers(totals, config)

        # Merge final results
        result_lines = totals.finished_lines + totals.wire_totals + totals.bit_totals

        # Filter empty lines
        result_lines = [line for line in result_lines if line.strip() or line == "\n"]

        return "\n".join(result_lines)


def draw_text(
    qdev: object,
    wire_order: Sequence[int | str] | None = None,
    show_all_wires: bool = False,
    decimals: int | None = 3,
    max_length: int = 100,
    show_initial_state: bool = False,
) -> str:
    """
    Draw a text circuit diagram

    Args:
        qdev: FlagQuantum Circuit, CircuitIR, or device object
        wire_order: Wire order (top to bottom), e.g., [0, 1, 2, 3] or ["q0", "q1"]
        show_all_wires: Whether to show all wires (including unused ones)
        decimals: Precision for parameter display
        max_length: Maximum width per line
        show_initial_state: Whether to show initial state (e.g., "0: |0⟩")

    Returns:
        str: Circuit diagram string
    """
    drawer = TextDrawer(
        qdev, wire_order, show_all_wires, decimals, max_length, show_initial_state
    )
    return drawer.draw()
