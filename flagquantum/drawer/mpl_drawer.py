"""
Matplotlib mode circuit drawer
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

from .ir_adapter import to_drawable_circuit


class MPLDrawer:
    """Matplotlib circuit drawer"""

    _box_length = 0.8
    _circ_rad = 0.25
    _ctrl_rad = 0.1
    _swap_dx = 0.2
    _fontsize = 12
    _pad = 0.1
    _boxstyle = f"round, pad={_pad}"

    GATE_COLORS = {
        "single": "#7B9EC2",  # Fixed single-qubit gates (X, Y, Z, H, S, T, etc.) - blue
        "param": "#E15759",  # Parameterized gates (RX, RY, RZ, U1, U2, U3, etc.) - red
        "multi": "#F28E2B",  # Multi-qubit gates (QFT, Toffoli, etc.) - orange
        "ctrl": "#000000",  # Control point - black
        "target": "#E15759",  # Target point - red
        "swap": "#76B7B2",  # SWAP gates - teal
        "measure": "#59A14F",  # Measurement - green
        "cr": "#F28E2B",  # CRX/CRY/CRZ - orange
        "cp": "#76B7B2",  # CPhase - teal
        "p": "#DDA0DD",  # Phase gate - plum
        "ising": "#FFB6C1",  # RXX/RYY/RZZ - light pink
    }

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
        "sdg": "S†",
        "tdg": "T†",
        "sx": "SX",
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
        "cphase": "CPhase",
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
        # Measurement
        "measure": "Meas",
        "measurement": "M",
        "measure_allz": "MZ",
    }

    def __init__(self, qdev, wire_order=None, fig=None, **kwargs):
        qdev = to_drawable_circuit(qdev)
        self.qdev = qdev
        self.op_history = getattr(qdev, "op_history", [])
        self.n_wires = getattr(qdev, "n_wires", self._detect_n_wires())
        self.decimals = kwargs.get("decimals", 2)
        self.active_notches = kwargs.get("active_wire_notches", True)
        self.wire_map = self._create_wire_map(wire_order)
        self.layers = self._create_layers()
        self.n_layers = len(self.layers)
        self._setup_figure(fig, kwargs.get("figsize"))
        self._draw_wires(kwargs.get("wire_options"))
        self._draw_operations()
        self._draw_measurements()
        if kwargs.get("show_wire_labels", True):
            self._draw_labels(kwargs.get("label_options"))
        else:
            self._crop_labels()

    def _detect_n_wires(self):
        """Detect the number of qubits from operation history"""
        max_wire = -1
        for op in self.op_history:
            wires = op.get("wires", [])
            if isinstance(wires, int):
                wires = [wires]
            for w in wires:
                if isinstance(w, int) and w > max_wire:
                    max_wire = w
        return max_wire + 1 if max_wire >= 0 else 0

    def _create_wire_map(self, wire_order):
        """Create a mapping from wire labels to display indices"""
        if wire_order is None:
            wire_order = list(range(self.n_wires))
        for w in range(self.n_wires):
            if w not in wire_order:
                wire_order.append(w)
        return {wire: idx for idx, wire in enumerate(wire_order)}

    def _create_layers(self):
        """
        Layer assignment algorithm

        Ensures that operations in the same layer share the same x-coordinate
        """
        last_op_layer = {}
        layers = []
        for op in self.op_history:
            wires = op.get("wires", [])
            if isinstance(wires, int):
                wires = [wires]
            if not wires:
                continue

            # Get all wires occupied by this operation (including intermediate wires)
            min_w = min(wires)
            max_w = max(wires)
            occupied_wires = set(range(min_w, max_w + 1))

            # Map to display wire indices
            mapped_occupied = [
                self.wire_map[w] for w in occupied_wires if w in self.wire_map
            ]
            if not mapped_occupied:
                continue

            max_layer = max([last_op_layer.get(w, -1) for w in mapped_occupied])
            new_layer = max_layer + 1

            while len(layers) <= new_layer:
                layers.append([])
            layers[new_layer].append(op)

            for w in mapped_occupied:
                last_op_layer[w] = new_layer
        return layers

    def _setup_figure(self, fig, figsize):
        """Initialize the matplotlib figure and axes"""
        if figsize is None:
            figsize = (self.n_layers + 3, self.n_wires + 1)
        self._fig = fig if fig else plt.figure(figsize=figsize)
        self._ax = self._fig.add_axes(
            [0, 0, 1, 1],
            xlim=(-2, self.n_layers + 1),
            ylim=(-1, self.n_wires + 0.5),
            xticks=[],
            yticks=[],
        )
        self._ax.axis("off")
        self._ax.set_facecolor("#F7F7F7")
        self._ax.invert_yaxis()

    def _draw_wires(self, wire_options):
        """Draw horizontal quantum wire lines"""
        opts = wire_options or {}
        for wire_label, idx in self.wire_map.items():
            line = plt.Line2D(
                (-1, self.n_layers),
                (idx, idx),
                color=opts.get("color", "#333333"),
                linewidth=1.2,
                zorder=1,
            )
            self._ax.add_line(line)

    def _get_param_str(self, params):
        """Format parameter string for display"""
        if params is None or self.decimals is None:
            return ""
        if isinstance(params, list) and len(params) == 0:
            return ""
        p = params[0] if isinstance(params, list) else params
        if isinstance(p, (int, float)):
            formatted = f"{p:.{self.decimals}f}"
            formatted = formatted.rstrip("0").rstrip(".")
            return f"({formatted})"
        return f"({p})"

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

    def _draw_operations(self):
        """
        Draw all operations

        Operations in the same layer share the same x-coordinate
        """
        for layer_idx, layer in enumerate(self.layers):
            x = layer_idx
            for op in layer:
                self._draw_operation(op, x)

    def _draw_operation(self, op, x):
        """Draw a single operation at the specified x-coordinate"""
        name = op.get("name_or_mat", "").lower()
        wires = op.get("wires", [])
        params = op.get("params", [])
        if isinstance(wires, int):
            wires = [wires]
        if not wires:
            return

        # Measurement operations are handled separately
        if name == "measure_allz":
            return

        symbol = self.GATE_SYMBOLS.get(name, name.upper() if name else "?")
        param_str = self._get_param_str(params)

        # Build label with parameter if present
        if param_str:
            param_clean = param_str.strip("()")
            label = f"{symbol}\n{param_clean}"
        else:
            label = symbol

        # Determine gate type (special color for parameterized gates)
        is_param_gate = self._is_parameterized_gate(name)
        gate_type = "param" if is_param_gate else "single"

        # Single-qubit gate
        if len(wires) == 1:
            if name in ["rx", "ry", "rz"]:
                self._draw_box(x, wires[0], label, gate_type)
            elif name in ["p", "phase"]:
                self._draw_box(x, wires[0], label, "p")
            else:
                self._draw_box(x, wires[0], label, gate_type)

        # Toffoli (CCX) - 3 qubits
        elif name in ["ccx", "toffoli"] and len(wires) == 3:
            control1, control2, target = wires[0], wires[1], wires[2]
            self._draw_toffoli(x, control1, control2, target)

        # Fredkin (CSWAP) - 3 qubits
        elif name in ["cswap", "fredkin"] and len(wires) == 3:
            control, target1, target2 = wires[0], wires[1], wires[2]
            self._draw_cswap(x, control, target1, target2)

        # CNOT/CX
        elif name in ["cx", "cnot"] and len(wires) >= 2:
            self._draw_controlled_gate(x, wires[0], wires[1], target_symbol="X")

        # CY
        elif name == "cy" and len(wires) >= 2:
            self._draw_controlled_gate(x, wires[0], wires[1], target_symbol="Y")

        # CZ
        elif name == "cz" and len(wires) >= 2:
            self._draw_controlled_gate(x, wires[0], wires[1], target_symbol="Z")

        # CPhase
        elif name in ["cphase", "controlledphase"] and len(wires) >= 2:
            self._draw_controlled_phase_gate_with_param(
                x, wires[0], wires[1], params, "CP"
            )

        # CRX, CRY, CRZ
        elif name == "crx" and len(wires) >= 2:
            self._draw_controlled_gate_with_param(x, wires[0], wires[1], params, "CRX")
        elif name == "cry" and len(wires) >= 2:
            self._draw_controlled_gate_with_param(x, wires[0], wires[1], params, "CRY")
        elif name == "crz" and len(wires) >= 2:
            self._draw_controlled_gate_with_param(x, wires[0], wires[1], params, "CRZ")

        # Ising (RXX, RYY, RZZ)
        elif name in ["rxx", "ryy", "rzz"] and len(wires) >= 2:
            self._draw_ising_gate(x, wires, name.upper(), params)

        # SWAP
        elif name == "swap" and len(wires) >= 2:
            self._draw_swap_gate(x, wires)

        # Multi-qubit gate
        elif len(wires) > 1:
            self._draw_multi_gate(x, wires, label)

    def _draw_controlled_gate_with_param(self, x, control, target, params, gate_name):
        """Draw parameterized controlled gates (CRX, CRY, CRZ)"""
        # Connecting wire
        line = plt.Line2D(
            (x, x), (control, target), color="#333333", linewidth=1.5, zorder=1
        )
        self._ax.add_line(line)

        # Control point
        ctrl_circ = Circle(
            (x, control),
            self._ctrl_rad,
            facecolor=self.GATE_COLORS["ctrl"],
            edgecolor="black",
            linewidth=1,
            zorder=2,
        )
        self._ax.add_patch(ctrl_circ)

        # Parameterized target gate
        param_str = self._get_param_str(params)
        if param_str:
            param_clean = param_str.strip("()")
            label = f"{gate_name}\n{param_clean}"
        else:
            label = gate_name

        # Draw target box with parameters
        self._draw_box(x, target, label, "cr")

    def _draw_controlled_phase_gate_with_param(
        self, x, control, target, params, gate_name
    ):
        """Draw parameterized controlled phase gate (CP)"""
        # Connecting wire
        line = plt.Line2D(
            (x, x), (control, target), color="#333333", linewidth=1.5, zorder=1
        )
        self._ax.add_line(line)

        # Control point
        ctrl_circ = Circle(
            (x, control),
            self._ctrl_rad,
            facecolor=self.GATE_COLORS["ctrl"],
            edgecolor="black",
            linewidth=1,
            zorder=2,
        )
        self._ax.add_patch(ctrl_circ)

        # Parameterized target gate
        param_str = self._get_param_str(params)
        if param_str:
            param_clean = param_str.strip("()")
            label = f"{gate_name}\n{param_clean}"
        else:
            label = gate_name

        # Draw target box with parameters
        self._draw_box(x, target, label, "cp")

    def _draw_ising_gate(self, x, wires, gate_name, params):
        """Draw Ising gates (RXX, RYY, RZZ) - boxes spanning all involved wires"""
        min_w = min(wires)
        max_w = max(wires)
        half = self._box_length / 2
        height = max_w - min_w + self._box_length
        y = min_w - half

        param_str = self._get_param_str(params)
        if param_str:
            param_clean = param_str.strip("()")
            label = f"{gate_name}\n{param_clean}"
        else:
            label = gate_name

        box = FancyBboxPatch(
            (x - half + self._pad, y + self._pad),
            self._box_length - 2 * self._pad,
            height - 2 * self._pad,
            boxstyle=self._boxstyle,
            facecolor=self.GATE_COLORS["ising"],
            edgecolor="#333333",
            linewidth=1.5,
            zorder=2,
        )
        self._ax.add_patch(box)

        center_y = (min_w + max_w) / 2
        self._ax.text(
            x,
            center_y,
            label,
            ha="center",
            va="center",
            fontsize=self._fontsize,
            zorder=3,
            family="sans-serif",
            fontweight="bold",
        )

    def _draw_toffoli(self, x, control1, control2, target):
        """Draw Toffoli gate (CCX)"""
        min_wire = min(control1, control2, target)
        max_wire = max(control1, control2, target)

        line = plt.Line2D(
            (x, x), (min_wire, max_wire), color="#333333", linewidth=1.5, zorder=1
        )
        self._ax.add_line(line)

        for ctrl in [control1, control2]:
            ctrl_circ = Circle(
                (x, ctrl),
                self._ctrl_rad,
                facecolor=self.GATE_COLORS["ctrl"],
                edgecolor="black",
                linewidth=1,
                zorder=2,
            )
            self._ax.add_patch(ctrl_circ)

        circ = Circle(
            (x, target),
            self._circ_rad,
            facecolor="white",
            edgecolor="#333333",
            linewidth=1.5,
            zorder=2,
        )
        self._ax.add_patch(circ)
        d = self._circ_rad * 0.7
        self._ax.add_line(
            plt.Line2D(
                (x - d, x + d),
                (target - d, target + d),
                color="#333333",
                linewidth=1.5,
                zorder=3,
            )
        )
        self._ax.add_line(
            plt.Line2D(
                (x - d, x + d),
                (target + d, target - d),
                color="#333333",
                linewidth=1.5,
                zorder=3,
            )
        )

    def _draw_cswap(self, x, control, target1, target2):
        """Draw Fredkin gate (CSWAP)"""
        min_wire = min(control, target1, target2)
        max_wire = max(control, target1, target2)

        line = plt.Line2D(
            (x, x), (min_wire, max_wire), color="#333333", linewidth=1.5, zorder=1
        )
        self._ax.add_line(line)

        ctrl_circ = Circle(
            (x, control),
            self._ctrl_rad,
            facecolor=self.GATE_COLORS["ctrl"],
            edgecolor="black",
            linewidth=1,
            zorder=2,
        )
        self._ax.add_patch(ctrl_circ)

        d = self._swap_dx
        for target in [target1, target2]:
            l1 = plt.Line2D(
                (x - d, x + d),
                (target - d, target + d),
                color="#333333",
                linewidth=1.5,
                zorder=2,
            )
            l2 = plt.Line2D(
                (x - d, x + d),
                (target + d, target - d),
                color="#333333",
                linewidth=1.5,
                zorder=2,
            )
            self._ax.add_line(l1)
            self._ax.add_line(l2)

    def _draw_controlled_gate(self, x, control, target, target_symbol="X"):
        """Draw controlled gates (CX, CY, CZ, CPhase)"""
        line = plt.Line2D(
            (x, x), (control, target), color="#333333", linewidth=1.5, zorder=1
        )
        self._ax.add_line(line)

        ctrl_circ = Circle(
            (x, control),
            self._ctrl_rad,
            facecolor=self.GATE_COLORS["ctrl"],
            edgecolor="black",
            linewidth=1,
            zorder=2,
        )
        self._ax.add_patch(ctrl_circ)

        if target_symbol == "X":
            circ = Circle(
                (x, target),
                self._circ_rad,
                facecolor="white",
                edgecolor="#333333",
                linewidth=1.5,
                zorder=2,
            )
            self._ax.add_patch(circ)
            d = self._circ_rad * 0.7
            self._ax.add_line(
                plt.Line2D(
                    (x - d, x + d),
                    (target - d, target + d),
                    color="#333333",
                    linewidth=1.5,
                    zorder=3,
                )
            )
            self._ax.add_line(
                plt.Line2D(
                    (x - d, x + d),
                    (target + d, target - d),
                    color="#333333",
                    linewidth=1.5,
                    zorder=3,
                )
            )
        elif target_symbol == "P":
            # CPhase gate: display P
            circ = Circle(
                (x, target),
                self._circ_rad,
                facecolor="#F0F0F0",
                edgecolor="#333333",
                linewidth=1.5,
                zorder=2,
            )
            self._ax.add_patch(circ)
            self._ax.text(
                x,
                target,
                "P",
                ha="center",
                va="center",
                fontsize=self._fontsize,
                fontweight="bold",
                zorder=3,
                family="sans-serif",
            )
        else:
            circ = Circle(
                (x, target),
                self._circ_rad,
                facecolor="#F0F0F0",
                edgecolor="#333333",
                linewidth=1.5,
                zorder=2,
            )
            self._ax.add_patch(circ)
            self._ax.text(
                x,
                target,
                target_symbol,
                ha="center",
                va="center",
                fontsize=self._fontsize,
                fontweight="bold",
                zorder=3,
                family="sans-serif",
            )

    def _draw_swap_gate(self, x, wires):
        """Draw SWAP gate - draw X at both ends, connect with a line in between"""
        if len(wires) < 2:
            return

        mapped_wires = [self.wire_map[w] for w in wires if w in self.wire_map]
        if not mapped_wires:
            return

        min_wire = min(mapped_wires)
        max_wire = max(mapped_wires)

        line = plt.Line2D(
            (x, x), (min_wire, max_wire), color="#333333", linewidth=1.5, zorder=1
        )
        self._ax.add_line(line)

        d = self._swap_dx

        # Top X
        l1 = plt.Line2D(
            (x - d, x + d),
            (min_wire - d, min_wire + d),
            color="#333333",
            linewidth=1.5,
            zorder=2,
        )
        l2 = plt.Line2D(
            (x - d, x + d),
            (min_wire + d, min_wire - d),
            color="#333333",
            linewidth=1.5,
            zorder=2,
        )
        self._ax.add_line(l1)
        self._ax.add_line(l2)

        # Bottom X
        l1 = plt.Line2D(
            (x - d, x + d),
            (max_wire - d, max_wire + d),
            color="#333333",
            linewidth=1.5,
            zorder=2,
        )
        l2 = plt.Line2D(
            (x - d, x + d),
            (max_wire + d, max_wire - d),
            color="#333333",
            linewidth=1.5,
            zorder=2,
        )
        self._ax.add_line(l1)
        self._ax.add_line(l2)

    def _draw_box(self, x, y, text, gate_type="single"):
        """Draw a single-qubit gate box"""
        half = self._box_length / 2
        box = FancyBboxPatch(
            (x - half + self._pad, y - half + self._pad),
            self._box_length - 2 * self._pad,
            self._box_length - 2 * self._pad,
            boxstyle=self._boxstyle,
            facecolor=self.GATE_COLORS.get(gate_type, "#FFFFFF"),
            edgecolor="#333333",
            linewidth=1.5,
            zorder=2,
        )
        self._ax.add_patch(box)
        self._ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=self._fontsize,
            zorder=3,
            family="sans-serif",
            fontweight="bold",
        )

    def _draw_multi_gate(self, x, wires, text):
        """Draw multi-qubit gate box spanning multiple wires"""
        min_w, max_w = min(wires), max(wires)
        half = self._box_length / 2
        height = max_w - min_w + self._box_length
        y = min_w - half
        box = FancyBboxPatch(
            (x - half + self._pad, y + self._pad),
            self._box_length - 2 * self._pad,
            height - 2 * self._pad,
            boxstyle=self._boxstyle,
            facecolor=self.GATE_COLORS["multi"],
            edgecolor="#333333",
            linewidth=1.5,
            zorder=2,
        )
        self._ax.add_patch(box)
        self._ax.text(
            x,
            (min_w + max_w) / 2,
            text,
            ha="center",
            va="center",
            fontsize=self._fontsize,
            zorder=3,
            family="sans-serif",
            fontweight="bold",
        )

    def _draw_measurements(self):
        """Draw measurement gates"""
        has_measure = any(
            op.get("name_or_mat", "").lower() in ["measure_allz"]
            for op in self.op_history
        )
        if has_measure:
            x = self.n_layers
            for wire in range(self.n_wires):
                self._draw_box(x, wire, "MZ", "measure")

    def _draw_labels(self, label_options):
        """Draw wire labels (can optionally display initial state)"""
        opts = label_options or {}
        show_initial_state = (
            opts.pop("show_initial_state", False) if isinstance(opts, dict) else False
        )

        for wire_label, idx in self.wire_map.items():
            if show_initial_state:
                label_text = f"{wire_label}: |0⟩"
            else:
                label_text = str(wire_label)

            self._ax.text(
                -1.5,
                idx,
                label_text,
                ha="center",
                va="center",
                fontsize=self._fontsize,
                fontfamily="sans-serif",
                **opts if isinstance(opts, dict) else {},
            )

    def _crop_labels(self):
        """Remove label area from view"""
        xlim = self._ax.get_xlim()
        self._ax.set_xlim((-1, xlim[1]))

    @property
    def fig(self):
        """Return the matplotlib figure"""
        return self._fig

    @property
    def ax(self):
        """Return the matplotlib axes"""
        return self._ax


def draw_mpl(qdev, show_initial_state=False, **kwargs):
    """
    Draw a circuit diagram in matplotlib format

    Args:
        qdev: FlagQuantum Circuit, CircuitIR, or device object
        show_initial_state: Whether to show the initial state (e.g., "0: |0⟩")
        **kwargs: Additional parameters
            - decimals: Precision for parameter display
            - wire_order: Wire order
            - fig: Existing matplotlib figure
            - figsize: Figure size
            - wire_options: Wire style options
            - label_options: Label style options
            - show_wire_labels: Whether to show wire labels
    """
    label_options = kwargs.get("label_options", {})
    if isinstance(label_options, dict):
        label_options["show_initial_state"] = show_initial_state
    else:
        label_options = {"show_initial_state": show_initial_state}
    kwargs["label_options"] = label_options

    drawer = MPLDrawer(qdev, **kwargs)
    return drawer.fig, drawer.ax
