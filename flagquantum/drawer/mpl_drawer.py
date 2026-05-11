# ----------------- 优化版 MPLDrawer -----------------
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch


class MPLDrawer:
    """Matplotlib 电路绘图器（优化版，专业配色）"""

    _box_length = 0.8
    _circ_rad = 0.25
    _ctrl_rad = 0.1
    _swap_dx = 0.2
    _fontsize = 12
    _pad = 0.1
    _boxstyle = f"round, pad={_pad}"

    GATE_COLORS = {
        "single": "#7B9EC2",  # 固定单量子门（X, Y, Z, H, S, T 等）- 蓝色
        "param": "#E15759",  # 参数门（RX, RY, RZ, U1, U2, U3 等）- 红色
        "multi": "#F28E2B",  # 多量子门（QFT, Toffoli 等）- 橙色
        "ctrl": "#000000",  # 控制点 - 黑色
        "target": "#E15759",  # 目标点 - 红色
        "swap": "#76B7B2",  # SWAP 门 - 青色
        "measure": "#59A14F",  # 测量 - 绿色
        "cr": "#F28E2B",  # crx/cry/crz
        "cp": "#76B7B2",  # cphase
        "p": "#DDA0DD",  # phase
        "ising": "#FFB6C1",
    }

    GATE_SYMBOLS = {
        # 单量子门
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
        # 两量子门
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
        # 三量子门
        "ccx": "Toffoli",
        "toffoli": "Toffoli",
        "cswap": "CSWAP",
        "fredkin": "CSWAP",
        # 测量
        "measure": "Meas",
        "measurement": "M",
        "measure_allz": "MZ",
    }

    def __init__(self, qdev, wire_order=None, fig=None, **kwargs):
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
        if wire_order is None:
            wire_order = list(range(self.n_wires))
        for w in range(self.n_wires):
            if w not in wire_order:
                wire_order.append(w)
        return {wire: idx for idx, wire in enumerate(wire_order)}

    def _create_layers(self):
        """分层算法 - 确保同一层的操作共享 x 坐标"""
        last_op_layer = {}
        layers = []
        for op in self.op_history:
            wires = op.get("wires", [])
            if isinstance(wires, int):
                wires = [wires]
            if not wires:
                continue

            # 获取操作占用的所有线（包括中间线）
            min_w = min(wires)
            max_w = max(wires)
            occupied_wires = set(range(min_w, max_w + 1))

            # 映射到显示线索引
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
        if not params or self.decimals is None:
            return ""
        p = params[0] if isinstance(params, list) else params
        if isinstance(p, (int, float)):
            formatted = f"{p:.{self.decimals}f}"
            formatted = formatted.rstrip("0").rstrip(".")
            return f"({formatted})"
        return f"({p})"

    def _is_parameterized_gate(self, name: str) -> bool:
        """判断是否为参数门（有可训练参数）"""
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
            "xx",
            "yy",
            "zz",
        ]

    def _draw_operations(self):
        """绘制所有操作 - 同一层的操作共享相同的 x 坐标"""
        for layer_idx, layer in enumerate(self.layers):
            x = layer_idx
            for op in layer:
                self._draw_operation(op, x)

    def _draw_operation(self, op, x):
        """绘制单个操作 - 使用指定的 x 坐标"""
        name = op.get("name_or_mat", "").lower()
        wires = op.get("wires", [])
        params = op.get("params", [])
        if isinstance(wires, int):
            wires = [wires]
        if not wires:
            return

        # 测量操作单独处理
        if name == "measure_allz":
            return

        symbol = self.GATE_SYMBOLS.get(name, name.upper() if name else "?")
        param_str = self._get_param_str(params)

        # 判断是否有参数
        if param_str:
            param_clean = param_str.strip("()")
            label = f"{symbol}\n{param_clean}"
        else:
            label = symbol

        # 判断门类型（参数门用特殊颜色）
        is_param_gate = self._is_parameterized_gate(name)
        gate_type = "param" if is_param_gate else "single"

        # 单量子门
        if len(wires) == 1:
            if name in ["rx", "ry", "rz"]:
                self._draw_box(x, wires[0], label, gate_type)
            elif name in ["p", "phase"]:
                self._draw_box(x, wires[0], label, "p")

            else:
                self._draw_box(x, wires[0], label, gate_type)

        # Toffoli (CCX) - 3 个量子比特
        elif name in ["ccx", "toffoli"] and len(wires) == 3:
            control1, control2, target = wires[0], wires[1], wires[2]
            self._draw_toffoli(x, control1, control2, target)

        # Fredkin (CSWAP) - 3 个量子比特
        elif name in ["cswap", "fredkin"] and len(wires) == 3:
            control, target1, target2 = wires[0], wires[1], wires[2]
            self._draw_cswap(x, control, target1, target2)

        # CNOT/CX
        elif name in ["cx", "cnot"] and len(wires) >= 2:
            self._draw_controlled_gate(x, wires[0], wires[1], target_symbol="X")

        # CY 门
        elif name == "cy" and len(wires) >= 2:
            self._draw_controlled_gate(x, wires[0], wires[1], target_symbol="Y")

        # CZ 门
        elif name == "cz" and len(wires) >= 2:
            self._draw_controlled_gate(x, wires[0], wires[1], target_symbol="Z")

        # CPhase 门
        elif name in ["cphase", "controlledphase"] and len(wires) >= 2:
            self._draw_controlled_phase_gate_with_param(
                x, wires[0], wires[1], params, "CP"
            )

        # CRX, CRY, CRZ 受控旋转门
        elif name == "crx" and len(wires) >= 2:
            self._draw_controlled_gate_with_param(x, wires[0], wires[1], params, "CRX")
        elif name == "cry" and len(wires) >= 2:
            self._draw_controlled_gate_with_param(x, wires[0], wires[1], params, "CRY")
        elif name == "crz" and len(wires) >= 2:
            self._draw_controlled_gate_with_param(x, wires[0], wires[1], params, "CRZ")

        # Ising 门 (RXX, RYY, RZZ)
        elif name in ["rxx", "ryy", "rzz"] and len(wires) >= 2:
            self._draw_ising_gate(x, wires, name.upper(), params)

        # SWAP - 支持跨越中间线
        elif name == "swap" and len(wires) >= 2:
            self._draw_swap_gate(x, wires)

        # 多量子门
        elif len(wires) > 1:
            self._draw_multi_gate(x, wires, label)

    def _draw_controlled_gate_with_param(self, x, control, target, params, gate_name):
        """绘制带参数的受控门 (CRX, CRY, CRZ)"""
        # 连接线
        line = plt.Line2D(
            (x, x), (control, target), color="#333333", linewidth=1.5, zorder=1
        )
        self._ax.add_line(line)

        # 控制点
        ctrl_circ = Circle(
            (x, control),
            self._ctrl_rad,
            facecolor=self.GATE_COLORS["ctrl"],
            edgecolor="black",
            linewidth=1,
            zorder=2,
        )
        self._ax.add_patch(ctrl_circ)

        # 参数化目标门
        param_str = self._get_param_str(params)
        if param_str:
            param_clean = param_str.strip("()")
            label = f"{gate_name}\n{param_clean}"
        else:
            label = gate_name

        # 绘制带参数的目标框
        self._draw_box(x, target, label, "cr")

    def _draw_controlled_phase_gate_with_param(
        self, x, control, target, params, gate_name
    ):
        """绘制带参数的受控Phase门 (CP)"""
        # 连接线
        line = plt.Line2D(
            (x, x), (control, target), color="#333333", linewidth=1.5, zorder=1
        )
        self._ax.add_line(line)

        # 控制点
        ctrl_circ = Circle(
            (x, control),
            self._ctrl_rad,
            facecolor=self.GATE_COLORS["ctrl"],
            edgecolor="black",
            linewidth=1,
            zorder=2,
        )
        self._ax.add_patch(ctrl_circ)

        # 参数化目标门
        param_str = self._get_param_str(params)
        if param_str:
            param_clean = param_str.strip("()")
            label = f"{gate_name}\n{param_clean}"
        else:
            label = gate_name

        # 绘制带参数的目标框
        self._draw_box(x, target, label, "cp")

    def _draw_controlled_gate_with_param(self, x, control, target, params, gate_name):
        """绘制带参数的受控门 (CRX, CRY, CRZ)"""
        # 连接线
        line = plt.Line2D(
            (x, x), (control, target), color="#333333", linewidth=1.5, zorder=1
        )
        self._ax.add_line(line)

        # 控制点
        ctrl_circ = Circle(
            (x, control),
            self._ctrl_rad,
            facecolor=self.GATE_COLORS["ctrl"],
            edgecolor="black",
            linewidth=1,
            zorder=2,
        )
        self._ax.add_patch(ctrl_circ)

        # 参数化目标门
        param_str = self._get_param_str(params)
        if param_str:
            param_clean = param_str.strip("()")
            label = f"{gate_name}\n{param_clean}"
        else:
            label = gate_name

        # 绘制带参数的目标框
        self._draw_box(x, target, label, "cr")

    def _draw_ising_gate(self, x, wires, gate_name, params):
        """绘制 Ising 门 (RXX, RYY, RZZ) - 用方框覆盖所有线"""
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
        """绘制 Toffoli (CCX) 门"""
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
        """绘制 Fredkin (CSWAP) 门"""
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
        """绘制受控门 (CX, CY, CZ, CPhase)"""
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
            # CPhase 门：显示 P
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
        """绘制 SWAP 门（只画两端，中间用线连接）"""
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

        # 顶部 X
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

        # 底部 X
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
        """绘制单量子门方框"""
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
        """绘制多量子门方框"""
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
        """绘制测量门"""
        has_measure = any(
            op.get("name_or_mat", "").lower() in ["measure_allz"]
            for op in self.op_history
        )
        if has_measure:
            x = self.n_layers
            for wire in range(self.n_wires):
                self._draw_box(x, wire, "MZ", "measure")

    def _draw_labels(self, label_options):
        """绘制线标签（可显示初始状态）"""
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
        xlim = self._ax.get_xlim()
        self._ax.set_xlim((-1, xlim[1]))

    @property
    def fig(self):
        return self._fig

    @property
    def ax(self):
        return self._ax


def draw_mpl(qdev, show_initial_state=False, **kwargs):
    """
    绘制 matplotlib 格式电路图

    Args:
        qdev: FlagQuantum 设备对象
        show_initial_state: 是否显示初始状态（如 "0: |0⟩"）
        **kwargs: 其他参数
            - decimals: 参数显示精度
            - wire_order: 线顺序
            - fig: 现有的 matplotlib figure
            - figsize: 图形大小
            - wire_options: 线样式选项
            - label_options: 标签样式选项
            - show_wire_labels: 是否显示线标签
    """
    label_options = kwargs.get("label_options", {})
    if isinstance(label_options, dict):
        label_options["show_initial_state"] = show_initial_state
    else:
        label_options = {"show_initial_state": show_initial_state}
    kwargs["label_options"] = label_options

    drawer = MPLDrawer(qdev, **kwargs)
    return drawer.fig, drawer.ax
