"""
文本模式电路绘图器
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class _CurrentTotals:
    """累积的电路字符串"""

    finished_lines: List[str]  # 已完成的行（换行时使用）
    wire_totals: List[str]  # 量子线累积字符串
    bit_totals: List[str]  # 经典线累积字符串


@dataclass
class _Config:
    """绘图配置"""

    wire_map: Dict[Any, int]  # 线标签 -> 显示位置
    wire_order: List[Any]  # 线顺序（从上到下）
    num_op_layers: int  # 操作层数
    cur_layer: int = -1  # 当前层索引
    decimals: Optional[int] = None  # 参数精度
    show_wire_labels: bool = True  # 是否显示线标签

    @property
    def wire_filler(self) -> str:
        """填充字符：操作层用'─'，测量层用空格"""
        return "─" if self.cur_layer < self.num_op_layers else " "

    @property
    def n_wires(self) -> int:
        return len(self.wire_map)


class TextDrawer:
    """
    文本电路绘图器

    核心机制：
    1. 维护 totals.wire_totals 作为累积字符串
    2. 每层通过 filler.join([t, s]) 连接
    3. _left_justify 确保所有线等长
    """

    # 门名称到符号的映射
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
        "sx": "SX",
        "sdg": "S†",
        "tdg": "T†",
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
        "cphase": "P",
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

    def __init__(
        self,
        qdev,
        wire_order=None,
        show_all_wires=False,
        decimals=3,
        max_length=100,
        show_initial_state=False,
    ):
        self.qdev = qdev
        self.op_history = qdev.op_history if hasattr(qdev, "op_history") else []
        self.n_wires = (
            qdev.n_wires if hasattr(qdev, "n_wires") else self._detect_n_wires()
        )
        self.decimals = decimals
        self.max_length = max_length
        self.show_all_wires = show_all_wires
        self.show_initial_state = show_initial_state  # 新增

        # 确定线序
        self.wire_order = self._create_wire_order(wire_order)
        self.wire_map = {wire: idx for idx, wire in enumerate(self.wire_order)}
        self.reverse_wire_map = {idx: wire for wire, idx in self.wire_map.items()}

        # 创建分层
        self.layers = self._create_layers()
        self.num_op_layers = len(self.layers)

    def _detect_n_wires(self) -> int:
        """自动检测量子比特数"""
        max_wire = -1
        for op in self.op_history:
            wires = op.get("wires", [])
            if isinstance(wires, int):
                wires = [wires]
            for w in wires:
                if isinstance(w, int) and w > max_wire:
                    max_wire = w
        return max_wire + 1 if max_wire >= 0 else 0

    def _create_wire_order(self, wire_order):
        """创建线顺序"""
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

        return wire_order

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
            "rxx",
            "ryy",
            "rzz",
        ]

    def _create_layers(self) -> List[List[Dict]]:
        """
        分层算法

        多量子门会占用中间的所有线，防止其他操作重叠
        """
        last_layer = {}  # wire -> last_layer_index
        layers = []

        for op in self.op_history:
            wires = op.get("wires", [])
            if isinstance(wires, int):
                wires = [wires]

            if not wires:
                # 无 wires 的操作（如全局操作）占用所有线
                wires = list(range(self.n_wires))

            # 获取操作占用的所有线（包括中间线）
            min_wire = min(wires)
            max_wire = max(wires)
            occupied_wires = set(range(min_wire, max_wire + 1))

            # 映射到显示线索引
            mapped_occupied = set()
            for w in occupied_wires:
                if w in self.wire_map:
                    mapped_occupied.add(self.wire_map[w])

            if not mapped_occupied:
                continue

            # 找到这些线上最后一层的最大索引
            max_layer = -1
            for w in mapped_occupied:
                if w in last_layer:
                    max_layer = max(max_layer, last_layer[w])

            new_layer = max_layer + 1

            # 确保层列表足够长
            while len(layers) <= new_layer:
                layers.append([])

            layers[new_layer].append(op)

            # 更新这些线的最后层
            for w in mapped_occupied:
                last_layer[w] = new_layer

        return layers

    def _get_param_str(self, params) -> str:
        """格式化参数字符串（固定小数位数，保留末尾0）"""
        if not params or self.decimals is None:
            return ""

        if isinstance(params, list):
            p = params[0] if params else None
        else:
            p = params

        if isinstance(p, (int, float)):
            # 固定小数位数，保留末尾0
            return f"{p:.{self.decimals}f}"
        elif p is not None:
            return f"{p}"
        return ""

    def _render_single_gate(self, name: str, params) -> str:
        """渲染单量子门：─RX(0.31)─"""
        symbol = self.GATE_SYMBOLS.get(name.lower(), name.upper() if name else "?")
        param_str = self._get_param_str(params)
        if param_str:
            return f"─{symbol}({param_str})─"
        return f"─{symbol}─"

    def _render_toffoli(self, wires: List[int]) -> List[tuple]:
        """
        渲染 Toffoli (CCX) 门

        格式：
        控制线1: ╭●
        控制线2: ├●
        目标线:  ╰X
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
                # 第一个控制线
                lines.append((abs_idx, "╭●"))
            elif i == n_lines - 2:
                # 第二个控制线（如果是3线，这是中间线）
                lines.append((abs_idx, "├●"))
            elif i == n_lines - 1:
                # 目标线
                lines.append((abs_idx, "╰X"))
            else:
                # 其他中间线：连接线
                lines.append((abs_idx, "│"))

        return lines

    def _render_cswap(self, wires: List[int]) -> List[tuple]:
        """
        渲染 Fredkin (CSWAP) 门

        格式：
        控制线:  ╭●
        目标线1: ├╳
        目标线2: ╰╳
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
                # 控制线
                lines.append((abs_idx, "╭●"))
            elif i == n_lines - 2:
                # 第一个目标线
                lines.append((abs_idx, "├╳"))
            elif i == n_lines - 1:
                # 第二个目标线
                lines.append((abs_idx, "╰╳"))
            else:
                # 其他中间线：连接线
                lines.append((abs_idx, "│"))

        return lines

    def _render_swap_gate(self, wires: List[int]) -> List[tuple]:
        """
        渲染 SWAP 门（避免误导：只画两端，中间用连接线）

        相邻线 SWAP(wires=[0,1]):
            0: ╳
            1: ╳

        不相邻线 SWAP(wires=[0,2]):
            0: ╳
            1: │   (只是连接线)
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
                # 两端：画 X
                lines.append((abs_idx, "╳"))
            else:
                # 中间线：只画连接线
                lines.append((abs_idx, "│"))

        return lines

    def _render_controlled_gate(self, wires, target_symbol):
        """渲染受控门（CZ, CY, CX 等）"""
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
                # 顶部：控制线
                lines.append((abs_idx, "╭●"))
            elif i == n_lines - 1:
                # 底部：目标线（显示 Z、X 或 Y）
                lines.append((abs_idx, f"╰{target_symbol}"))
            else:
                # 中间线：连接线
                lines.append((abs_idx, "│"))

        return lines

    def _render_controlled_gate_with_param(self, wires, gate_name, params):
        """渲染带参数的受控门（CRX, CRY, CRZ）"""
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

    def _render_ising_gate(self, wires, gate_name, params):
        """渲染 Ising 门（RXX, RYY, RZZ）"""
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

    def _render_multi_gate(self, name: str, wires: List[int], params) -> List[tuple]:
        """
        渲染多量子门（如 QFT）

        格式：
        顶部: ╭QFT─
        中间: ├QFT─
        底部: ╰QFT─
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

    def _render_op(self, op: Dict) -> List[tuple]:
        """
        渲染单个操作，返回 [(wire_index, string), ...]

        注意：这里只返回添加的内容，不包含 filler 字符
        filler 字符由 _initialize_layer_str 和 _left_justify 处理
        """
        name = op.get("name_or_mat", "").lower()
        wires = op.get("wires", [])
        params = op.get("params", [])

        if isinstance(wires, int):
            wires = [wires]

        # 跳过测量操作（在 _finalize_layers 中统一处理）
        if name == "measure_allz":
            return []

        if not wires:
            return []

        # 过滤出存在的线
        valid_wires = [w for w in wires if w in self.wire_map]
        if not valid_wires:
            return []

        # 单量子门
        if len(valid_wires) == 1:
            wire_idx = self.wire_map[valid_wires[0]]
            return [(wire_idx, self._render_single_gate(name, params))]

        # Toffoli (CCX) - 3 个量子比特
        if name in ["ccx", "toffoli"] and len(valid_wires) == 3:
            return self._render_toffoli(valid_wires)

        # Fredkin (CSWAP) - 3 个量子比特
        if name in ["cswap", "fredkin"] and len(valid_wires) == 3:
            return self._render_cswap(valid_wires)

        # CZ 门（受控-Z）
        if name == "cz":
            return self._render_controlled_gate(valid_wires, "Z")

        # CNOT/CX
        if name in ["cx", "cnot"]:
            return self._render_controlled_gate(valid_wires, "X")

        # CY 门（受控-Y）
        if name == "cy":
            return self._render_controlled_gate(valid_wires, "Y")

        # CPhase 门
        if name in ["cphase", "controlledphase"]:
            return self._render_controlled_gate_with_param(valid_wires, "CP", params)

        # CRX, CRY, CRZ 受控旋转门
        if name == "crx":
            return self._render_controlled_gate_with_param(valid_wires, "CRX", params)
        if name == "cry":
            return self._render_controlled_gate_with_param(valid_wires, "CRY", params)
        if name == "crz":
            return self._render_controlled_gate_with_param(valid_wires, "CRZ", params)

        # Ising 门 (RXX, RYY, RZZ)
        if name in ["rxx", "ryy", "rzz"]:
            return self._render_ising_gate(valid_wires, name.upper(), params)

        # SWAP 门
        if name == "swap":
            return self._render_swap_gate(valid_wires)

        # 默认多量子门
        return self._render_multi_gate(name, valid_wires, params)

    def _initialize_layer_str(self, config: _Config) -> List[str]:
        """初始化新层的字符串数组"""
        return [config.wire_filler] * config.n_wires

    def _left_justify(self, layer_str: List[str], config: _Config) -> List[str]:
        """将这一层中所有线填充到相同长度"""
        if not layer_str:
            return layer_str

        max_label_len = max(len(s) for s in layer_str)

        for w in range(config.n_wires):
            layer_str[w] = layer_str[w].ljust(max_label_len, config.wire_filler)

        return layer_str

    def _add_layer_str_to_totals(
        self, totals: _CurrentTotals, layer_str: List[str], config: _Config
    ) -> _CurrentTotals:
        """将当前层合并到累积字符串中"""
        totals.wire_totals = [
            config.wire_filler.join([t, s])
            for t, s in zip(totals.wire_totals, layer_str[: config.n_wires])
        ]

        return totals

    def _add_to_finished_lines(
        self, totals: _CurrentTotals, config: _Config, add_measurement: bool = False
    ) -> _CurrentTotals:
        """当前行超过 max_length 时，保存到 finished_lines 并开始新行"""
        suffix = " ···"

        saved_lines = [line + suffix for line in totals.wire_totals]

        totals.finished_lines += saved_lines
        totals.finished_lines[-1] += "\n"

        # 重置 totals（新行）
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
        """添加行尾标记（测量符号）"""

        # 检查是否有 measure_allZ 操作
        has_all_measure = any(
            op.get("name_or_mat", "").lower() == "measure_allz"
            for op in self.op_history
        )

        if has_all_measure:
            # 所有线都测量
            for i in range(len(totals.wire_totals)):
                totals.wire_totals[i] = f"{totals.wire_totals[i]}─┤  <Z>"
        else:
            # 逐线检查测量
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
        """初始化 wire_totals（包含线标签，可选显示初始状态）"""
        if config.show_wire_labels:
            # 检查是否需要显示初始状态（可以通过属性控制）
            show_initial_state = getattr(self, "show_initial_state", False)

            if show_initial_state:
                wire_totals = [f"{wire}: |0⟩ " for wire in config.wire_order]
            else:
                wire_totals = [f"{wire}: " for wire in config.wire_order]

            # 右对齐线标签（使所有线的标签等宽）
            line_length = max(len(s) for s in wire_totals)
            wire_totals = [s.rjust(line_length, " ") for s in wire_totals]
        else:
            wire_totals = [""] * config.n_wires

        return wire_totals

    def draw(self) -> str:
        """绘制电路图（支持自动换行）"""
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

        # 初始化累加器（包含线标签）
        wire_totals = self._initialize_wire_totals(config)

        totals = _CurrentTotals(
            finished_lines=[], wire_totals=wire_totals, bit_totals=[]
        )

        len_suffix = 4  # " ···" 的长度

        # 逐层处理
        for layer_idx, layer in enumerate(self.layers):
            config.cur_layer = layer_idx

            # 初始化这一层
            layer_str = self._initialize_layer_str(config)

            # 添加这一层的所有操作
            for op in layer:
                rendered = self._render_op(op)
                for wire_idx, s in rendered:
                    layer_str[wire_idx] += s

            # 左对齐（填充到相同长度）
            layer_str = self._left_justify(layer_str, config)

            # 检查是否需要换行
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

            # 合并到累积字符串
            totals = self._add_layer_str_to_totals(totals, layer_str, config)

        # 所有层处理完成后，最后添加测量标记
        totals = self._finalize_layers(totals, config)

        # 合并最终结果
        result_lines = totals.finished_lines + totals.wire_totals + totals.bit_totals

        # 过滤空行
        result_lines = [line for line in result_lines if line.strip() or line == "\n"]

        return "\n".join(result_lines)


def draw_text(
    qdev,
    wire_order=None,
    show_all_wires=False,
    decimals=3,
    max_length=100,
    show_initial_state=False,
):
    """
    绘制文本电路图

    Args:
        qdev: FlagQuantum 设备对象
        wire_order: 线顺序（从上到下），例如 [0, 1, 2, 3] 或 ["q0", "q1"]
        show_all_wires: 是否显示所有线（包括未使用的）
        decimals: 参数显示精度
        max_length: 单行最大宽度
        show_initial_state: 是否显示初始状态（如 "0: |0⟩"）

    Returns:
        str: 电路图字符串
    """
    drawer = TextDrawer(
        qdev, wire_order, show_all_wires, decimals, max_length, show_initial_state
    )
    return drawer.draw()
