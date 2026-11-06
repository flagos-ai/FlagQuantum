# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# SVG glyphs adapted from Qualtran v0.7.0 musical_score.js.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/musical_score.js

"""Build a self-contained browser viewer for the register musical score."""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import defaultdict
from heapq import heappop, heappush

from .register_score_data import MusicalScoreData, ScoreHorizontalLine, ScorePort


def _dependency_view(data: MusicalScoreData) -> MusicalScoreData:
    """Place operations by dependency generation and repack overlapping lifetimes."""
    ports = data["soqs"]
    columns = {port["seq_x"]: port["topo_gen"] for port in ports}
    if -1 in columns:
        columns[-1] = -1
    if any(
        port["seq_x"] == data["max_x"] and port["ident"].startswith("dang,")
        for port in ports
    ):
        columns[data["max_x"]] = 1 + max(
            (value for key, value in columns.items() if key != data["max_x"]),
            default=-1,
        )

    by_lane: dict[int, list[tuple[int, int]]] = defaultdict(list)
    lines: list[ScoreHorizontalLine] = []
    live: list[tuple[int, int]] = []
    free: list[int] = []
    lane_count = 0
    ordered = sorted(
        data["hlines"], key=lambda line: (columns[line["seq_x_start"]], line["y"])
    )
    for line in ordered:
        start, end = (columns[line[key]] for key in ("seq_x_start", "seq_x_end"))
        while live and live[0][0] < start:
            _, lane = heappop(live)
            heappush(free, lane)
        if free:
            lane = heappop(free)
        else:
            lane = lane_count
            lane_count += 1
        heappush(live, (end, lane))
        by_lane[line["y"]].append((line["seq_x_start"], lane))
        lines.append({**line, "y": lane, "seq_x_start": start, "seq_x_end": end})

    for items in by_lane.values():
        items.sort()
    projected_ports: list[ScorePort] = []
    lanes_by_column: dict[int, list[int]] = defaultdict(list)
    for port in ports:
        sequence = port["seq_x"]
        items = by_lane[port["y"]]
        _, lane = items[bisect_right(items, sequence, key=lambda item: item[0]) - 1]
        lanes_by_column[sequence].append(lane)
        projected_ports.append({**port, "seq_x": columns[sequence], "y": lane})
    return {
        "max_x": max(columns.values(), default=0),
        "max_y": max(lane_count - 1, 0),
        "soqs": projected_ports,
        "hlines": lines,
        "vlines": [
            {
                **line,
                "x": columns[line["x"]],
                "top_y": max(lanes_by_column[line["x"]]),
                "bottom_y": min(lanes_by_column[line["x"]]),
            }
            for line in data["vlines"]
        ],
    }


_PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Register musical score</title>
<style>
body { font: 14px system-ui, sans-serif; margin: 1rem; }
#score-container { overflow-x: auto; margin-top: 1rem; }
svg { display: block; }
text { text-anchor: middle; dominant-baseline: middle; fill: black; }
</style>
<label>Column view
  <select id="score-view">
    <option value="sequential">Sequential</option>
    <option value="dependency">Dependency</option>
  </select>
</label>
<div id="score-container"></div>
<script id="score-records" type="application/json">
__SCORE_RECORDS__
</script>
<script>
"use strict";
const views = JSON.parse(document.getElementById("score-records").textContent);
const selector = document.getElementById("score-view");
const container = document.getElementById("score-container");
const namespace = "http://www.w3.org/2000/svg";

function element(parent, name, attributes = {}) {
  const node = document.createElementNS(namespace, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  parent.appendChild(node);
  return node;
}

function label(parent, x, y, text, size) {
  element(parent, "text", {x, y, "font-size": size}).textContent = text;
}

function portGlyph(svg, port, x, y) {
  const kind = port.symb_cls;
  const attrs = port.symb_attributes;
  const group = element(svg, "g", {
    class: kind, "data-port": port.ident,
    "data-column": port.seq_x, "data-lane": port.y,
    transform: `translate(${x},${y})`
  });
  if (kind === "Circle") {
    element(group, "circle", {r: 5, stroke: "black", fill: attrs.filled ? "black" : "white"});
    return;
  }
  if (kind === "ModPlus") {
    element(group, "circle", {r: 7, stroke: "black", fill: "none"});
    element(group, "path", {d: "M 0 -7 l 0 14 M -7 0 l 14 0", stroke: "black"});
    return;
  }
  const w = Math.max(22, attrs.text.length * 3.5 + 6);
  if (kind === "TextBox") {
    element(group, "rect", {x: -w, y: -14, width: 2 * w, height: 28,
      fill: "white", stroke: "black"});
  } else if (kind === "RarrowTextBox" || kind === "LarrowTextBox") {
    const outline = kind === "RarrowTextBox"
      ? `${-w},-14 ${w-7},-14 ${w},0 ${w-7},14 ${-w},14`
      : `${-w},0 ${7-w},-14 ${w},-14 ${w},14 ${7-w},14`;
    element(group, "polygon", {points: outline, fill: "white", stroke: "black"});
  }
  label(group, 0, 0, attrs.text, attrs.fontsize || 12);
}

function render() {
  const data = views[selector.value];
  const cell = data.soqs.reduce((size, port) => Math.max(
    size, (port.symb_attributes.text || "").length * 7 + 14
  ), 55);
  const x = column => (column + 2) * cell;
  const y = lane => 32 + lane * 42;
  const width = x(data.max_x + 1);
  const height = y(data.max_y) + 28;
  container.replaceChildren();
  const svg = element(container, "svg", {
    width, height, viewBox: `0 0 ${width} ${height}`,
    role: "img", "aria-label": "Register musical score"
  });
  for (const line of data.hlines) {
    element(svg, "line", {class: "h", x1: x(line.seq_x_start), x2: x(line.seq_x_end),
      y1: y(line.y), y2: y(line.y),
      stroke: line.flavor === "HLineFlavor.CLASSICAL" ? "blue" : "black"});
  }
  for (const line of data.vlines) {
    element(svg, "line", {class: "v", x1: x(line.x), x2: x(line.x),
      y1: y(line.bottom_y), y2: y(line.top_y), stroke: "black"});
    const text = typeof line.label === "string" ? line.label : line.label.text;
    if (text) label(svg, x(line.x), y(line.bottom_y) - 22, text, 11);
  }
  for (const port of data.soqs) portGlyph(svg, port, x(port.seq_x), y(port.y));
}

selector.addEventListener("change", render);
render();
</script>
</html>
"""


def export_register_score_html(data: MusicalScoreData) -> str:
    """Export an offline HTML page with sequential and dependency column views."""
    payload = json.dumps(
        {"sequential": data, "dependency": _dependency_view(data)}, ensure_ascii=False
    )
    # Prevent a register label from closing the script data element.
    return _PAGE.replace("__SCORE_RECORDS__", payload.replace("<", "\\u003c"))
