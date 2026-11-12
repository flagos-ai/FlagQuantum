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
#score-bounds input { width: 7rem; }
#score-container { overflow-x: auto; margin-top: 1rem; }
svg { display: block; }
text { text-anchor: middle; dominant-baseline: middle; fill: black; }
.fixed-score { font: initial; max-width: 100%; height: auto; }
.score-axis text { text-anchor: inherit; dominant-baseline: auto; fill: currentColor; }
</style>
<label>Column view
  <select id="score-view">
    <option value="sequential">Sequential</option>
    <option value="dependency">Dependency</option>
  </select>
</label>
<label>Score style
  <select id="score-style">
    <option value="adaptive">Adaptive</option>
    <option value="fixed">Fixed with numeric axes</option>
  </select>
</label>
<form id="score-bounds">
  Fixed score bounds
  <label>X start <input id="score-x-start" type="number" step="any" required></label>
  <label>X end <input id="score-x-end" type="number" step="any" required></label>
  <label>Y start <input id="score-y-start" type="number" step="any" required></label>
  <label>Y end <input id="score-y-end" type="number" step="any" required></label>
  <button type="submit">Apply bounds</button>
  <button id="score-fit" type="button">Fit selected view</button>
</form>
<form id="score-load">
  <label>Score JSON URL <input id="score-url" type="text" required></label>
  <button type="submit">Load score</button>
</form>
<div id="score-container"></div>
<script id="score-records" type="application/json">
__SCORE_RECORDS__
</script>
<script>
"use strict";
const views = JSON.parse(document.getElementById("score-records").textContent);
const selector = document.getElementById("score-view");
const container = document.getElementById("score-container");
const styleSelector = document.getElementById("score-style");
const fixedStyle = () => styleSelector.value === "fixed";
const namespace = "http://www.w3.org/2000/svg";
let loadVersion = 0;
let manualDomains = null;
const boundInputs = ["x-start", "x-end", "y-start", "y-end"].map(
  name => document.getElementById(`score-${name}`));

function scoreDomains(data) {
  return manualDomains || [[-2, data.max_x + 2], [0, data.max_y + 1]];
}

function validateDomains(domains) {
  const [[x0, x1], [y0, y1]] = domains;
  const width = (x1 - x0) * 55 + 40 + 30 + 6 + 6;
  const height = (y1 - y0) * 42 + 30 + 6 + 20 + 6;
  if (![x0, x1, y0, y1].every(value => Number.isFinite(value) && Math.abs(value) <= Number.MAX_SAFE_INTEGER) ||
      ![width, height].every(Number.isFinite) || width <= 0 || height <= 0) {
    throw new RangeError("Bounds must have finite safe coordinate magnitudes and positive canvas dimensions");
  }
  // Preflight tick admission before changing either the current bounds or view.
  axisTicks(x0, x1, width / 80);
  axisTicks(y0, y1, height / 50);
}

function applyBounds(event) {
  event.preventDefault();
  const values = boundInputs.map(input => input.valueAsNumber);
  const domains = [values.slice(0, 2), values.slice(2, 4)];
  try {
    validateDomains(domains);
    manualDomains = domains;
    styleSelector.value = "fixed";
    render();
  } catch (error) {
    showScoreError("Invalid score bounds", String(error));
  }
}

function validateScore(data) {
  const numbers = (row, fields) => fields.every(key => Number.isFinite(row?.[key]));
  if (!numbers(data, ["max_x", "max_y"]) || data.max_x < 0 || data.max_y < 0 ||
      ![data.soqs, data.hlines, data.vlines].every(Array.isArray)) {
    throw new TypeError("Invalid musical score dimensions or record arrays");
  }
  const kinds = new Set(["Circle", "ModPlus", "Text", "TextBox", "LarrowTextBox", "RarrowTextBox"]);
  const identities = new Set();
  for (const port of data.soqs) {
    const attrs = port?.symb_attributes;
    const key = `${port?.symb_cls},${port?.ident}`;
    if (!numbers(port, ["seq_x", "y"]) || !kinds.has(port.symb_cls) ||
        typeof port.ident !== "string" || identities.has(key) || !attrs ||
        typeof attrs !== "object" || Array.isArray(attrs) ||
        (attrs.text !== undefined && typeof attrs.text !== "string") ||
        (port.symb_cls !== "Circle" && port.symb_cls !== "ModPlus" && typeof attrs.text !== "string") ||
        (attrs.fontsize !== undefined && (!Number.isFinite(attrs.fontsize) || attrs.fontsize < 0)) ||
        (attrs.filled !== undefined && typeof attrs.filled !== "boolean")) {
      throw new TypeError("Invalid musical score glyph record");
    }
    identities.add(key);
  }
  if (!data.hlines.every(row => numbers(row, ["y", "seq_x_start", "seq_x_end"])) ||
      !data.vlines.every(row => numbers(row, ["x", "top_y", "bottom_y"]) &&
        (typeof row.label === "string" || typeof row.label?.text === "string")) ||
      new Set(data.hlines.map(row => `${row.y},${row.seq_x_start}`)).size !== data.hlines.length ||
      new Set(data.vlines.map(row => row.x)).size !== data.vlines.length) {
    throw new TypeError("Invalid musical score line record");
  }
  if (fixedStyle()) validateDomains(scoreDomains(data));
}

function showScoreError(message, details) {
  const alert = document.createElement("div");
  alert.setAttribute("role", "alert");
  const heading = document.createElement("h3");
  heading.style.color = "red";
  heading.textContent = message;
  const paragraph = document.createElement("p");
  paragraph.textContent = details;
  alert.append(heading, paragraph);
  container.appendChild(alert);
}

async function loadScore(filename) {
  const version = ++loadVersion;
  try {
    const response = await fetch(filename);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const data = await response.json();
    if (version !== loadVersion) return false;
    validateScore(data);
    views.sequential = data;
    // Imported source JSON carries no precomputed, collision-free dependency view.
    delete views.dependency;
    selector.querySelector('option[value="dependency"]').disabled = true;
    selector.value = "sequential";
    render();
    return true;
  } catch (error) {
    if (version === loadVersion) showScoreError(`Error loading '${filename}': ${error}`,
      "Make sure the file exists and make sure you're serving this page from `python -m http.server` instead of from a file");
    return false;
  }
}

function element(parent, name, attributes = {}) {
  const node = document.createElementNS(namespace, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  parent.appendChild(node);
  return node;
}

function label(parent, x, y, text, size) {
  const node = element(parent, "text", {x, y, "font-size": size});
  node.textContent = text;
  return node;
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
    return group;
  }
  if (kind === "ModPlus") {
    element(group, "circle", {r: 7, stroke: "black", fill: "none"});
    element(group, "path", {d: "M 0 -7 l 0 14 M -7 0 l 14 0", stroke: "black"});
    return group;
  }
  group.addEventListener("click", event => console.log(event, ports.get(`${kind},${port.ident}`).data));
  const w = fixedStyle() ? 22 : Math.max(22, attrs.text.length * 3.5 + 6);
  if (kind === "TextBox") {
    element(group, "rect", {x: -w, y: -14, width: 2 * w, height: 28,
      fill: "white", stroke: "black"});
  } else if (kind === "RarrowTextBox" || kind === "LarrowTextBox") {
    const outline = kind === "RarrowTextBox"
      ? `${-w},-14 ${w-7},-14 ${w},0 ${w-7},14 ${-w},14`
      : `${-w},0 ${7-w},-14 ${w},-14 ${w},14 ${7-w},14`;
    element(group, "polygon", {x: -w, y: -14, points: outline, fill: "white", stroke: "black"});
  }
  label(group, 0, 0, attrs.text, attrs.fontsize || 12);
  return group;
}

// Keyed joins and cubic-in-out timing follow the frozen source's D3 transitions.
const svg = element(container, "svg", {role: "img", "aria-label": "Register musical score"});
const axes = element(svg, "g", {class: "score-axes"});
const horizontal = new Map();
const vertical = new Map();
const captions = new Map();
const ports = new Map();
let frame = 0;
let tweens = [];

function transition(node, targets, finish = () => {}) {
  const from = {};
  for (const name of Object.keys(targets)) {
    const value = node.getAttribute(name);
    from[name] = name === "transform"
      ? value.slice(10, -1).split(",").map(Number) : Number(value);
  }
  tweens.push({node, from, targets, finish});
}

function join(nodes, rows, key, enter, update, exit) {
  const seen = new Set();
  for (const data of rows) {
    const identity = key(data);
    seen.add(identity);
    let record = nodes.get(identity);
    if (!record) {
      record = {node: enter(data), data};
      nodes.set(identity, record);
    }
    record.data = data;
    update(record.node, data);
  }
  for (const [identity, record] of nodes) {
    if (!seen.has(identity)) {
      exit(record.node, record.data, () => {
        record.node.remove();
        nodes.delete(identity);
      });
    }
  }
}

function updatePort(group, port) {
  const attrs = port.symb_attributes;
  group.dataset.column = port.seq_x;
  group.dataset.lane = port.y;
  if (port.symb_cls === "Circle") {
    group.querySelector("circle").setAttribute("fill", attrs.filled ? "black" : "white");
  } else if (port.symb_cls !== "ModPlus") {
    const text = group.querySelector("text");
    text.textContent = attrs.text;
    if (fixedStyle()) text.removeAttribute("font-size");
    else text.setAttribute("font-size", attrs.fontsize || 12);
    const w = fixedStyle() ? 22 : Math.max(22, attrs.text.length * 3.5 + 6);
    const rect = group.querySelector("rect");
    if (rect) {
      rect.setAttribute("x", -w);
      rect.setAttribute("width", 2 * w);
    }
    const polygon = group.querySelector("polygon");
    if (polygon) {
      polygon.setAttribute("x", -w);
      polygon.setAttribute("points", port.symb_cls === "RarrowTextBox"
        ? `${-w},-14 ${w-7},-14 ${w},0 ${w-7},14 ${-w},14`
        : `${-w},0 ${7-w},-14 ${w},-14 ${w},14 ${7-w},14`);
    }
  }
}

// Linear 1/2/5 ticks use geometric midpoints, as in the test-pinned D3 axes.
// Integer tick indices avoid repeated floating-point addition.
function axisTicks(start, stop, count) {
  if (stop < start) return axisTicks(stop, start, count).reverse();
  const raw = (stop - start) / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const error = raw / power;
  const factor = error >= Math.sqrt(50) ? 10 : error >= Math.sqrt(10) ? 5
    : error >= Math.sqrt(2) ? 2 : 1;
  const step = factor * power;
  const precision = start === stop ? 0 : Math.min(20, Math.max(0, -Math.floor(Math.log10(step))));
  const format = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: precision, maximumFractionDigits: precision
  });
  const label = value => ({value,
    text: format.format(Number(value.toFixed(precision)) === 0 ? 0 : value).replace("-", "−")});
  if (start === stop) return [label(start)];
  const reciprocal = step < 1 ? Math.round(1 / step) : 0;
  const first = Math.ceil(reciprocal ? start * reciprocal : start / step);
  const last = Math.floor(reciprocal ? stop * reciprocal : stop / step);
  if (!Number.isSafeInteger(first) || !Number.isSafeInteger(last) || last - first >= 20000) {
    throw new RangeError("Bounds require unsafe tick indices or more than 20000 ticks per axis");
  }
  if (last < first && count >= 0.5 && count < 2) return axisTicks(start, stop, count * 2);
  return Array.from({length: Math.max(0, last - first + 1)}, (_, i) =>
    label(reciprocal ? (first + i) / reciprocal : (first + i) * step));
}

function drawAxes(width, height, domains, x, y) {
  axes.replaceChildren();
  if (!fixedStyle()) return;
  const offset = window.devicePixelRatio > 1 ? 0 : 0.5;
  for (const [isX, start, stop, count, scale, transform] of [
    [true, ...domains[0], width / 80, x, `translate(0,${height - 30})`],
    [false, ...domains[1], height / 50, y, "translate(40,0)"]
  ]) {
    const axis = element(axes, "g", {class: "score-axis", transform, fill: "none",
      "font-size": 10, "font-family": "sans-serif", "text-anchor": isX ? "middle" : "end"});
    const ticks = axisTicks(start, stop, count);
    // The source y scale has a reversed domain and preserves that tick order.
    if (!isX) ticks.reverse();
    for (const {value, text} of ticks) {
      const tick = element(axis, "g", {class: "tick", opacity: 1,
        transform: isX ? `translate(${scale(value) + offset},0)`
          : `translate(0,${scale(value) + offset})`});
      element(tick, "line", {stroke: "currentColor", [isX ? "y2" : "x2"]: isX ? 6 : -6});
      const node = element(tick, "text", {fill: "currentColor",
        [isX ? "y" : "x"]: isX ? 9 : -9, dy: isX ? "0.71em" : "0.32em"});
      node.textContent = text;
    }
  }
}

function linearPosition(value, start, stop, first, last) {
  if (stop < start) return linearPosition(value, stop, start, last, first);
  const fraction = start === stop ? 0.5 : (value - start) / (stop - start);
  return first * (1 - fraction) + last * fraction;
}

function render() {
  cancelAnimationFrame(frame);
  tweens = [];
  const data = views[selector.value];
  const cell = data.soqs.reduce((size, port) => Math.max(
    size, (port.symb_attributes.text || "").length * 7 + 14
  ), 55);
  const fixed = fixedStyle();
  const domains = scoreDomains(data);
  const [[x0, x1], [y0, y1]] = domains;
  if (!manualDomains) boundInputs.forEach((input, i) => {input.value = domains.flat()[i];});
  const width = fixed ? (x1 - x0) * 55 + 40 + 30 + 6 + 6 : (data.max_x + 3) * cell;
  const height = fixed ? (y1 - y0) * 42 + 30 + 6 + 20 + 6 : 32 + data.max_y * 42 + 28;
  const x = column => fixed ? linearPosition(column, x0, x1, 46, width - 36) : (column + 2) * cell;
  const y = lane => fixed ? linearPosition(lane, y1, y0, height - 36, 26) : 32 + lane * 42;
  const baseline = y(fixed ? y1 : data.max_y + 1);
  svg.setAttribute("class", fixed ? "fixed-score" : "");
  drawAxes(width, height, domains, x, y);
  for (const [name, value] of Object.entries({width, height, viewBox: `0 0 ${width} ${height}`})) {
    svg.setAttribute(name, value);
  }
  join(horizontal, data.hlines, d => `${d.y},${d.seq_x_start}`,
    d => element(svg, "line", {class: "h", x1: x(d.seq_x_start), x2: x(d.seq_x_start),
      y1: y(d.y), y2: y(d.y)}),
    (node, d) => {
      node.setAttribute("stroke", fixed ? "lightblue" : d.flavor === "HLineFlavor.CLASSICAL" ? "blue" : "black");
      transition(node, {x1: x(d.seq_x_start), x2: x(d.seq_x_end), y1: y(d.y), y2: y(d.y)});
    },
    (node, d, remove) => transition(node, {x2: x(d.seq_x_start)}, remove)
  );
  join(vertical, data.vlines, d => d.x,
    d => element(svg, "line", {class: "v", stroke: "black", x1: x(d.x), x2: x(d.x),
      y1: y(d.bottom_y), y2: y(d.bottom_y)}),
    (node, d) => transition(node, {x1: x(d.x), x2: x(d.x), y1: y(d.bottom_y), y2: y(d.top_y)}),
    (node, d, remove) => transition(node, {y2: y(d.bottom_y)}, remove)
  );
  const caption = d => typeof d.label === "string" ? d.label : d.label.text;
  join(captions, data.vlines.filter(d => !fixed && caption(d)), d => d.x,
    d => {
      const node = label(svg, x(d.x), y(d.bottom_y) - 22, caption(d), 11);
      node.setAttribute("opacity", 0);
      return node;
    },
    (node, d) => {
      node.textContent = caption(d);
      transition(node, {x: x(d.x), y: y(d.bottom_y) - 22, opacity: 1});
    },
    (node, d, remove) => transition(node, {opacity: 0}, remove)
  );
  join(ports, data.soqs, d => `${d.symb_cls},${d.ident}`,
    d => {
      const startY = d.symb_cls === "Circle" ? 0 : d.symb_cls === "ModPlus" ? baseline : y(d.y);
      const node = portGlyph(svg, d, x(d.seq_x), startY);
      node.setAttribute("opacity", 0);
      return node;
    },
    (node, d) => {
      updatePort(node, d);
      transition(node, {transform: [x(d.seq_x), y(d.y)], opacity: 1});
    },
    (node, d, remove) => {
      const targets = {opacity: 0};
      if (d.symb_cls === "Circle" || d.symb_cls === "ModPlus") {
        targets.transform = [x(d.seq_x), d.symb_cls === "Circle" ? 0 : baseline];
      }
      transition(node, targets, remove);
    }
  );
  // Keep line layers below ports even when a later update introduces new lines.
  for (const record of [...vertical.values()].reverse()) svg.prepend(record.node);
  for (const record of [...horizontal.values()].reverse()) svg.prepend(record.node);
  const started = performance.now();
  function tick(now) {
    const t = Math.max(0, Math.min(1, (now - started) / 750));
    const eased = t < 0.5 ? 4 * t * t * t : 1 - 4 * (1 - t) ** 3;
    for (const {node, from, targets, finish} of tweens) {
      for (const [name, target] of Object.entries(targets)) {
        if (name === "transform") {
          const xy = target.map((end, i) => from[name][i] + (end - from[name][i]) * eased);
          node.setAttribute(name, `translate(${xy[0]},${xy[1]})`);
        } else {
          node.setAttribute(name, from[name] + (target - from[name]) * eased);
        }
      }
      if (t === 1) finish();
    }
    if (t < 1) frame = requestAnimationFrame(tick);
  }
  frame = requestAnimationFrame(tick);
}

selector.addEventListener("change", render);
styleSelector.addEventListener("change", render);
document.getElementById("score-bounds").addEventListener("submit", applyBounds);
document.getElementById("score-fit").addEventListener("click", () => {
  manualDomains = null;
  render();
});
document.getElementById("score-load").addEventListener("submit", event => {
  event.preventDefault();
  loadScore(document.getElementById("score-url").value);
});
render();
</script>
</html>
"""


def export_register_score_html(data: MusicalScoreData) -> str:
    """Export an offline score with keyed, 750 ms cubic-in-out view transitions.

    SVG nodes retain identity across view changes. Fresh lines grow, ports fade
    in, and retired nodes complete their exit animation before removal. Rapid
    changes resume from current geometry through the same shared score data.
    """
    payload = json.dumps(
        {"sequential": data, "dependency": _dependency_view(data)}, ensure_ascii=False
    )
    # Prevent a register label from closing the script data element.
    return _PAGE.replace("__SCORE_RECORDS__", payload.replace("<", "\\u003c"))
