# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix physical circuit viewer.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/circuit/circuit_diagram_html.cc

"""Browser viewport for a physical circuit's serialized SVG geometry."""

VIEWER_SCRIPT = r"""
    const input = JSON.parse(document.getElementById('circuit-data').textContent);
    const svgNS = 'http://www.w3.org/2000/svg';
    const parser = new DOMParser();

    class WireObject {
        constructor(content) {
            this.content = content;
            this.element = undefined;
        }

        ensureElementExists() {
            if (this.element !== undefined) return;
            const group = parser.parseFromString(
                '<g xmlns="' + svgNS + '">' + this.content + '</g>', 'image/svg+xml'
            ).documentElement;
            this.element = document.importNode(group, true);
        }
    }

    class BoundedObject extends WireObject {
        constructor(start, content, farContent) {
            super(content);
            this.start_x = start;
            this.far = new WireObject(farContent);
        }

        ensureFarElementExists() {
            this.far.ensureElementExists();
            this.far_element = this.far.element;
        }
    }

    const objects = input.objects.map(row => new BoundedObject(...row));
    const wire_transitions = input.transitions;
    const wire_objects = input.wires.map(row => row.map(markup => new WireObject(markup)));
    const min_x = 0;
    const min_y = 0;
    const max_x = input.width;
    const max_y = input.height;

    const target_objects = 2000;
    const target_wire_transitions = 20;
    const view_pad = 40;
    const max_view_width = (max_x - min_x) * 3;
    const max_view_height = max_y - min_y;
    let viewBox = { x: min_x, y: min_y, width: max_x - min_x, height: max_y - min_y };

    document.addEventListener('DOMContentLoaded', () => {
        let old_shown_start_i = 0;
        let old_shown_end_i = -1;
        let old_shown_stride = 1;
        let old_shown_far = false;
        let old_shown_wire_start_i = [];
        let old_shown_wire_end_i = [];

        const svg = document.getElementById('viewportSvg');
        const zoomStep = 1.002;

        let dragPrevPoint = undefined;

        const updateViewBox = () => {
            viewBox.width = Math.min(viewBox.width, max_view_width);
            viewBox.height = Math.min(viewBox.height, max_view_height);

            let w2h = svg.clientHeight / svg.clientWidth;
            if (Number.isFinite(w2h) && w2h > 0 && w2h * max_view_width > max_view_height) {
                viewBox.height = viewBox.width * w2h;
            }
            viewBox.width = Math.min(viewBox.width, max_view_width);
            viewBox.height = Math.min(viewBox.height, max_view_height);

            viewBox.x = Math.max(viewBox.x, min_x - viewBox.width * 0.5);
            viewBox.x += viewBox.width;
            viewBox.x = Math.min(viewBox.x, max_x + viewBox.width * 0.5);
            viewBox.x -= viewBox.width;

            viewBox.y = Math.max(viewBox.y, min_y);
            viewBox.y += viewBox.height;
            viewBox.y = Math.min(viewBox.y, max_y);
            viewBox.y -= viewBox.height;

            let hash = `x=${viewBox.x},y=${viewBox.y},w=${viewBox.width},h=${viewBox.height}`;
            try {
                let new_url = new URL(window.location);
                new_url.hash = hash;
                history.replaceState(null, '', new_url);
            } catch {
                window.location.hash = hash;
            }
            checkAndRemoveWires();
            checkAndRemoveRects();
        };

        function find_wire_start_index(q) {
            let ts = wire_transitions[q];
            let start_i = 0;
            let end_i = ts.length - 1;
            while (start_i < end_i) {
                let mid_i = (start_i + end_i) >>1;
                if (ts[mid_i] < viewBox.x) {
                    start_i = mid_i + 1;
                } else {
                    end_i = mid_i;
                }
            }
            if (start_i > 0) {
                start_i -= 1;
            }
            if (start_i % 2 === 1) {
                start_i -= 1;
            }
            return start_i / 2;
        }

        function find_wire_end_index(q) {
            let ts = wire_transitions[q];
            let start_i = 0;
            let end_i = ts.length - 1;
            while (start_i < end_i) {
                let mid_i = (start_i + end_i) >> 1;
                if (ts[mid_i] < viewBox.x + viewBox.width) {
                    start_i = mid_i + 1;
                } else {
                    end_i = mid_i;
                }
            }
            if (end_i > 0 && ts[end_i] >= viewBox.x + viewBox.width) {
                end_i -= 1;
            }
            if (end_i % 2 == 1) {
                end_i += 1;
            }
            if (end_i == ts.length) {
                end_i -= 2;
            }
            return end_i / 2;
        }

        function find_first_shown_object_index() {
            let start_i = 0;
            let end_i = objects.length - 1;
            while (start_i < end_i) {
                let mid_i = (start_i + end_i) >>1;
                if (objects[mid_i].start_x < viewBox.x - view_pad) {
                    start_i = mid_i + 1;
                } else {
                    end_i = mid_i;
                }
            }
            if (start_i > 0) {
                start_i -= 1;
            }
            return start_i;
        }

        function find_last_shown_object_index() {
            let start_i = 0;
            let end_i = objects.length - 1;
            while (start_i < end_i) {
                let mid_i = (start_i + end_i) >> 1;
                if (objects[mid_i].start_x < viewBox.x + viewBox.width + view_pad) {
                    start_i = mid_i + 1;
                } else {
                    end_i = mid_i;
                }
            }
            if (end_i > 0 && objects[end_i].start_x >= viewBox.x + viewBox.width + view_pad) {
                end_i -= 1;
            }
            return end_i;
        }

        const checkAndRemoveWires = () => {
            while (old_shown_wire_start_i.length < wire_transitions.length) {
                old_shown_wire_start_i.push(-1);
                old_shown_wire_end_i.push(-1);
            }
            for (let q = 0; q < wire_transitions.length; q++) {
                let start_i = find_wire_start_index(q);
                let end_i = find_wire_end_index(q);
                if (end_i - start_i > target_wire_transitions) {
                    start_i = -1;
                    end_i = -1;
                }

                for (let i = start_i; i >= 0 && i <= end_i; i++) {
                    if (!(i >= old_shown_wire_start_i[q] && i <= old_shown_wire_end_i[q])) {
                        wire_objects[q][i].ensureElementExists();
                        svg.appendChild(wire_objects[q][i].element)
                    }
                }
                if (old_shown_wire_start_i[q] != start_i || old_shown_wire_end_i[q] != end_i) {
                    for (let i = old_shown_wire_start_i[q]; i >= 0 && i <= old_shown_wire_end_i[q]; i++) {
                        if (!(i >= start_i && i <= end_i)) {
                            wire_objects[q][i].element.remove();
                        }
                    }
                    old_shown_wire_start_i[q] = start_i;
                    old_shown_wire_end_i[q] = end_i;
                }
            }
        };
        const checkAndRemoveRects = () => {
            let start_i = find_first_shown_object_index();
            let end_i = find_last_shown_object_index();
            let stride = 1;
            while (end_i - start_i > target_objects * stride) {
                stride *= 2;
            }
            start_i = Math.floor(start_i / stride) * stride;
            let far = stride > 1 || viewBox.width / svg.clientWidth > 2.5 || viewBox.height / svg.clientHeight > 2.5;

            if (stride !== old_shown_stride || old_shown_far != far) {
                for (let i = old_shown_start_i; i <= old_shown_end_i; i += old_shown_stride) {
                    if (old_shown_far) {
                        objects[i].far_element.remove();
                    } else {
                        objects[i].element.remove();
                    }
                }
                old_shown_start_i = 0;
                old_shown_end_i = -1;
            }
            for (let i = start_i; i <= end_i; i += stride) {
                if (!(i >= old_shown_start_i && i <= old_shown_end_i)) {
                    if (far) {
                        objects[i].ensureFarElementExists();
                        svg.appendChild(objects[i].far_element)
                    } else {
                        objects[i].ensureElementExists();
                        svg.appendChild(objects[i].element)
                    }
                }
            }
            svg.setAttribute('viewBox', `${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`);
            for (let i = old_shown_start_i; i <= old_shown_end_i; i += old_shown_stride) {
                if (!(i >= start_i && i <= end_i)) {
                    if (old_shown_far) {
                        objects[i].far_element.remove();
                    } else {
                        objects[i].element.remove();
                    }
                }
            }
            old_shown_start_i = start_i;
            old_shown_end_i = end_i;
            old_shown_stride = stride;
            old_shown_far = far;
        };

        svg.addEventListener('mousedown', e => {
            if (e.button === 0) {
                dragPrevPoint = {x: e.clientX, y: e.clientY};
                svg.style.cursor = 'grabbing';
                e.preventDefault();
            }
        });

        window.addEventListener('mousemove', e => {
            if (dragPrevPoint === undefined || e.button !== 0) {
                return;
            }

            e.preventDefault();
            const dx = (dragPrevPoint.x - e.clientX) * (viewBox.width / svg.clientWidth);
            const dy = (dragPrevPoint.y - e.clientY) * (viewBox.height / svg.clientHeight);

            viewBox.x += dx;
            viewBox.y += dy;
            dragPrevPoint = {x: e.clientX, y: e.clientY};

            requestAnimationFrame(updateViewBox);
        });

        window.addEventListener('mouseup', () => {
            if (dragPrevPoint !== undefined) {
                dragPrevPoint = undefined;
                svg.style.cursor = 'grab';
            }
        });

        svg.addEventListener('wheel', e => {
            e.preventDefault();
            const mousePointToSvg = svg.createSVGPoint();
            mousePointToSvg.x = e.clientX;
            mousePointToSvg.y = e.clientY;

            const screenToSvgMatrix = svg.getScreenCTM().inverse();
            const svgPoint = mousePointToSvg.matrixTransform(screenToSvgMatrix);
            const desiredZoomFactor = Math.pow(zoomStep, e.deltaY);
            const newWidth = Math.min(viewBox.width * desiredZoomFactor, max_view_width)
            const newHeight = Math.min(viewBox.height * desiredZoomFactor, max_view_height)
            const zoomFactor = Math.max(newWidth / viewBox.width, newHeight / viewBox.height);
            viewBox.x = svgPoint.x - (svgPoint.x - viewBox.x) * zoomFactor;
            viewBox.y = svgPoint.y - (svgPoint.y - viewBox.y) * zoomFactor;
            viewBox.width *= zoomFactor;
            viewBox.height *= zoomFactor;

            requestAnimationFrame(updateViewBox);
        });

        svg.style.cursor = 'grab';

        let v = window.location.hash;
        if (v.startsWith('#')) {
            v = v.substring(1);
        }
        let pieces = v.split(',');
        for (let piece of pieces) {
            let terms = piece.split('=');
            if (terms.length === 2) {
                try {
                    const value = Number(terms[1]);
                    if (!terms[1] || !Number.isFinite(value)) continue;
                    if ((terms[0] === 'w' || terms[0] === 'h') && value <= 0) continue;
                    if (terms[0] === 'x') {
                        viewBox.x = value
                    }
                    if (terms[0] === 'y') {
                        viewBox.y = value
                    }
                    if (terms[0] === 'w') {
                        viewBox.width = value
                    }
                    if (terms[0] === 'h') {
                        viewBox.height = value
                    }
                } catch (ex) {
                    console.error(ex);
                }
            }
        }
        requestAnimationFrame(updateViewBox);

        new ResizeObserver(() => requestAnimationFrame(updateViewBox)).observe(svg);
    });
"""
