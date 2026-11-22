// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

const VIEWS = [
  { axis: 2, name: "Axial", cols: 0, rows: 1, labels: ["L", "R", "A", "P"] },
  { axis: 1, name: "Coronal", cols: 0, rows: 2, labels: ["L", "R", "S", "I"] },
  { axis: 0, name: "Sagittal", cols: 1, rows: 2, labels: ["P", "A", "S", "I"] },
];

const LAYER = { fixed: 0, warped: 1, before: 2, edges: 3 };
const CACHE_LIMIT = 200;
const EDGE_COLOUR = [255, 212, 0];
const EDGE_ALPHA = 0.85;
const FLICKER_MS = 1000;

export function createViewer({ root, fetchSlice }) {
  const q = (selector) => root.querySelector(selector);
  const qa = (selector) => [...root.querySelectorAll(selector)];
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const views = VIEWS.map((v) => {
    const figure = q('.fw-view[data-axis="' + v.axis + '"]');
    const canvas = figure.querySelector("canvas");
    return {
      ...v,
      figure,
      canvas,
      ctx: canvas.getContext("2d"),
      slider: figure.querySelector("input"),
      caption: figure.querySelector("figcaption"),
      tmp: document.createElement("canvas"),
      token: 0,
      painted: false,
    };
  });
  const cache = new Map();
  const state = {
    info: null,
    generation: 0,
    pos: [0, 0, 0],
    mode: "blend",
    when: "after",
    blend: 0.5,
    flickerMoving: false,
    playing: false,
    timer: 0,
    scheduled: false,
    pending: 0,
  };

  function setPressed(buttons, test) {
    for (const b of buttons) {
      const on = test(b);
      b.setAttribute("aria-pressed", String(on));
      b.classList.toggle("fw-secondary", !on);
    }
  }

  function movingLayer() {
    return state.when === "before" ? LAYER.before : LAYER.warped;
  }

  function layersNeeded() {
    const moving = movingLayer();
    if (state.mode === "blend") {
      if (state.blend <= 0) return [LAYER.fixed];
      if (state.blend >= 1) return [moving];
      return [LAYER.fixed, moving];
    }
    if (state.mode === "edges") return [moving, LAYER.edges];
    return [LAYER.fixed, moving];
  }

  function getSlice(axis, index, layer) {
    const key = state.generation + ":" + axis + ":" + index + ":" + layer;
    let hit = cache.get(key);
    if (!hit) {
      hit = fetchSlice(axis, index, layer);
      hit.catch(() => cache.delete(key));
      cache.set(key, hit);
      if (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value);
    }
    return hit;
  }

  function geometry(view) {
    const { dims, spacing } = state.info;
    return {
      width: dims[view.cols],
      height: dims[view.rows],
      physical: (dims[view.cols] * spacing[view.cols]) / (dims[view.rows] * spacing[view.rows]),
    };
  }

  function compose(view, slices, width, height) {
    const out = new ImageData(width, height);
    const px = out.data;
    const fixed = slices[LAYER.fixed];
    const moving = slices[movingLayer()];
    const edges = slices[LAYER.edges];
    const showMoving = state.mode === "flicker" ? state.flickerMoving : false;
    const t = state.blend;
    const [ar, ag, ab] = EDGE_COLOUR;
    for (let r = 0, i = 0; r < height; r++) {
      for (let c = 0; c < width; c++, i++) {
        let v;
        let red;
        let green;
        let blue;
        if (state.mode === "blend") {
          if (!moving) v = fixed[i];
          else if (!fixed) v = moving[i];
          else v = fixed[i] + (moving[i] - fixed[i]) * t;
          red = green = blue = v;
        } else if (state.mode === "flicker") {
          v = showMoving ? moving[i] : fixed[i];
          red = green = blue = v;
        } else {
          v = moving[i];
          const a = (edges[i] / 255) * EDGE_ALPHA;
          red = v + (ar - v) * a;
          green = v + (ag - v) * a;
          blue = v + (ab - v) * a;
        }
        const p = i * 4;
        px[p] = red;
        px[p + 1] = green;
        px[p + 2] = blue;
        px[p + 3] = 255;
      }
    }
    return out;
  }

  function fitCanvas(view) {
    const { physical } = geometry(view);
    const dpr = window.devicePixelRatio || 1;
    const cssWidth = view.canvas.clientWidth;
    if (!cssWidth) return false;
    const width = Math.max(1, Math.round(cssWidth * dpr));
    const height = Math.max(1, Math.round(width / physical));
    if (view.canvas.width !== width || view.canvas.height !== height) {
      view.canvas.width = width;
      view.canvas.height = height;
    }
    return true;
  }

  function overlay(view) {
    const { ctx, canvas } = view;
    const { dims, oriented } = state.info;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width;
    const h = canvas.height;
    const x = ((state.pos[view.cols] + 0.5) / dims[view.cols]) * w;
    const y = ((dims[view.rows] - 1 - state.pos[view.rows] + 0.5) / dims[view.rows]) * h;
    const size = 18 * dpr;
    const m = 4 * dpr;
    const [left, right, top, bottom] = view.labels;
    const chips = oriented
      ? [
          [left, m + size / 2, h / 2],
          [right, w - m - size / 2, h / 2],
          [top, w / 2, m + size / 2],
          [bottom, w / 2, h - m - size / 2],
        ]
      : [];
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, w, h);
    for (const [, cx, cy] of chips) ctx.rect(cx - size / 2 - 3 * dpr, cy - size / 2 - 3 * dpr, size + 6 * dpr, size + 6 * dpr);
    ctx.clip("evenodd");
    ctx.lineCap = "butt";
    for (const [width, style] of [
      [3 * dpr, "rgba(0, 0, 0, 0.35)"],
      [1 * dpr, "rgba(255, 255, 255, 0.85)"],
    ]) {
      ctx.lineWidth = width;
      ctx.strokeStyle = style;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    ctx.restore();
    ctx.save();
    ctx.font = "600 " + Math.round(12 * dpr) + "px sans-serif";
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    for (const [text, cx, cy] of chips) {
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.beginPath();
      ctx.roundRect(cx - size / 2, cy - size / 2, size, size, 4 * dpr);
      ctx.fill();
      ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
      ctx.fillText(text, cx, cy + 0.5 * dpr);
    }
    ctx.restore();
  }

  function paint(view, image) {
    if (!fitCanvas(view)) return;
    view.tmp.width = image.width;
    view.tmp.height = image.height;
    view.tmp.getContext("2d").putImageData(image, 0, 0);
    view.painted = true;
    redraw(view);
  }

  function redraw(view) {
    if (!view.painted || !state.info || !fitCanvas(view)) return;
    const { ctx, canvas } = view;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(view.tmp, 0, 0, canvas.width, canvas.height);
    overlay(view);
  }

  function settle(generation) {
    if (generation !== state.generation) return;
    state.pending -= 1;
    if (state.pending <= 0) {
      state.pending = 0;
      root.dataset.state = "ready";
    }
  }

  async function render(view) {
    const generation = state.generation;
    const token = ++view.token;
    const layers = layersNeeded();
    const { width, height } = geometry(view);
    state.pending += 1;
    root.dataset.state = "loading";
    try {
      const data = await Promise.all(layers.map((l) => getSlice(view.axis, state.pos[view.axis], l)));
      if (token !== view.token || generation !== state.generation) return;
      const slices = {};
      layers.forEach((l, i) => (slices[l] = data[i]));
      paint(view, compose(view, slices, width, height));
    } catch {
      return;
    } finally {
      settle(generation);
    }
  }

  function renderAll() {
    if (!state.info) return;
    for (const view of views) render(view);
    updateText();
  }

  function schedule() {
    if (state.scheduled) return;
    state.scheduled = true;
    requestAnimationFrame(() => {
      state.scheduled = false;
      renderAll();
    });
  }

  function updateText() {
    const { dims } = state.info;
    for (const view of views) {
      const n = dims[view.axis];
      const i = state.pos[view.axis];
      view.caption.textContent = view.name + " " + (i + 1) + " / " + n;
      view.slider.value = String(i);
      view.slider.setAttribute("aria-valuetext", view.name + " slice " + (i + 1) + " of " + n);
      view.canvas.setAttribute("aria-label", view.name + " view, slice " + (i + 1) + " of " + n + ". " + describeMode());
    }
    const registration = state.when === "before" ? "before registration" : "after registration";
    q("#fw-flicker-state").textContent =
      "Showing " + (state.flickerMoving ? "your image " + registration : "the template");
  }

  function describeMode() {
    const registration = state.when === "before" ? "before registration" : "after registration";
    if (state.mode === "edges") return "Edges of the template over your image " + registration + ".";
    if (state.mode === "flicker") return "The template and your image " + registration + ", alternating.";
    return "The template blended with your image " + registration + ".";
  }

  function setPos(pos) {
    const { dims } = state.info;
    const next = pos.map((p, a) => Math.min(dims[a] - 1, Math.max(0, Math.round(p))));
    if (next.every((p, a) => p === state.pos[a])) return;
    state.pos = next;
    schedule();
  }

  function pointer(view, event) {
    const rect = view.canvas.getBoundingClientRect();
    const { dims } = state.info;
    const fx = (event.clientX - rect.left) / rect.width;
    const fy = (event.clientY - rect.top) / rect.height;
    const next = [...state.pos];
    next[view.cols] = Math.floor(fx * dims[view.cols]);
    next[view.rows] = dims[view.rows] - 1 - Math.floor(fy * dims[view.rows]);
    setPos(next);
  }

  function stopFlicker() {
    clearInterval(state.timer);
    state.timer = 0;
  }

  function startFlicker() {
    stopFlicker();
    if (state.mode !== "flicker" || !state.playing || document.hidden) return;
    state.timer = setInterval(() => {
      state.flickerMoving = !state.flickerMoving;
      schedule();
    }, FLICKER_MS);
  }

  function setPlaying(on) {
    state.playing = on;
    q("#fw-flicker-play").textContent = on ? "Pause" : "Play";
    startFlicker();
  }

  function showControls() {
    for (const name of ["blend", "flicker", "edges"]) {
      q("#fw-ctl-" + name).hidden = state.mode !== name;
    }
  }

  function setMode(mode) {
    state.mode = mode;
    setPressed(qa("#fw-modes button"), (b) => b.dataset.mode === mode);
    showControls();
    if (mode === "flicker") {
      state.flickerMoving = false;
      setPlaying(!reducedMotion.matches);
    } else {
      stopFlicker();
    }
    schedule();
  }

  function setWhen(when) {
    state.when = when;
    setPressed(qa("#fw-when button"), (b) => b.dataset.when === when);
    schedule();
  }

  for (const b of qa("#fw-modes button")) b.addEventListener("click", () => setMode(b.dataset.mode));
  for (const b of qa("#fw-when button")) b.addEventListener("click", () => setWhen(b.dataset.when));
  q("#fw-blend").addEventListener("input", (e) => {
    state.blend = Number(e.target.value) / 100;
    schedule();
  });
  q("#fw-flicker-play").addEventListener("click", () => setPlaying(!state.playing));
  q("#fw-flicker-switch").addEventListener("click", () => {
    state.flickerMoving = !state.flickerMoving;
    schedule();
  });
  document.addEventListener("visibilitychange", startFlicker);

  for (const view of views) {
    view.slider.addEventListener("input", () => {
      const next = [...state.pos];
      next[view.axis] = Number(view.slider.value);
      setPos(next);
    });
    view.canvas.addEventListener("pointerdown", (e) => {
      if (!state.info || (e.pointerType === "mouse" && e.button !== 0)) return;
      view.canvas.setPointerCapture(e.pointerId);
      pointer(view, e);
    });
    view.canvas.addEventListener("pointermove", (e) => {
      if (state.info && view.canvas.hasPointerCapture(e.pointerId)) pointer(view, e);
    });
    new ResizeObserver(() => redraw(view)).observe(view.canvas);
  }

  const button = q("#fw-fullscreen");
  const viewsBox = q(".fw-views");
  const GAP = 12;
  const nativeFullscreen = () => document.fullscreenElement ?? document.webkitFullscreenElement ?? null;

  function isFull() {
    return root.classList.contains("fw-full");
  }

  function clearLayout() {
    viewsBox.removeAttribute("data-layout");
    for (const view of views) view.figure.style.removeProperty("width");
  }

  function layoutFull() {
    if (!isFull() || !state.info) return;
    const { dims, spacing } = state.info;
    const size = views.map((view) => ({
      w: dims[view.cols] * spacing[view.cols],
      h: dims[view.rows] * spacing[view.rows],
    }));
    const first = views[0];
    const extra = first.figure.getBoundingClientRect().height - first.canvas.getBoundingClientRect().height || 64;
    const note = q(".fw-note");
    const rootStyle = getComputedStyle(root);
    const width = viewsBox.clientWidth;
    const height =
      root.clientHeight -
      parseFloat(rootStyle.paddingBottom) -
      note.offsetHeight -
      parseFloat(getComputedStyle(note).marginTop) -
      viewsBox.offsetTop;
    const area = (scales) => scales.reduce((sum, s, i) => sum + s * s * size[i].w * size[i].h, 0);
    const rowScale = Math.min(
      (width - 2 * GAP) / (size[0].w + size[1].w + size[2].w),
      (height - extra) / Math.max(size[0].h, size[1].h, size[2].h),
    );
    const lower = Math.min(
      (width - GAP) / (size[1].w + size[2].w),
      (height - 2 * extra - GAP) / (2 * Math.max(size[1].h, size[2].h)),
    );
    const upper = Math.min(
      width / size[0].w,
      (height - 2 * extra - GAP - lower * Math.max(size[1].h, size[2].h)) / size[0].h,
    );
    const rows = [rowScale, rowScale, rowScale];
    const grid = [upper, lower, lower];
    const useGrid = area(grid) > area(rows);
    const scales = (useGrid ? grid : rows).map((s) => Math.max(0.1, s));
    viewsBox.dataset.layout = useGrid ? "grid" : "row";
    views.forEach((view, i) => {
      view.figure.style.width = Math.floor(size[i].w * scales[i]) + "px";
    });
  }

  function syncFull() {
    const on = isFull();
    button.setAttribute("aria-pressed", String(on));
    button.textContent = on ? "Exit full screen" : "Full screen";
    button.classList.toggle("fw-secondary", !on);
    document.documentElement.classList.toggle("fw-noscroll", on && !nativeFullscreen());
    if (on) layoutFull();
    else clearLayout();
  }

  let inerted = [];

  function enterOverlay() {
    root.classList.add("fw-overlay");
    root.setAttribute("role", "dialog");
    root.setAttribute("aria-modal", "true");
    root.tabIndex = -1;
    let node = root;
    while (node.parentElement && node !== document.body) {
      for (const sibling of node.parentElement.children) {
        if (sibling !== node && !sibling.inert) {
          sibling.inert = true;
          inerted.push(sibling);
        }
      }
      node = node.parentElement;
    }
    root.focus({ preventScroll: true });
  }

  function leaveOverlay() {
    root.classList.remove("fw-overlay");
    root.removeAttribute("role");
    root.removeAttribute("aria-modal");
    root.removeAttribute("tabindex");
    for (const sibling of inerted) sibling.inert = false;
    inerted = [];
  }

  async function setFull(on) {
    if (on === isFull()) return;
    if (on) {
      root.classList.add("fw-full");
      const request = root.requestFullscreen ?? root.webkitRequestFullscreen;
      if (request) {
        try {
          await request.call(root);
        } catch {
          enterOverlay();
        }
      } else {
        enterOverlay();
      }
    } else {
      root.classList.remove("fw-full");
      leaveOverlay();
      if (nativeFullscreen()) {
        try {
          await (document.exitFullscreen ?? document.webkitExitFullscreen).call(document);
        } catch {}
      }
      button.focus({ preventScroll: true });
    }
    syncFull();
    schedule();
  }

  function onFullscreenChange() {
    if (!nativeFullscreen() && isFull() && !root.classList.contains("fw-overlay")) {
      root.classList.remove("fw-full");
      button.focus({ preventScroll: true });
    }
    syncFull();
    schedule();
  }

  button.addEventListener("click", () => setFull(!isFull()));
  document.addEventListener("fullscreenchange", onFullscreenChange);
  document.addEventListener("webkitfullscreenchange", onFullscreenChange);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && isFull() && root.classList.contains("fw-overlay")) setFull(false);
  });
  window.addEventListener("resize", layoutFull);
  new ResizeObserver(layoutFull).observe(root);

  function hide() {
    if (isFull()) setFull(false);
    state.generation += 1;
    state.info = null;
    cache.clear();
    stopFlicker();
    root.hidden = true;
    for (const view of views) {
      view.painted = false;
      view.token += 1;
    }
  }

  function show(info) {
    hide();
    state.info = info;
    const { dims, spacing } = info;
    state.pos = dims.map((n) => Math.floor(n / 2));
    state.pending = 0;
    for (const view of views) {
      const g = geometry(view);
      view.figure.style.setProperty("--fw-grow", String(Math.round(dims[view.cols] * spacing[view.cols])));
      view.canvas.style.aspectRatio = g.physical.toFixed(4);
      view.slider.min = "0";
      view.slider.max = String(dims[view.axis] - 1);
      view.slider.setAttribute("aria-label", view.name + " slice");
    }
    q("#fw-when").hidden = !info.before;
    state.when = "after";
    setPressed(qa("#fw-when button"), (b) => b.dataset.when === "after");
    state.blend = Number(q("#fw-blend").value) / 100;
    root.hidden = false;
    setMode(state.mode);
    renderAll();
  }

  return { show, hide };
}
