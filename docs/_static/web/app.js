// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

import { createViewer } from "./viewer.js";
import {
  clip,
  dismissUnfinished,
  endRun,
  featureList,
  isoDims,
  limitsObject,
  mailtoHref,
  niftiGeometry,
  platformOf,
  reportText,
  saveRun,
  stageText,
  supportHint,
  takeUnfinished,
} from "./diag.js";

const el = (id) => document.getElementById(id);
const here = new URL("./", import.meta.url);
const params = new URLSearchParams(location.search);
const autorun = params.get("autorun");

const SAMPLES = {
  fixed: "sample/mni152nlin2009basym-head-1mm.nii.gz",
  moving: "sample/sample-t1w-defaced.nii.gz",
};

const BROWSERS =
  "WebGPU runs in Chrome and Edge 113 or later on Windows, macOS and ChromeOS, Chrome 121 or " +
  "later on Android, Firefox 141 or later on Windows, and Safari 26 or later on macOS, iOS and " +
  "iPadOS.";

const STALL_MS = 180000;

const PHASES = {
  gpu: ["Opening the GPU", 0.0, 0.02],
  read: ["Reading the images", 0.02, 0.05],
  prepare: ["Resampling and preparing the images", 0.05, 0.1],
  rigid: ["Rigid", 0.1, 0.25],
  affine: ["Affine", 0.25, 0.4],
  syn: ["SyN", 0.4, 0.9],
  warp: ["Resampling the moving image onto the fixed image's grid", 0.9, 0.93],
  write: ["Writing the output files", 0.93, 0.99],
  done: ["Done", 1.0, 1.0],
};

const state = {
  inputs: { fixed: null, moving: null },
  worker: null,
  ready: false,
  running: false,
  started: 0,
  timer: 0,
  cancelTimer: 0,
  stallTimer: 0,
  lastKey: "",
  zipUrl: null,
  jsAdapter: null,
  adapterLabel: null,
  runId: 0,
  posted: false,
  restarts: 0,
  gpu: "starting",
  sample: false,
  done: false,
  progressAt: 0,
  savedAt: 0,
  stalled: false,
};

const diag = {
  platform: platformOf(navigator.userAgent, navigator.maxTouchPoints ?? 0),
  hints: null,
  pageGpu: Boolean(navigator.gpu),
  adapter: null,
  workerAdapter: null,
  device: null,
  version: null,
  events: [],
  droppedEvents: 0,
  console: [],
  run: null,
  memory: null,
  error: null,
  previous: null,
  report: null,
};

const sliceRequests = new Map();
let sliceCounter = 0;

function fetchSlice(axis, index, layer) {
  return new Promise((resolve, reject) => {
    if (!state.worker) {
      reject(new Error("no worker"));
      return;
    }
    const id = ++sliceCounter;
    sliceRequests.set(id, { resolve, reject });
    state.worker.postMessage({ type: "slice", id, axis, index, layer });
  });
}

function dropSliceRequests() {
  for (const { reject } of sliceRequests.values()) reject(new Error("the worker was replaced"));
  sliceRequests.clear();
}

const viewer = createViewer({ root: document.getElementById("fw-viewer"), fetchSlice });

function log(line) {
  const box = el("fw-log");
  const t = state.running ? `[${clock(performance.now() - state.started)}] ` : "";
  box.textContent += `${t}${line}\n`;
  box.scrollTop = box.scrollHeight;
  console.log(`fastwarp: ${line}`);
}

function message(text, kind = "info") {
  const box = el("fw-message");
  box.textContent = text;
  box.dataset.kind = kind;
  box.hidden = !text;
}

function clock(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function bytes(n) {
  if (n == null) return "unknown";
  if (n >= 2 ** 30) return `${(n / 2 ** 30).toFixed(1)} GiB`;
  if (n >= 2 ** 20) return `${(n / 2 ** 20).toFixed(0)} MiB`;
  if (n >= 2 ** 10) return `${(n / 2 ** 10).toFixed(0)} KiB`;
  return `${n} bytes`;
}

function seconds(ms) {
  return Math.round(ms / 100) / 10;
}

function isMobile() {
  if (navigator.userAgentData) return navigator.userAgentData.mobile;
  const ua = navigator.userAgent;
  return /Android|iPhone|iPad|iPod|Mobile/i.test(ua) || (navigator.maxTouchPoints > 1 && /Macintosh/.test(ua));
}

async function exists(path) {
  try {
    const r = await fetch(new URL(path, here), { method: "HEAD", cache: "no-store" });
    return r.ok;
  } catch {
    return false;
  }
}

function result(status, extra = {}) {
  const out = { status, ...extra };
  const box = el("fw-result");
  box.textContent = JSON.stringify(out);
  box.dataset.status = status;
  if (autorun) {
    console.log(`FASTWARP_RESULT ${JSON.stringify(out)}`);
    document.title = `FASTWARP_${status.toUpperCase()}`;
  }
}

function browserInfo() {
  const d = navigator.userAgentData;
  return {
    user_agent: navigator.userAgent,
    brands: d?.brands?.map((b) => `${b.brand} ${b.version}`) ?? null,
    mobile: d?.mobile ?? diag.platform.mobile,
    platform: d?.platform || diag.platform.os,
    platform_version: diag.hints?.platformVersion || null,
    model: diag.hints?.model || null,
    architecture: diag.hints?.architecture || null,
    cores: navigator.hardwareConcurrency ?? null,
    device_memory_gb: navigator.deviceMemory ?? null,
    touch_points: navigator.maxTouchPoints ?? 0,
    secure_context: window.isSecureContext,
    cross_origin_isolated: window.crossOriginIsolated ?? false,
  };
}

function lastPanic() {
  for (let i = diag.console.length - 1; i >= 0; i--) {
    const text = diag.console[i].text ?? "";
    const at = text.indexOf("internal error:");
    if (at >= 0) return text.slice(at + "internal error:".length).trim();
  }
  return null;
}

function stageOf(run) {
  return run?.stage ?? null;
}

function makeReport(kind, text) {
  const now = performance.now();
  const run = diag.run;
  const end = run?.end ?? now;
  return {
    schema: 1,
    kind,
    message: clip(text, 2000),
    time_utc: `${new Date().toISOString().slice(0, 16)}Z`,
    fastwarp: diag.version,
    page: location.pathname,
    browser: browserInfo(),
    webgpu: {
      navigator_gpu: diag.pageGpu,
      adapter: diag.adapter?.info ?? null,
      features: diag.adapter?.features ?? null,
      limits: diag.adapter?.limits ?? null,
      wgsl_language_features: diag.adapter?.wgsl ?? null,
      worker_adapter: diag.workerAdapter,
      device: diag.device,
    },
    run: run
      ? {
          grid: run.grid,
          fixed: run.fixed,
          moving: run.moving,
          registration_grid: run.registrationGrid,
          started_utc: run.startedUtc,
          stage: stageOf(run),
          elapsed_s: seconds(end - run.started),
          since_progress_s: seconds(end - run.progressAt),
          outcome: run.outcome ?? "running",
          timeline: run.timeline,
        }
      : null,
    memory: {
      wasm_bytes: diag.memory,
      js_heap_bytes: performance.memory?.usedJSHeapSize ?? null,
    },
    errors: {
      name: diag.error?.name ?? null,
      message: diag.error?.message ?? null,
      phase: diag.error?.phase ?? null,
      stack: diag.error?.stack ?? null,
      panic: lastPanic(),
      gpu: diag.events,
      gpu_events_dropped: diag.droppedEvents,
      console: diag.console,
    },
    previous_run: diag.previous,
  };
}

function refreshReport() {
  if (!diag.report) return;
  const report = makeReport(diag.report.kind, diag.report.text);
  el("fw-mail").href = mailtoHref(report).href;
  if (el("fw-report-box").open) el("fw-report-text").textContent = reportText(report);
}

function showReport(kind, text, { tone = "error", dismiss = false } = {}) {
  diag.report = { kind, text };
  const box = el("fw-report");
  box.dataset.kind = tone;
  el("fw-dismiss").hidden = !dismiss;
  el("fw-copy-state").textContent = "";
  box.hidden = false;
  refreshReport();
  el("fw-message").scrollIntoView({ block: "nearest" });
}

function hideReport() {
  diag.report = null;
  el("fw-report").hidden = true;
  el("fw-report-text").textContent = "";
}

function copyFallback(text) {
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.append(area);
  area.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  area.remove();
  return ok;
}

async function copyText(text, status) {
  let ok = false;
  try {
    await navigator.clipboard.writeText(text);
    ok = true;
  } catch {
    ok = copyFallback(text);
  }
  status.textContent = ok
    ? "Copied. Paste it into an e-mail or a message to the FastWarp team."
    : "The browser did not allow copying; open Show report and copy the text by hand.";
}

function deviceReport() {
  return makeReport("DeviceReport", "No problem; this report describes the device only.");
}

function onDiag(kind, detail) {
  const at_s = seconds(performance.now());
  if (kind === "device") diag.device = detail;
  else if (kind === "adapter") diag.workerAdapter = detail;
  else if (kind === "console") {
    diag.console.push({ at_s, ...detail });
    if (diag.console.length > 20) diag.console.shift();
  } else {
    diag.events.push({ at_s, ...detail });
    if (diag.events.length > 40) {
      diag.events.shift();
      diag.droppedEvents += 1;
    }
  }
  refreshReport();
}

function updateButtons() {
  const ok = state.ready && state.inputs.fixed && state.inputs.moving;
  const run = el("fw-run");
  run.disabled = !ok || state.running;
  run.hidden = state.running;
  run.textContent = state.done ? "Run again" : "Run";
  run.classList.toggle("fw-secondary", state.done && !run.disabled);
  el("fw-cancel").hidden = !state.running;
  el("fw-cancel").disabled = !state.running;
  let hint = "";
  if (!state.running) {
    if (state.gpu === "starting") hint = "Starting GPU...";
    else if (state.gpu === "failed") hint = "WebGPU unavailable";
    else if (!state.inputs.fixed || !state.inputs.moving) hint = "Choose two images";
  }
  el("fw-hint").textContent = hint;
  el("fw-sample").disabled = state.running;
  el("fw-sample").hidden = state.sample;
  el("fw-sample-state").hidden = !state.sample;
  el("fw-fixed").disabled = state.running;
  el("fw-moving").disabled = state.running;
  el("fw-grid").disabled = state.running;
  for (const key of ["fixed", "moving"]) {
    const name = el(`fw-${key}-name`);
    const full = state.inputs[key]?.label ?? "";
    name.textContent = full || "No file chosen";
    name.title = full;
  }
}

function adapterLabel(info) {
  const js = state.jsAdapter ?? {};
  return info.name || [js.vendor, js.architecture, js.device].filter(Boolean).join(" ") || "unnamed adapter";
}

function showAdapter(info) {
  const js = state.jsAdapter ?? {};
  const software = info.software || js.isFallbackAdapter;
  const rows = [
    ["GPU", adapterLabel(info)],
    ["Driver", [info.driver, info.driver_info, js.description].filter(Boolean).join(", ") || "not reported"],
    ["Largest buffer", bytes(info.limits?.max_buffer_size)],
    ["Largest storage binding", bytes(info.limits?.max_storage_buffer_binding_size)],
    ["FastWarp", info.fastwarp_version],
  ];
  const dl = el("fw-adapter");
  dl.replaceChildren();
  for (const [k, v] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    dl.append(dt, dd);
  }
  const gpuName = document.createElement("code");
  gpuName.textContent = adapterLabel(info);
  el("fw-adapter-summary").replaceChildren("WebGPU · ", gpuName, ` · FastWarp ${info.fastwarp_version}`);
  el("fw-support").textContent = "";
  el("fw-adapter-box").hidden = false;
  if (software && diag.report?.kind !== "UnfinishedRun") {
    message(
      "The browser gave FastWarp a software (CPU) adapter instead of a GPU. A registration will be very slow; use the 2 mm grid, or a browser and computer with WebGPU on the GPU.",
      "warning",
    );
  }
}

function describe(p) {
  const [name] = PHASES[p.phase] ?? [p.phase];
  if (!p.levels) return name;
  let text = `${name}, level ${p.level} of ${p.levels}`;
  if (p.grid) text += ` (${p.grid.join(" x ")} voxels)`;
  if (p.iterations) text += `, iteration ${Math.min(p.iteration, p.iterations)} of at most ${p.iterations}`;
  return text;
}

function fraction(p) {
  const [, start, end] = PHASES[p.phase] ?? ["", 0, 0];
  if (!p.levels) return start;
  const within = p.iterations ? Math.min(1, p.iteration / p.iterations) : 0;
  const weights = Array.from({ length: p.levels }, (_, i) => 2 ** i);
  const total = weights.reduce((a, b) => a + b, 0);
  const done = weights.slice(0, p.level - 1).reduce((a, b) => a + b, 0) + within * weights[p.level - 1];
  return start + (end - start) * (done / total);
}

function crumb() {
  const run = diag.run;
  if (!run) return;
  state.savedAt = performance.now();
  saveRun({
    started_utc: run.startedUtc,
    grid: run.grid,
    fixed: run.fixed,
    moving: run.moving,
    registration_grid: run.registrationGrid,
    stage: run.stage,
    elapsed_s: seconds(performance.now() - run.started),
    wasm_bytes: diag.memory,
    adapter: state.adapterLabel,
  });
}

function onProgress(p, memory) {
  const now = performance.now();
  state.progressAt = now;
  if (memory != null) diag.memory = memory;
  if (state.stalled) {
    state.stalled = false;
    message("");
    hideReport();
  }
  el("fw-stage").textContent = describe(p);
  el("fw-progress").value = fraction(p);
  const key = `${p.phase}/${p.level}`;
  const run = diag.run;
  if (run) {
    run.stage = { phase: p.phase, level: p.level ?? null, levels: p.levels ?? null, iteration: p.iteration ?? null, iterations: p.iterations ?? null, grid: p.grid ?? null };
    run.progressAt = now;
  }
  if (key !== state.lastKey) {
    state.lastKey = key;
    const [name] = PHASES[p.phase] ?? [p.phase];
    log(p.levels ? `${name}, level ${p.level} of ${p.levels}${p.grid ? ` (${p.grid.join(" x ")})` : ""}` : name);
    if (run && run.timeline.length < 60) run.timeline.push([seconds(now - run.started), p.phase, p.levels ? p.level : null]);
    crumb();
  } else if (now - state.savedAt > 10000) {
    crumb();
  }
}

function watchStall() {
  if (!state.running || !state.posted || state.stalled) return;
  const quiet = performance.now() - state.progressAt;
  if (quiet < STALL_MS) return;
  state.stalled = true;
  const stage = diag.run?.stage ? describe(diag.run.stage) : "the start";
  const minutes = Math.max(1, Math.round(quiet / 60000));
  log(`No progress for ${minutes} minutes.`);
  message(
    `No progress for ${minutes} minutes (${stage}). The GPU may have stopped responding. You can wait, or cancel and try the 2 mm grid; the report below helps us find out why.`,
    "warning",
  );
  showReport("Stall", `no progress for ${minutes} minutes during ${stageText(diag.run?.stage)}`, { tone: "warning" });
}

function zipEntries(buffer) {
  const v = new DataView(buffer);
  let end = buffer.byteLength - 22;
  while (end >= 0 && v.getUint32(end, true) !== 0x06054b50) end--;
  if (end < 0) return [];
  const count = v.getUint16(end + 10, true);
  let p = v.getUint32(end + 16, true);
  const text = new TextDecoder();
  const out = [];
  for (let i = 0; i < count; i++) {
    const size = v.getUint32(p + 24, true);
    const nameLength = v.getUint16(p + 28, true);
    const skip = nameLength + v.getUint16(p + 30, true) + v.getUint16(p + 32, true);
    const local = v.getUint32(p + 42, true);
    const name = text.decode(new Uint8Array(buffer, p + 46, nameLength));
    const start = local + 30 + v.getUint16(local + 26, true) + v.getUint16(local + 28, true);
    out.push({ name, size, start });
    p += 46 + skip;
  }
  return out;
}

function finish(outcome) {
  const wasRunning = state.running;
  state.running = false;
  state.stalled = false;
  clearInterval(state.timer);
  clearInterval(state.stallTimer);
  clearTimeout(state.cancelTimer);
  if (wasRunning && diag.run && !diag.run.end) {
    diag.run.end = performance.now();
    diag.run.outcome = outcome;
    endRun(outcome === "ok");
  }
  updateButtons();
}

function onDone(zip, memory, view) {
  const elapsed = (performance.now() - state.started) / 1000;
  if (memory != null) diag.memory = memory;
  finish("ok");
  diag.previous = null;
  hideReport();
  state.restarts = 0;
  el("fw-progress").value = 1;
  const entries = zipEntries(zip);
  el("fw-stage").textContent = `Done in ${clock(elapsed * 1000)} · ${entries.length} files, ${bytes(zip.byteLength)}`;
  el("fw-elapsed").textContent = "";
  log(`Files: ${entries.map((e) => `${e.name} (${bytes(e.size)})`).join(", ")}`);
  state.done = true;
  el("fw-progress").hidden = true;
  updateButtons();
  let run = null;
  const runEntry = entries.find((e) => e.name === "run.json");
  if (runEntry) {
    try {
      run = JSON.parse(new TextDecoder().decode(new Uint8Array(zip, runEntry.start, runEntry.size)));
    } catch {
      run = null;
    }
  }
  if (run) {
    const t = run.timings_s;
    log(
      `Registration grid ${run.registration_grid.size.join(" x ")}; linear ${t.linear.toFixed(1)} s, SyN ${t.syn.toFixed(1)} s, total ${t.total.toFixed(1)} s; GPU memory peak ${bytes(run.peak_gpu_bytes)}`,
    );
    if (diag.run) diag.run.registrationGrid = run.registration_grid.size;
  }
  if (memory) log(`WebAssembly memory of the worker: ${bytes(memory)}`);
  if (state.zipUrl) URL.revokeObjectURL(state.zipUrl);
  state.zipUrl = URL.createObjectURL(new Blob([zip], { type: "application/zip" }));
  const link = el("fw-download");
  link.href = state.zipUrl;
  link.download = "fastwarp-registration.zip";
  link.textContent = `Download zip (${bytes(zip.byteLength)})`;
  link.hidden = false;
  message("");
  log(`Done: ${bytes(zip.byteLength)} zip with ${entries.length} files.`);
  if (view) viewer.show(view);
  result("ok", {
    zip_bytes: zip.byteLength,
    elapsed_s: elapsed,
    files: entries.map((e) => [e.name, e.size]),
    timings_s: run?.timings_s ?? null,
    registration_grid: run?.registration_grid?.size ?? null,
    peak_gpu_bytes: run?.peak_gpu_bytes ?? null,
    wasm_memory_bytes: memory ?? null,
    adapter: state.adapterLabel ?? null,
    affine_parameters: run?.results?.affine_parameters ?? null,
    viewer: view ? { dims: view.dims, spacing: view.spacing, before: view.before } : null,
  });
}

function onError(name, text, phase, extra = {}) {
  const wasRunning = state.running;
  if (extra.memory != null) diag.memory = extra.memory;
  if (name !== "Cancelled") diag.error = { name, message: clip(text, 2000), phase, stack: extra.stack ?? null };
  finish(name === "Cancelled" ? "cancelled" : "error");
  if (name === "Cancelled") {
    el("fw-stage").textContent = "Cancelled";
    message("The registration was cancelled.", "info");
    hideReport();
    log("Cancelled.");
    result("cancelled");
    return;
  }
  const crashed = name === "RuntimeError" || name === "WorkerError";
  const panic = lastPanic();
  const shown = crashed
    ? `FastWarp stopped with an internal error (${clip(panic ?? text, 300)}). The page has restarted its worker. If it happens again, try the 2 mm grid.`
    : text;
  message(shown, "error");
  log(`Error (${name}): ${text}`);
  if (panic) log(`Internal error: ${clip(panic, 600)}`);
  if (wasRunning) el("fw-stage").textContent = "Failed";
  showReport(name, text);
  result("error", { name, message: text, phase });
  if (!crashed) return;
  if (state.restarts < 2) {
    state.restarts += 1;
    startWorker(false);
  } else {
    state.ready = false;
    updateButtons();
    message(`${shown} The worker failed again; reload the page to try once more.`, "error");
  }
}

function startFailed(name, text, extra = {}) {
  state.gpu = "failed";
  updateButtons();
  if (extra.memory != null) diag.memory = extra.memory;
  diag.error = { name, message: clip(text, 2000), phase: "start", stack: extra.stack ?? null };
  const panic = lastPanic();
  const what = name === "RuntimeError" && panic ? `FastWarp stopped with an internal error (${clip(panic, 300)}).` : text;
  const hint = /adapter/i.test(text) ? supportHint(diag.platform, "no-adapter") : "";
  el("fw-support").textContent = "WebGPU could not be started.";
  message(`${what} ${hint} The report below helps us find out which devices need a fix.`.replace(/\s+/g, " "), "error");
  showReport(name, text);
  result("error", { name, message: text, phase: "gpu_info" });
}

function startWorker(requestInfo = true) {
  state.worker?.terminate();
  dropSliceRequests();
  viewer.hide();
  const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }) => {
    if (worker !== state.worker) return;
    if (data.type === "info") {
      state.ready = true;
      diag.version = data.info.fastwarp_version ?? diag.version;
      if (data.memory != null) diag.memory = data.memory;
      showAdapter(data.info);
      state.gpu = "ready";
      state.adapterLabel = adapterLabel(data.info);
      updateButtons();
      if (autorun === "sample") runSample();
    } else if (data.type === "diag") {
      onDiag(data.kind, data.detail ?? {});
    } else if (data.type === "progress") {
      onProgress(data.progress, data.memory);
    } else if (data.type === "done") {
      onDone(data.zip, data.memory, data.view);
    } else if (data.type === "slice") {
      const pending = sliceRequests.get(data.id);
      sliceRequests.delete(data.id);
      pending?.resolve(new Uint8Array(data.data));
    } else if (data.type === "slice-error") {
      const pending = sliceRequests.get(data.id);
      sliceRequests.delete(data.id);
      pending?.reject(new Error(data.message));
    } else if (data.type === "error") {
      if (!state.ready && !state.running) {
        startFailed(data.name, data.message, data);
        return;
      }
      onError(data.name, data.message, state.lastKey, data);
    }
  };
  worker.onerror = (e) => {
    e.preventDefault?.();
    const text = e.message || "the worker could not be started";
    if (!state.ready && !state.running) {
      startFailed("WorkerError", text);
      el("fw-support").textContent = "FastWarp's worker could not be started.";
      return;
    }
    onError("WorkerError", text, state.lastKey);
  };
  state.worker = worker;
  if (requestInfo) {
    el("fw-support").textContent = "Starting the GPU and compiling FastWarp's kernels...";
    worker.postMessage({ type: "info" });
  } else {
    state.ready = true;
  }
}

async function geometry(buffer) {
  try {
    return await niftiGeometry(buffer);
  } catch {
    return null;
  }
}

async function run() {
  if (state.running) return;
  const { fixed, moving } = state.inputs;
  if (!fixed || !moving) return;
  el("fw-download").hidden = true;
  viewer.hide();
  el("fw-progress").hidden = false;
  state.done = false;
  el("fw-log").textContent = "";
  message("");
  hideReport();
  const grid = el("fw-grid").value;
  const token = ++state.runId;
  state.posted = false;
  state.running = true;
  state.started = performance.now();
  state.progressAt = state.started;
  state.lastKey = "";
  diag.error = null;
  diag.run = {
    grid,
    fixed: null,
    moving: null,
    registrationGrid: null,
    startedUtc: `${new Date().toISOString().slice(0, 16)}Z`,
    started: state.started,
    progressAt: state.started,
    stage: null,
    timeline: [],
  };
  updateButtons();
  state.timer = setInterval(() => {
    el("fw-elapsed").textContent = clock(performance.now() - state.started);
  }, 250);
  el("fw-progress").value = 0;
  el("fw-stage").textContent = "Reading the files";
  log(`Fixed ${fixed.label}, moving ${moving.label}, ${grid} grid`);
  let buffers;
  try {
    buffers = await Promise.all([fixed.load(), moving.load()]);
  } catch (err) {
    if (token === state.runId && state.running) {
      onError("InvalidInput", `The images could not be read: ${err.message ?? err}`, "load");
    }
    return;
  }
  if (token !== state.runId || !state.running) return;
  const [fixedGeometry, movingGeometry] = await Promise.all(buffers.map(geometry));
  if (token !== state.runId || !state.running) return;
  diag.run.fixed = fixedGeometry;
  diag.run.moving = movingGeometry;
  diag.run.registrationGrid = isoDims(fixedGeometry, grid === "2mm" ? 2 : 1);
  state.posted = true;
  state.progressAt = performance.now();
  diag.run.progressAt = state.progressAt;
  state.stallTimer = setInterval(watchStall, 5000);
  crumb();
  state.worker.postMessage(
    { type: "run", fixed: buffers[0], moving: buffers[1], options: { grid } },
    buffers,
  );
}

function cancelRun() {
  if (!state.running) return;
  log("Cancelling...");
  el("fw-cancel").disabled = true;
  if (!state.posted) {
    state.runId += 1;
    onError("Cancelled", "cancelled", "load");
    return;
  }
  state.worker.postMessage({ type: "cancel" });
  state.cancelTimer = setTimeout(() => {
    if (!state.running) return;
    log("The worker did not stop in time; stopping it.");
    startWorker(false);
    onError("Cancelled", "cancelled", state.lastKey);
  }, 15000);
}

function fromUrl(path, label) {
  return { label, load: async () => {
    const r = await fetch(new URL(path, here));
    if (!r.ok) throw new Error(`${label}: HTTP ${r.status}`);
    return r.arrayBuffer();
  } };
}

function fromFile(file) {
  return { label: file.name, load: () => file.arrayBuffer() };
}

function runSample() {
  state.sample = true;
  state.inputs.fixed = fromUrl(SAMPLES.fixed, "MNI 2009b template");
  state.inputs.moving = fromUrl(SAMPLES.moving, "Defaced T1w");
  updateButtons();
  if (autorun === "sample") {
    const grid = params.get("grid");
    if (grid === "1mm" || grid === "2mm") el("fw-grid").value = grid;
    run();
  }
}

function showUnfinished(previous) {
  const [name] = PHASES[previous.stage?.phase] ?? [previous.stage?.phase ?? "the start"];
  const after = previous.elapsed_s != null ? ` after ${clock(previous.elapsed_s * 1000)}` : "";
  const advice = previous.grid === "1mm" ? "Try the 2 mm grid." : "Try smaller images, or a computer with more memory.";
  message(
    `The last run on this page did not finish: it stopped during "${name}"${after}. If you did not close or reload the page yourself, the browser most likely closed it because it ran out of memory. ${advice} The report below helps us find out which devices need a fix.`,
    "warning",
  );
  showReport("UnfinishedRun", `the page closed during ${stageText(previous.stage)}`, { tone: "warning", dismiss: true });
}

function setupReports() {
  el("fw-copy").addEventListener("click", () => {
    if (diag.report) copyText(reportText(makeReport(diag.report.kind, diag.report.text)), el("fw-copy-state"));
  });
  el("fw-mail").addEventListener("click", () => {
    if (diag.report) el("fw-mail").href = mailtoHref(makeReport(diag.report.kind, diag.report.text)).href;
  });
  el("fw-report-box").addEventListener("toggle", refreshReport);
  el("fw-dismiss").addEventListener("click", () => {
    dismissUnfinished();
    diag.previous = null;
    hideReport();
    message("");
  });
  el("fw-device-copy").addEventListener("click", () => copyText(reportText(deviceReport()), el("fw-device-state")));
  el("fw-device-mail").addEventListener("click", () => {
    el("fw-device-mail").href = mailtoHref(deviceReport()).href;
  });
  navigator.userAgentData
    ?.getHighEntropyValues?.(["model", "platformVersion", "architecture"])
    .then((values) => {
      diag.hints = values;
    })
    .catch(() => {});
}

function adapterDetails(adapter) {
  const i = adapter.info ?? {};
  return {
    info: {
      vendor: i.vendor ?? "",
      architecture: i.architecture ?? "",
      device: i.device ?? "",
      description: i.description ?? "",
      fallback: Boolean(i.isFallbackAdapter ?? adapter.isFallbackAdapter ?? false),
    },
    features: featureList(adapter.features),
    limits: limitsObject(adapter.limits),
    wgsl: featureList(navigator.gpu.wgslLanguageFeatures ?? []),
  };
}

async function main() {
  el("fw-grid").value = isMobile() ? "2mm" : "1mm";
  for (const key of ["fixed", "moving"]) {
    const input = el(`fw-${key}`);
    const zone = el(`fw-pick-${key}`);
    const choose = (file) => {
      if (!file || state.running) return;
      state.inputs[key] = fromFile(file);
      state.sample = false;
      updateButtons();
    };
    input.addEventListener("change", (e) => choose(e.target.files[0]));
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("fw-drag");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("fw-drag"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("fw-drag");
      choose(e.dataTransfer.files[0]);
    });
  }
  el("fw-sample").addEventListener("click", runSample);
  el("fw-form").addEventListener("submit", (e) => {
    e.preventDefault();
    run();
  });
  el("fw-cancel").addEventListener("click", cancelRun);
  setupReports();
  updateButtons();

  diag.previous = takeUnfinished();
  if (diag.previous) showUnfinished(diag.previous);

  if (!navigator.gpu) {
    state.gpu = "failed";
    updateButtons();
    el("fw-support").textContent = "This browser does not offer WebGPU.";
    const hint = window.isSecureContext
      ? supportHint(diag.platform, "no-gpu")
      : "WebGPU needs a secure page (https or localhost).";
    message(hint || BROWSERS, "error");
    showReport("NoWebGPU", "navigator.gpu is missing");
    result("error", { name: "NoWebGPU", message: "navigator.gpu is missing" });
    return;
  }
  let adapter = null;
  let adapterError = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  } catch (err) {
    adapter = null;
    adapterError = `${err?.name}: ${err?.message ?? err}`;
  }
  if (!adapter) {
    state.gpu = "failed";
    updateButtons();
    el("fw-support").textContent = "WebGPU is present, but the browser gave no GPU adapter.";
    message(supportHint(diag.platform, "no-adapter") || BROWSERS, "error");
    showReport("NoAdapter", adapterError ?? "requestAdapter returned null");
    result("error", { name: "NoAdapter", message: "requestAdapter returned null" });
    return;
  }
  state.jsAdapter = adapter.info ?? null;
  diag.adapter = adapterDetails(adapter);
  const [haveWasm, haveFixed, haveMoving] = await Promise.all([
    exists("pkg/fastwarp_web_bg.wasm"),
    exists(SAMPLES.fixed),
    exists(SAMPLES.moving),
  ]);
  el("fw-sample-row").hidden = !(haveFixed && haveMoving);
  if (!haveWasm) {
    state.gpu = "failed";
    updateButtons();
    el("fw-support").textContent = "The FastWarp WebAssembly module is not part of this build of the documentation.";
    message(
      "It is built on Read the Docs, or locally with docs/_scripts/build_web_demo.sh before sphinx-build.",
      "error",
    );
    result("error", { name: "NoModule", message: "pkg/fastwarp_web_bg.wasm is missing" });
    return;
  }
  startWorker(true);
}

main();
