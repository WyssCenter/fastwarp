// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

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
  lastKey: "",
  zipUrl: null,
  jsAdapter: null,
  adapterLabel: null,
  runId: 0,
  posted: false,
  restarts: 0,
};

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

function updateButtons() {
  const ok = state.ready && state.inputs.fixed && state.inputs.moving;
  el("fw-run").disabled = !ok || state.running;
  el("fw-cancel").disabled = !state.running;
  el("fw-sample").disabled = state.running;
  el("fw-fixed").disabled = state.running;
  el("fw-moving").disabled = state.running;
  el("fw-grid").disabled = state.running;
  const names = ["fixed", "moving"].map((k) => state.inputs[k]?.label ?? "none");
  el("fw-selected").textContent = `Fixed: ${names[0]}. Moving: ${names[1]}.`;
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
  dl.hidden = false;
  if (software) {
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

function onProgress(p) {
  el("fw-stage").textContent = describe(p);
  el("fw-progress").value = fraction(p);
  const key = `${p.phase}/${p.level}`;
  if (key !== state.lastKey) {
    state.lastKey = key;
    const [name] = PHASES[p.phase] ?? [p.phase];
    log(p.levels ? `${name}, level ${p.level} of ${p.levels}${p.grid ? ` (${p.grid.join(" x ")})` : ""}` : name);
  }
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

function finish() {
  state.running = false;
  clearInterval(state.timer);
  clearTimeout(state.cancelTimer);
  updateButtons();
}

function onDone(zip, memory) {
  const elapsed = (performance.now() - state.started) / 1000;
  finish();
  state.restarts = 0;
  el("fw-progress").value = 1;
  el("fw-stage").textContent = `Done in ${clock(elapsed * 1000)}`;
  el("fw-elapsed").textContent = "";
  const entries = zipEntries(zip);
  const list = el("fw-files");
  list.replaceChildren();
  for (const e of entries) {
    const li = document.createElement("li");
    li.textContent = `${e.name} (${bytes(e.size)})`;
    list.append(li);
  }
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
  });
}

function onError(name, text, phase) {
  const wasRunning = state.running;
  finish();
  if (name === "Cancelled") {
    el("fw-stage").textContent = "Cancelled";
    message("The registration was cancelled.", "info");
    log("Cancelled.");
    result("cancelled");
    return;
  }
  const crashed = name === "RuntimeError" || name === "WorkerError";
  const shown = crashed
    ? `FastWarp stopped with an internal error (${text}). The page has restarted its worker; the browser's console may say more. If it happens again, try the 2 mm grid.`
    : text;
  message(shown, "error");
  log(`Error (${name}): ${text}`);
  if (wasRunning) el("fw-stage").textContent = "Failed";
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

function startWorker(requestInfo = true) {
  state.worker?.terminate();
  const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.type === "info") {
      state.ready = true;
      showAdapter(data.info);
      el("fw-support").textContent = "WebGPU is available. Choose two images, or load the sample data.";
      state.adapterLabel = adapterLabel(data.info);
      log(`GPU: ${state.adapterLabel}`);
      updateButtons();
      if (autorun === "sample") runSample();
    } else if (data.type === "progress") {
      onProgress(data.progress);
    } else if (data.type === "done") {
      onDone(data.zip, data.memory);
    } else if (data.type === "error") {
      if (!state.ready && !state.running) {
        el("fw-support").textContent = "WebGPU could not be started.";
        message(`${data.message} ${BROWSERS}`, "error");
        result("error", { name: data.name, message: data.message, phase: "gpu_info" });
        return;
      }
      onError(data.name, data.message, state.lastKey);
    }
  };
  worker.onerror = (e) => {
    e.preventDefault?.();
    const text = e.message || "the worker could not be started";
    if (!state.ready && !state.running) {
      el("fw-support").textContent = "FastWarp's worker could not be started.";
      message(text, "error");
      result("error", { name: "WorkerError", message: text, phase: "start" });
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

async function run() {
  if (state.running) return;
  const { fixed, moving } = state.inputs;
  if (!fixed || !moving) return;
  el("fw-download").hidden = true;
  el("fw-files").replaceChildren();
  el("fw-log").textContent = "";
  message("");
  const grid = el("fw-grid").value;
  const token = ++state.runId;
  state.posted = false;
  state.running = true;
  state.started = performance.now();
  state.lastKey = "";
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
  state.posted = true;
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
  state.inputs.fixed = fromUrl(SAMPLES.fixed, "MNI ICBM 2009b template (sample)");
  state.inputs.moving = fromUrl(SAMPLES.moving, "defaced T1w (sample)");
  updateButtons();
  if (autorun === "sample") {
    const grid = params.get("grid");
    if (grid === "1mm" || grid === "2mm") el("fw-grid").value = grid;
    run();
  }
}

async function main() {
  el("fw-grid").value = isMobile() ? "2mm" : "1mm";
  el("fw-fixed").addEventListener("change", (e) => {
    if (e.target.files[0]) state.inputs.fixed = fromFile(e.target.files[0]);
    updateButtons();
  });
  el("fw-moving").addEventListener("change", (e) => {
    if (e.target.files[0]) state.inputs.moving = fromFile(e.target.files[0]);
    updateButtons();
  });
  el("fw-sample").addEventListener("click", runSample);
  el("fw-form").addEventListener("submit", (e) => {
    e.preventDefault();
    run();
  });
  el("fw-cancel").addEventListener("click", cancelRun);
  updateButtons();

  if (!("gpu" in navigator)) {
    el("fw-support").textContent = "This browser does not offer WebGPU.";
    message(window.isSecureContext ? BROWSERS : `WebGPU needs a secure page (https or localhost). ${BROWSERS}`, "error");
    result("error", { name: "NoWebGPU", message: "navigator.gpu is missing" });
    return;
  }
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  } catch {
    adapter = null;
  }
  if (!adapter) {
    el("fw-support").textContent = "WebGPU is present, but the browser gave no GPU adapter.";
    message(
      `The GPU may be on the browser's block list, or WebGPU may be switched off (see chrome://gpu or about:support). ${BROWSERS}`,
      "error",
    );
    result("error", { name: "NoAdapter", message: "requestAdapter returned null" });
    return;
  }
  state.jsAdapter = adapter.info ?? null;
  const [haveWasm, haveFixed, haveMoving] = await Promise.all([
    exists("pkg/fastwarp_web_bg.wasm"),
    exists(SAMPLES.fixed),
    exists(SAMPLES.moving),
  ]);
  el("fw-sample-row").hidden = !(haveFixed && haveMoving);
  if (!haveWasm) {
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
