// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { extname, join, resolve, sep } from "node:path";

const USAGE = `usage: node ci/web-faults.mjs <built html directory> [--chrome PATH] [--swiftshader]
       [--suite start|faults|all] [--only NAME,NAME] [--timeout SECONDS]

Serves the built documentation on 127.0.0.1 and opens browser.html in headless Chrome. The suite
"start" starts the GPU, which compiles every kernel with the browser's own shader compiler and runs
the self-test, and then registers the sample on the 2 mm grid; it fails on any WebGPU error. The
suite "faults" runs the demo once per fault of its fault injection (?fault=..., ?limits=...) and
checks that the page shows the error panel and a report with the expected contents. Exit code 0
when every case passes.`;

const SAMPLE_NAMES = ["mni152nlin2009basym", "sample-t1w-defaced", ".nii"];
const MAILTO_LIMIT = 1800;

function options(argv) {
  const out = { root: null, chrome: process.env.CHROME_PATH ?? null, swiftshader: false, only: null, suite: "all", timeout: 600 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--chrome") out.chrome = argv[++i];
    else if (a === "--swiftshader") out.swiftshader = true;
    else if (a === "--only") out.only = new Set(argv[++i].split(","));
    else if (a === "--suite") out.suite = argv[++i];
    else if (a === "--timeout") out.timeout = Number(argv[++i]);
    else if (a === "--help" || a === "-h") {
      console.log(USAGE);
      process.exit(0);
    } else if (!out.root) out.root = resolve(a);
    else throw new Error(`unknown argument ${a}\n${USAGE}`);
  }
  if (!out.root || !out.chrome) throw new Error(USAGE);
  if (!["all", "start", "faults"].includes(out.suite)) throw new Error(`unknown suite ${out.suite}\n${USAGE}`);
  return out;
}

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".gz": "application/gzip",
  ".txt": "text/plain; charset=utf-8",
};

function serve(root) {
  const server = createServer((req, res) => {
    let file;
    try {
      const path = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
      file = resolve(join(root, path));
      if (file !== root && !file.startsWith(root + sep)) throw new Error("outside");
      if (statSync(file).isDirectory()) file = join(file, "index.html");
      const body = readFileSync(file);
      res.writeHead(200, { "content-type": TYPES[extname(file)] ?? "application/octet-stream", "cache-control": "no-store" });
      res.end(body);
    } catch {
      res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
      res.end("not found");
    }
  });
  return new Promise((ok) => server.listen(0, "127.0.0.1", () => ok(server)));
}

class Cdp {
  constructor(input, output) {
    this.input = input;
    this.next = 0;
    this.pending = new Map();
    this.waiters = [];
    this.console = [];
    let buffer = "";
    output.setEncoding("utf8");
    output.on("data", (chunk) => {
      buffer += chunk;
      let end;
      while ((end = buffer.indexOf("\0")) >= 0) {
        const message = JSON.parse(buffer.slice(0, end));
        buffer = buffer.slice(end + 1);
        this.receive(message);
      }
    });
  }

  receive(message) {
    if (message.id != null) {
      const p = this.pending.get(message.id);
      this.pending.delete(message.id);
      if (message.error) p?.reject(new Error(`${message.error.message} (${message.error.code})`));
      else p?.resolve(message.result);
      return;
    }
    if (message.method === "Runtime.consoleAPICalled") {
      const text = message.params.args.map((a) => a.value ?? a.description ?? "").join(" ");
      this.console.push(text);
      if (this.console.length > 200) this.console.shift();
    }
    if (message.method === "Runtime.exceptionThrown") {
      this.console.push(`exception: ${message.params.exceptionDetails?.exception?.description ?? message.params.exceptionDetails?.text}`);
    }
  }

  send(method, params = {}, sessionId = undefined) {
    const id = ++this.next;
    this.input.write(JSON.stringify({ id, method, params, sessionId }) + "\0");
    return new Promise((resolveCall, rejectCall) => this.pending.set(id, { resolve: resolveCall, reject: rejectCall }));
  }
}

const STATE = `(() => {
  const text = (id) => document.getElementById(id)?.textContent ?? "";
  const panel = document.getElementById("fw-report");
  const state = { title: document.title, message: text("fw-message"), visible: Boolean(panel && !panel.hidden) };
  if (!state.visible) return state;
  const box = document.getElementById("fw-report-box");
  if (!box.open) box.open = true;
  const report = text("fw-report-text");
  const start = report.indexOf('\\n{\\n  "schema"');
  try {
    state.json = start >= 0 ? JSON.parse(report.slice(start + 1)) : null;
  } catch {
    state.json = null;
  }
  state.text = report;
  const mail = document.getElementById("fw-mail");
  state.mail = mail && !mail.hidden ? mail.href : null;
  return state;
})()`;

const RESULT = `(() => {
  const box = document.getElementById("fw-result");
  let json = null;
  try {
    json = JSON.parse(box?.textContent || "null");
  } catch {
    json = null;
  }
  return { title: document.title, message: document.getElementById("fw-message")?.textContent ?? "", status: box?.dataset.status ?? null, json };
})()`;

const ADAPTER = `(async () => {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return "none";
  const i = adapter.info ?? {};
  const name = [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(" ") || "unnamed";
  const fallback = i.isFallbackAdapter ?? adapter.isFallbackAdapter ?? false;
  return \`\${name}, maxBufferSize \${adapter.limits.maxBufferSize}, fallback \${fallback}\`;
})()`;

const WORKER_PROBE = `new Promise((resolve) => {
  const worker = new Worker("/_static/web/worker.js?limits=64&fault=panic&fault_at=gpu", { type: "module" });
  const done = (value) => {
    worker.terminate();
    resolve(value);
  };
  worker.onmessage = ({ data }) => {
    if (data.type === "info") done(data.info.limits_adapter?.max_storage_buffer_binding_size ?? null);
    else if (data.type === "error") done("error: " + data.message);
  };
  worker.onerror = (e) => done("worker error: " + e.message);
  worker.postMessage({ type: "info" });
  setTimeout(() => done("timeout"), 120000);
})`;

const crashed = ["RuntimeError", "WorkerError"];
const GPU_ERRORS = ["GPUValidationError", "GPUInternalError", "GPUOutOfMemoryError"];

function gpuClean(r) {
  const events = r.gpu_events ?? [];
  const bad = events.filter((e) => GPU_ERRORS.includes(e.type) || e.kind === "uncaptured" || e.kind === "adapter-error");
  const internal = (r.worker_errors ?? []).filter((e) => /internal error/.test(e.text ?? ""));
  return [
    [bad.length === 0, `no WebGPU errors (got ${JSON.stringify(bad).slice(0, 800)})`],
    [internal.length === 0, `no internal errors (got ${JSON.stringify(internal).slice(0, 800)})`],
  ];
}

const CASES = [
  {
    name: "start",
    suite: "start",
    query: "autorun=info",
    result: "ready",
    checks: (r) => [[(r.selftest?.attempts ?? 0) >= 1, "the self-test ran"], ...gpuClean(r)],
  },
  {
    name: "run",
    suite: "start",
    query: "autorun=sample&grid=2mm&plan=short",
    result: "ok",
    checks: (r) => [
      [r.files?.length === 6 && r.zip_bytes > 0, "the zip holds the six output files"],
      [JSON.stringify(r.registration_grid) === "[99,117,95]", `the registration grid is 99x117x95 (got ${JSON.stringify(r.registration_grid)})`],
      [Array.isArray(r.affine_parameters) && r.affine_parameters.every(Number.isFinite), "the affine parameters are finite"],
      ...gpuClean(r),
    ],
  },
  {
    name: "panic",
    query: "autorun=sample&grid=2mm&plan=short&fault=panic&fault_at=rigid",
    kinds: crashed,
    checks: (j, s) => [
      [/fault injection: a panic/.test(j.errors.panic ?? ""), "the report holds the panic message"],
      [/crates\/fastwarp-web\/src\/web\.rs:\d+:\d+/.test(j.errors.panic ?? ""), "the report holds the panic location"],
      [j.run?.stage?.phase === "rigid", "the report holds the stage"],
      [/internal error/.test(s.message), "the panel says it was an internal error"],
    ],
  },
  {
    name: "alloc",
    query: "autorun=sample&grid=2mm&plan=short&fault=alloc&fault_at=rigid",
    kinds: crashed,
    checks: (j) => [[(j.errors.failed_alloc_bytes ?? 0) >= 2 ** 30, "the report holds the size of the failed allocation"]],
  },
  {
    name: "lost",
    query: "autorun=sample&grid=2mm&plan=short&fault=lost&fault_at=rigid",
    kinds: ["GpuDeviceLost"],
    checks: (j, s) => [
      [j.errors.gpu.some((e) => e.kind === "lost" && e.reason === "destroyed"), "the report holds the lost reason"],
      [/reset the GPU/.test(s.message) && /Run again/.test(s.message), "the panel says the GPU was reset and to run again"],
      [!/memory/i.test(s.message), "the panel does not blame memory"],
      [/device lost/i.test(s.message), "the panel keeps the browser's text"],
      [j.errors.engine?.gpu?.out_of_memory_errors === 0, "the engine caught no out-of-memory error"],
      [j.errors.engine?.gpu?.memory_evidence === null, "the engine found no memory evidence"],
    ],
  },
  {
    name: "lostoom",
    query: "autorun=sample&grid=2mm&plan=short&fault=lostoom&fault_at=rigid",
    kinds: ["GpuMemory"],
    checks: (j, s) => [
      [/memory/i.test(s.message), "the panel says GPU memory ran short"],
      [j.errors.gpu.some((e) => e.type === "GPUOutOfMemoryError"), "the report holds the out-of-memory error"],
      [(j.errors.engine?.gpu?.out_of_memory_errors ?? 0) >= 1, "the engine counted the out-of-memory error"],
      [j.errors.engine?.gpu?.memory_evidence === "oom_error", "the engine names the out-of-memory error as its evidence"],
    ],
  },
  {
    name: "oom",
    query: "autorun=sample&grid=2mm&plan=short&fault=oom&fault_at=rigid",
    kinds: ["GpuMemory"],
    checks: (j, s) => [
      [/Not enough GPU memory/.test(s.message), "the panel says GPU memory ran out"],
      [j.errors.gpu.some((e) => e.type === "GPUOutOfMemoryError"), "the report holds the error-scope result"],
      [j.errors.engine?.stage?.phase === "rigid", "the engine's diagnostics hold the stage"],
      [(j.errors.engine?.plan?.planned_gpu_bytes ?? 0) > 0, "the engine's diagnostics hold the planned GPU memory"],
      [j.errors.engine?.gpu?.peak_allocated_bytes != null, "the engine's diagnostics hold the GPU state"],
    ],
  },
  {
    name: "internal",
    query: "autorun=sample&grid=2mm&plan=short&fault=internal&fault_at=rigid",
    kinds: [...crashed, "RegistrationError"],
    checks: (j) => [[j.errors.gpu.some((e) => e.type === "GPUInternalError"), "the report holds the internal error"]],
  },
  {
    name: "limits",
    query: "autorun=sample&grid=1mm&limits=64",
    kinds: ["GpuLimit"],
    checks: (j, s) => [
      [/maxStorageBufferBindingSize/.test(s.message), "the panel names the limit"],
      [j.webgpu.engine?.limits_adapter?.max_storage_buffer_binding_size === 64 * 2 ** 20, "the report holds the adapter limits"],
      [j.webgpu.engine?.limits_requested != null && j.webgpu.engine?.limits_granted != null, "the report holds requested and granted limits"],
      [(j.webgpu.engine?.selftest?.attempts ?? 0) >= 1, "the report holds the self-test record"],
    ],
  },
  {
    name: "stall",
    query: "autorun=sample&grid=2mm&plan=short&fault=stall&fault_at=rigid&stall=6",
    kinds: ["Stall"],
    checks: (j, s) => [[/No progress/.test(s.message), "the panel says there is no progress"]],
  },
  {
    name: "kill",
    query: "autorun=sample&grid=2mm&plan=short&fault=kill&fault_at=rigid",
    reloads: true,
    kinds: ["UnfinishedRun"],
    checks: (j, s) => [
      [j.previous_run?.stage?.phase === "rigid", "the report holds the stage of the unfinished run"],
      [/did not finish/.test(s.message), "the panel says the last run did not finish"],
    ],
  },
  {
    name: "public",
    query: "autorun=info&fault=nogpu&limits=64&stall=5",
    public: true,
    result: "ready",
    probe: WORKER_PROBE,
    checks: (r, probe) => [
      [(r.selftest?.attempts ?? 0) >= 1, "a page that is not on 127.0.0.1 ignores the test parameters"],
      [typeof probe === "number" && probe !== 64 * 2 ** 20, `a worker that is not on 127.0.0.1 ignores ?limits= (got ${probe})`],
      ...gpuClean(r),
    ],
  },
  {
    name: "nogpu",
    query: "fault=nogpu",
    kinds: ["NoWebGPU"],
    checks: (j) => [[j.webgpu.navigator_gpu === false, "the report says WebGPU is missing"]],
  },
];

function common(c, s) {
  const j = s.json;
  const text = s.text ?? "";
  return [
    [c.kinds.includes(j.kind), `the report kind is one of ${c.kinds.join(", ")} (got ${j.kind})`],
    [s.mail != null && s.mail.length <= MAILTO_LIMIT, `the e-mail link is at most ${MAILTO_LIMIT} characters (got ${s.mail?.length})`],
    [!SAMPLE_NAMES.some((n) => text.includes(n)), "the report holds no file names"],
    [c.reloads || j.debug != null, "the report says it was a test run"],
  ];
}

const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));

async function runResult(cdp, session, origin, c, timeout) {
  await cdp.send("Page.navigate", { url: `${origin}/browser.html?${c.query}` }, session);
  const started = Date.now();
  let state = null;
  while (Date.now() - started < timeout * 1000) {
    await sleep(1000);
    try {
      const r = await cdp.send("Runtime.evaluate", { expression: RESULT, returnByValue: true }, session);
      state = r.result.value;
    } catch {
      continue;
    }
    if (["ok", "ready", "error", "cancelled"].includes(state?.status)) break;
  }
  const seconds = Math.round((Date.now() - started) / 1000);
  if (state?.status !== c.result) {
    const why = state?.status ? `the page ended with "${state.status}": ${JSON.stringify(state.json).slice(0, 800)}` : `no result within ${timeout} s`;
    return { seconds, failures: [why], state };
  }
  let probe = null;
  if (c.probe) {
    const r = await cdp.send("Runtime.evaluate", { expression: c.probe, awaitPromise: true, returnByValue: true }, session);
    probe = r.result.value;
  }
  const failures = c.checks(state.json, probe).filter(([ok]) => !ok).map(([, what]) => what);
  return { seconds, failures, state };
}

async function runCase(cdp, session, origin, c, timeout) {
  await cdp.send("Page.navigate", { url: `${origin}/browser.html?${c.query}` }, session);
  const started = Date.now();
  let state = null;
  while (Date.now() - started < timeout * 1000) {
    await sleep(1000);
    try {
      const r = await cdp.send("Runtime.evaluate", { expression: STATE, returnByValue: true }, session);
      state = r.result.value;
    } catch {
      continue;
    }
    if (state?.json && c.kinds.includes(state.json.kind)) break;
  }
  const seconds = Math.round((Date.now() - started) / 1000);
  if (!state?.json) return { seconds, failures: [`no report within ${timeout} s (title ${state?.title}, message: ${state?.message})`], state };
  const failures = [...common(c, state), ...c.checks(state.json, state)].filter(([ok]) => !ok).map(([, what]) => what);
  return { seconds, failures, state };
}

async function main() {
  const opt = options(process.argv.slice(2));
  const server = await serve(opt.root);
  const origin = `http://127.0.0.1:${server.address().port}`;
  const publicOrigin = `http://fastwarp.test:${server.address().port}`;
  const unknown = [...(opt.only ?? [])].filter((name) => !CASES.some((c) => c.name === name));
  if (unknown.length) throw new Error(`unknown case(s) ${unknown.join(", ")}; cases: ${CASES.map((c) => c.name).join(", ")}`);
  const profile = mkdtempSync(join(tmpdir(), "fastwarp-web-faults-"));
  const flags = [
    "--headless=new",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-background-networking",
    "--disable-sync",
    "--mute-audio",
    "--remote-debugging-pipe",
    `--user-data-dir=${profile}`,
    "--enable-unsafe-webgpu",
    "--host-resolver-rules=MAP fastwarp.test 127.0.0.1",
    `--unsafely-treat-insecure-origin-as-secure=${publicOrigin}`,
  ];
  if (opt.swiftshader) {
    flags.push("--use-webgpu-adapter=swiftshader", "--enable-unsafe-swiftshader");
    if (process.platform === "linux") flags.push("--enable-features=Vulkan", "--use-angle=swiftshader");
  }
  if (process.platform === "linux" && process.env.CI) flags.push("--no-sandbox");
  flags.push("about:blank");
  const chrome = spawn(opt.chrome, flags, { stdio: ["ignore", "ignore", "pipe", "pipe", "pipe"] });
  const chromeLog = [];
  chrome.stderr.setEncoding("utf8");
  chrome.stderr.on("data", (t) => {
    chromeLog.push(t);
    if (chromeLog.length > 50) chromeLog.shift();
  });
  const cdp = new Cdp(chrome.stdio[3], chrome.stdio[4]);
  let failed = 0;
  let ran = 0;
  try {
    const { product } = await cdp.send("Browser.getVersion");
    console.log(`${product}, ${opt.swiftshader ? "SwiftShader" : "the system's GPU"}, ${origin}`);
    const { targetId } = await cdp.send("Target.createTarget", { url: "about:blank" });
    const { sessionId } = await cdp.send("Target.attachToTarget", { targetId, flatten: true });
    await cdp.send("Page.enable", {}, sessionId);
    await cdp.send("Runtime.enable", {}, sessionId);
    await cdp.send("Page.navigate", { url: `${origin}/blank` }, sessionId);
    await sleep(300);
    const probe = await cdp.send("Runtime.evaluate", { expression: ADAPTER, awaitPromise: true, returnByValue: true }, sessionId);
    console.log(`WebGPU adapter: ${probe.result.value}`);
    for (const c of CASES) {
      if (opt.only && !opt.only.has(c.name)) continue;
      if (opt.suite !== "all" && (c.suite ?? "faults") !== opt.suite) continue;
      const at = c.public ? publicOrigin : origin;
      ran += 1;
      await cdp.send("Page.navigate", { url: `${at}/blank` }, sessionId);
      await sleep(300);
      await cdp.send("Runtime.evaluate", { expression: "localStorage.clear(); sessionStorage.clear();" }, sessionId);
      cdp.console.length = 0;
      const go = c.result ? runResult : runCase;
      const { seconds, failures, state } = await go(cdp, sessionId, at, c, opt.timeout);
      if (failures.length === 0) {
        const extra = state.mail ? `, e-mail link ${state.mail.length} characters` : "";
        console.log(`PASS ${c.name} (${seconds} s${extra})`);
        continue;
      }
      failed += 1;
      console.log(`FAIL ${c.name} (${seconds} s)`);
      for (const f of failures) console.log(`  - ${f}`);
      console.log(`  message: ${state?.message ?? ""}`);
      if (state?.json?.errors) console.log(`  report errors: ${JSON.stringify(state.json.errors).slice(0, 1500)}`);
      console.log(`  console:\n    ${cdp.console.slice(-20).join("\n    ")}`);
    }
  } finally {
    const exited = new Promise((ok) => (chrome.exitCode !== null ? ok() : chrome.once("exit", ok)));
    await Promise.race([cdp.send("Browser.close").catch(() => null), sleep(5000)]);
    await Promise.race([exited, sleep(5000)]);
    chrome.kill();
    server.close();
    await Promise.race([exited, sleep(10000)]);
    try {
      rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 500 });
    } catch (err) {
      console.log(`could not remove the Chrome profile ${profile}: ${err.message}`);
    }
  }
  if (ran === 0) {
    console.log("no case ran: check --suite and --only");
    process.exit(1);
  }
  if (failed) {
    console.log(`${failed} case(s) failed. Chrome's last output:\n${chromeLog.join("").slice(-3000)}`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err?.stack ?? err);
  process.exit(1);
});
