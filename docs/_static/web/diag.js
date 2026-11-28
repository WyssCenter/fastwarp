// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

export const REPORT_ADDRESS = "richard.koehler@wysscenter.ch";
export const MAILTO_LIMIT = 1800;

const RUN_PREFIX = "fastwarp-demo-run:";
const HEARTBEAT_MS = 120000;
const UNFINISHED_KEY = "fastwarp-demo-unfinished";
const CHANNEL = "fastwarp-demo";
const PING_MS = 1000;

function wellFormed(s) {
  if (typeof s.toWellFormed === "function") return s.toWellFormed();
  let out = "";
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    const next = s.charCodeAt(i + 1);
    if (c >= 0xd800 && c <= 0xdbff && next >= 0xdc00 && next <= 0xdfff) {
      out += s[i] + s[i + 1];
      i += 1;
    } else if (c >= 0xd800 && c <= 0xdfff) {
      out += String.fromCharCode(0xfffd);
    } else {
      out += s[i];
    }
  }
  return out;
}

function cut(s, n) {
  const out = s.slice(0, Math.max(0, n));
  const last = out.charCodeAt(out.length - 1);
  return last >= 0xd800 && last <= 0xdbff ? out.slice(0, -1) : out;
}

export function clip(text, n = 600) {
  const s = wellFormed(text == null ? "" : String(text));
  return s.length > n ? `${cut(s, n)}...` : s;
}

export function size(n) {
  if (n == null || !Number.isFinite(n)) return "?";
  if (n >= 2 ** 30) return `${(n / 2 ** 30).toFixed(1)} GiB`;
  if (n >= 2 ** 20) return `${Math.round(n / 2 ** 20)} MiB`;
  if (n >= 2 ** 10) return `${Math.round(n / 2 ** 10)} KiB`;
  return `${n} B`;
}

export function limitsObject(limits) {
  const out = {};
  if (!limits) return out;
  const keys = [];
  for (const key in limits) keys.push(key);
  for (const key of keys.sort()) {
    const value = limits[key];
    if (typeof value === "number") out[key] = value;
  }
  return out;
}

export function featureList(features) {
  try {
    return [...features].map(String).sort();
  } catch {
    return [];
  }
}

export function platformOf(ua, touchPoints = 0, platformVersion = null) {
  const num = (re) => {
    const m = ua.match(re);
    return m ? Number(m[1]) : null;
  };
  const ios = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && touchPoints > 1);
  let os = "other";
  let osVersion = null;
  if (ios) {
    os = "ios";
    osVersion = num(/OS (\d+)[_.]\d+/);
  } else if (/Android/.test(ua)) {
    os = "android";
    const declared = Number.parseInt(platformVersion ?? "", 10);
    if (Number.isFinite(declared) && declared > 0) osVersion = declared;
    else if (!/Android 10; K\)/.test(ua)) osVersion = num(/Android (\d+)/);
  } else if (/CrOS/.test(ua)) os = "chromeos";
  else if (/Windows/.test(ua)) os = "windows";
  else if (/Macintosh/.test(ua)) os = "mac";
  else if (/Linux/.test(ua)) os = "linux";
  let browser = "other";
  let browserVersion = null;
  const rules = [
    ["samsung", /SamsungBrowser\/(\d+)/],
    ["edge", /(?:EdgA|EdgiOS|Edg)\/(\d+)/],
    ["chrome-ios", /CriOS\/(\d+)/],
    ["firefox-ios", /FxiOS\/(\d+)/],
    ["opera", /OPR\/(\d+)/],
    ["firefox", /Firefox\/(\d+)/],
    ["chrome", /Chrome\/(\d+)/],
    ["safari", /Version\/(\d+)[.\d]* (?:Mobile\/\S+ )?Safari/],
  ];
  for (const [name, re] of rules) {
    const v = num(re);
    if (v != null) {
      browser = name;
      browserVersion = v;
      break;
    }
  }
  const inApp = /FBAN|FBAV|Instagram|LinkedInApp|GSA\/|Line\/|Snapchat|; wv\)/.test(ua);
  const mobile = ios || /Android|Mobile/.test(ua);
  return { os, osVersion, browser, browserVersion, inApp, mobile };
}

export function supportHint(p, what) {
  if (p.inApp) {
    return "This looks like a browser inside another app. Open the page in Safari (iPhone, iPad) or Chrome (Android) instead.";
  }
  if (p.os === "ios") {
    if (p.browser === "safari" && p.browserVersion && p.browserVersion < 26) {
      return `This is Safari ${p.browserVersion}. WebGPU needs Safari 26, which comes with iOS and iPadOS 26: update in Settings > General > Software Update.`;
    }
    if (p.browser !== "safari") {
      return "On iPhone and iPad, open this page in Safari on iOS or iPadOS 26 or later; other browsers there may not offer WebGPU.";
    }
    return "Safari offers WebGPU from iOS and iPadOS 26. If this device runs iOS 26, check that WebGPU is not switched off in Settings > Apps > Safari > Advanced > Feature Flags.";
  }
  if (p.os === "android") {
    if (p.browser === "firefox") return "Firefox on Android does not offer WebGPU yet. Use Chrome.";
    if (p.osVersion && p.osVersion < 12) {
      return `This phone runs Android ${p.osVersion}. Chrome offers WebGPU from Android 12.`;
    }
    if (p.browser === "chrome" && p.browserVersion && p.browserVersion < 121) {
      return `Update Chrome: WebGPU on Android needs Chrome 121 or later (this is Chrome ${p.browserVersion}).`;
    }
    const gpus =
      "Chrome offers WebGPU on Android 12 and later on phones with Arm Mali, Qualcomm Adreno or Intel GPUs, and with Imagination GPUs from Android 16; other phones, for example Samsung phones with Exynos chips, are not supported yet.";
    return p.browser === "chrome" ? gpus : `Try Chrome. ${gpus}`;
  }
  if (p.os === "mac" && p.browser === "safari") {
    return "Safari offers WebGPU on macOS 26 (Tahoe) and later. On older macOS, use Chrome or Edge.";
  }
  if (p.os === "linux" && p.browser === "firefox") return "Firefox on Linux offers WebGPU only in Nightly builds. Use Chrome.";
  if (p.os === "linux") {
    return "Chrome on Linux offers WebGPU on Intel GPUs of generation 12 or newer and on NVIDIA GPUs under Wayland; chrome://gpu says more.";
  }
  if (p.browser === "firefox") return "Firefox offers WebGPU from version 141 on Windows and 147 on macOS.";
  if (what === "no-adapter") {
    return "The GPU may be on the browser's block list, or WebGPU may be switched off (see chrome://gpu or about:support).";
  }
  return "";
}

async function headerBytes(buffer, want) {
  const u8 = new Uint8Array(buffer);
  if (!(u8[0] === 0x1f && u8[1] === 0x8b)) return u8.subarray(0, want);
  if (typeof DecompressionStream === "undefined") return null;
  const stream = new Blob([u8.subarray(0, Math.min(u8.length, 1 << 16))])
    .stream()
    .pipeThrough(new DecompressionStream("gzip"));
  const reader = stream.getReader();
  const out = new Uint8Array(want);
  let have = 0;
  try {
    while (have < want) {
      const { done, value } = await reader.read();
      if (done) break;
      const take = Math.min(value.length, want - have);
      out.set(value.subarray(0, take), have);
      have += take;
    }
  } catch {
    if (have < 348) return null;
  }
  reader.cancel().catch(() => {});
  return out.subarray(0, have);
}

export async function niftiGeometry(buffer) {
  const h = await headerBytes(buffer, 540);
  if (!h || h.length < 348) return null;
  const v = new DataView(h.buffer, h.byteOffset, h.byteLength);
  let little = null;
  for (const le of [true, false]) {
    const n = v.getInt32(0, le);
    if (n === 348 || n === 540) little = le;
  }
  if (little === null) return null;
  const round = (x) => Math.round(x * 1e4) / 1e4;
  if (v.getInt32(0, little) === 348) {
    return {
      nifti: 1,
      size: [1, 2, 3].map((i) => v.getInt16(40 + 2 * i, little)),
      spacing: [1, 2, 3].map((i) => round(Math.abs(v.getFloat32(76 + 4 * i, little)))),
      datatype: v.getInt16(70, little),
    };
  }
  if (h.length < 540) return null;
  return {
    nifti: 2,
    size: [1, 2, 3].map((i) => Number(v.getBigInt64(16 + 8 * i, little))),
    spacing: [1, 2, 3].map((i) => round(Math.abs(v.getFloat64(104 + 8 * i, little)))),
    datatype: v.getInt16(12, little),
  };
}

export function isoDims(geometry, mm) {
  if (!geometry) return null;
  return geometry.size.map((n, a) => {
    const s = geometry.spacing[a];
    if (Math.abs(s - mm) < 1e-6) return n;
    return Math.max(1, Math.floor(((n - 1) * s) / mm + 1e-9) + 1);
  });
}

export function stageText(p) {
  if (!p) return "before the first step";
  let text = p.phase ?? "?";
  if (p.levels) text += ` level ${p.level}/${p.levels}`;
  if (p.iterations) text += ` iteration ${p.iteration}/${p.iterations}`;
  if (p.grid) text += ` (${p.grid.join("x")})`;
  return text;
}

const geometryText = (g) => (g ? `${g.size.join("x")} at ${g.spacing.join("x")} mm` : "?");

const KEY_LIMITS = [
  ["maxBufferSize", "buffer", true],
  ["maxStorageBufferBindingSize", "binding", true],
  ["maxStorageBuffersPerShaderStage", "storage buffers", false],
  ["maxComputeInvocationsPerWorkgroup", "invocations", false],
  ["maxComputeWorkgroupStorageSize", "workgroup storage", true],
];

function limitsText(adapter, granted) {
  return KEY_LIMITS.map(([key, name, bytes]) => {
    const show = (x) => (x == null ? "?" : bytes ? size(x) : String(x));
    return `${name} ${show(adapter?.[key])}/${show(granted?.[key])}`;
  }).join(", ");
}

function eventText(e) {
  return [e.kind, e.type, e.reason, clip(e.message ?? e.text, 160)].filter(Boolean).join(" ");
}

function engineText(failure, engine) {
  const parts = [];
  const plan = failure?.plan;
  const gpu = failure?.gpu;
  if (plan?.planned_gpu_bytes != null) parts.push(`planned GPU ${size(plan.planned_gpu_bytes)}`);
  if (gpu?.peak_allocated_bytes != null) parts.push(`GPU peak ${size(gpu.peak_allocated_bytes)}`);
  if (gpu?.lost) parts.push("device lost");
  if (gpu?.memory_evidence) parts.push(`memory evidence: ${gpu.memory_evidence}`);
  const ooms = gpu?.out_of_memory_errors ?? 0;
  if (ooms) parts.push(`${ooms} out-of-memory error${ooms === 1 ? "" : "s"}`);
  if (gpu?.pending_errors) parts.push(`pending: ${clip(gpu.pending_errors, 160)}`);
  const attempts = engine?.selftest?.attempts;
  if (attempts > 1) parts.push(`self-test needed ${attempts} attempts`);
  return parts.join(", ");
}

function compactLines(r) {
  const b = r.browser ?? {};
  const w = r.webgpu ?? {};
  const a = w.adapter ?? {};
  const run = r.run;
  const lines = [];
  const add = (text, drop = 0, label = "") => lines.push({ text, drop, label });
  add(`FastWarp browser demo report, ${r.time_utc}, FastWarp ${r.fastwarp ?? "?"}`);
  add(`Problem: ${r.kind}: ${clip(r.message, 400)}`);
  if (r.debug) add(`Test mode: ${Object.entries(r.debug).map(([k, v]) => `${k}=${clip(v, 20)}`).join(", ")}`);
  if (run) {
    add(`Stage: ${stageText(run.stage)}; ${run.elapsed_s} s in, ${run.since_progress_s} s since the last progress`);
    const grid = run.registration_grid?.join("x") ?? "?";
    add(`Images: ${run.grid} grid; fixed ${geometryText(run.fixed)}; moving ${geometryText(run.moving)}; registration grid ${grid}`);
  }
  const device = [
    b.model,
    b.platform,
    b.platform_version,
    b.mobile ? "mobile" : "",
    b.cores ? `${b.cores} cores` : "",
    b.device_memory_gb ? `${b.device_memory_gb} GB` : "",
  ]
    .filter(Boolean)
    .join(", ");
  add(`Device: ${device || "?"}`);
  add(`Browser: ${clip(b.user_agent, 260)}`);
  const named = [a.vendor, a.architecture, a.device, a.description].filter(Boolean).join(" ");
  const adapter = named || (w.adapter ? "unnamed" : "none");
  add(`WebGPU: navigator.gpu ${w.navigator_gpu ? "yes" : "no"}; adapter ${adapter}${a.fallback ? " (fallback)" : ""}`);
  if (w.limits || w.device?.granted_limits) {
    add(`Limits adapter/device: ${limitsText(w.limits, w.device?.granted_limits)}`, 3, "limits");
  }
  const e = r.errors ?? {};
  const failedAlloc = e.failed_alloc_bytes ? `; an allocation of ${size(e.failed_alloc_bytes)} failed` : "";
  add(`Memory: WebAssembly ${size(r.memory?.wasm_bytes)}${failedAlloc}`);
  const engine = engineText(e.engine, w.engine);
  if (engine) add(`Engine: ${engine}`, 3, "engine");
  if (e.panic) add(`Panic: ${clip(e.panic, 300)}`);
  if (e.gpu?.length) add(`GPU events: ${e.gpu.slice(-3).map(eventText).join(" | ")}`, 2, "GPU events");
  if (e.stack) add(`Stack: ${clip(e.stack.split("\n").slice(0, 3).join(" / "), 300)}`, 6, "stack");
  if (run?.timeline?.length) {
    const steps = run.timeline.slice(-8).map(([t, phase, level]) => `${t}s ${phase}${level ? ` ${level}` : ""}`);
    add(`Timeline: ${steps.join(", ")}`, 5, "timeline");
  }
  const p = r.previous_run;
  if (p) {
    const when = p.started_utc ?? "?";
    add(`Unfinished earlier run: ${when}, ${p.grid}, ${stageText(p.stage)}, ${p.elapsed_s} s in, WebAssembly ${size(p.wasm_bytes)}`, r.kind === "UnfinishedRun" ? 0 : 4, "earlier run");
  }
  const ask =
    r.kind === "DeviceReport"
      ? "Did the demo work on this device? Which phone or computer is it?"
      : "What did you see (a message, a page reload, a frozen page)? Which phone or computer is it?";
  add(ask);
  return lines;
}

export function compactText(report) {
  return compactLines(report)
    .map((l) => l.text)
    .join("\n");
}

export function reportText(report) {
  return [
    "FastWarp browser demo: diagnostic report",
    "It holds no images, no image header text and no file names.",
    "",
    compactText(report),
    "",
    JSON.stringify(report, null, 2),
    "",
  ].join("\n");
}

export function mailtoHref(report, limit = MAILTO_LIMIT) {
  const subject = encodeURIComponent(wellFormed(`FastWarp browser demo: ${report.kind}`));
  const lines = compactLines(report);
  const dropped = [];
  let shortened = false;
  const href = (cap = Infinity) => {
    const notes = [];
    if (dropped.length || shortened) {
      const what = dropped.length ? `; left out: ${dropped.join(", ")}` : "";
      notes.push(`Shortened for e-mail${what}. The page's Copy diagnostic report button gives the full report.`);
    }
    let body = wellFormed([...lines.map((l) => l.text), ...notes].join("\r\n"));
    if (body.length > cap) body = cut(body, cap);
    return `mailto:${REPORT_ADDRESS}?subject=${subject}&body=${encodeURIComponent(body)}`;
  };
  let out = href();
  while (out.length > limit) {
    let worst = -1;
    lines.forEach((l, i) => {
      if (l.drop > 0 && (worst < 0 || l.drop > lines[worst].drop)) worst = i;
    });
    if (worst < 0) break;
    dropped.push(lines[worst].label);
    lines.splice(worst, 1);
    out = href();
  }
  while (out.length > limit) {
    let longest = 1;
    lines.forEach((l, i) => {
      if (i > 0 && l.text.length > lines[longest].text.length) longest = i;
    });
    const text = lines[longest].text;
    if (text.length < 40) break;
    lines[longest].text = `${cut(text, Math.floor(text.length * 0.8))}...`;
    shortened = true;
    out = href();
  }
  let cap = lines.reduce((n, l) => n + l.text.length + 2, 0);
  while (out.length > limit && cap > 0) {
    shortened = true;
    cap = Math.floor(cap * 0.9);
    out = href(cap);
  }
  return { href: out, length: out.length, dropped };
}

function storage() {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

function attempt(f) {
  try {
    return f();
  } catch {
    return null;
  }
}

const tab = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
let channel;

function openChannel() {
  if (channel !== undefined) return channel;
  channel = attempt(() => new globalThis.BroadcastChannel(CHANNEL)) ?? null;
  if (channel) {
    channel.onmessage = ({ data }) => {
      if (data?.type === "ping") attempt(() => channel.postMessage({ type: "alive", tab }));
    };
  }
  return channel;
}

function aliveTabs(wait) {
  const c = openChannel();
  if (!c) return Promise.resolve(null);
  const alive = new Set();
  const listener = ({ data }) => {
    if (data?.type === "alive" && typeof data.tab === "string") alive.add(data.tab);
  };
  c.addEventListener("message", listener);
  attempt(() => c.postMessage({ type: "ping" }));
  return new Promise((done) =>
    setTimeout(() => {
      c.removeEventListener("message", listener);
      done(alive);
    }, wait),
  );
}

function readRecord(text) {
  const record = attempt(() => JSON.parse(text));
  return record && typeof record === "object" ? record : null;
}

export function saveRun(record) {
  const s = storage();
  openChannel();
  const visible = attempt(() => globalThis.document?.visibilityState === "visible") ?? false;
  if (s) attempt(() => s.setItem(RUN_PREFIX + tab, JSON.stringify({ ...record, tab, beat: Date.now(), visible })));
}

export function beginRun(record) {
  const s = storage();
  if (s) attempt(() => s.removeItem(UNFINISHED_KEY));
  saveRun(record);
}

export function endRun(success) {
  const s = storage();
  if (!s) return;
  attempt(() => s.removeItem(RUN_PREFIX + tab));
  if (success && attempt(() => readRecord(s.getItem(UNFINISHED_KEY))?.tab) === tab) {
    attempt(() => s.removeItem(UNFINISHED_KEY));
  }
}

export async function takeUnfinished({ now = Date.now(), wait = PING_MS } = {}) {
  const s = storage();
  if (!s) return null;
  openChannel();
  const runs = attempt(() => {
    const out = [];
    for (let i = 0; i < s.length; i++) {
      const key = s.key(i);
      if (key?.startsWith(RUN_PREFIX) && key !== RUN_PREFIX + tab) out.push({ key, text: s.getItem(key) });
    }
    return out;
  }) ?? [];
  for (const run of runs) run.record = readRecord(run.text);
  const recent = (r) => now - (r.record.beat ?? 0) < HEARTBEAT_MS;
  const alive = runs.length ? await aliveTabs(wait) : new Set();
  const quiet = (r) => !r.record.visible && recent(r);
  const live = (r) => r.record && (alive ? alive.has(r.record.tab) || quiet(r) : recent(r));
  const dead = runs.filter((r) => !live(r));
  let newest = attempt(() => readRecord(s.getItem(UNFINISHED_KEY)));
  let newestText = null;
  for (const run of dead) {
    if (run.record && (run.record.beat ?? 0) >= (newest?.beat ?? -Infinity)) {
      newest = run.record;
      newestText = run.text;
    }
  }
  const saved = !newestText || attempt(() => s.setItem(UNFINISHED_KEY, newestText) ?? true);
  if (saved) for (const run of dead) attempt(() => s.removeItem(run.key));
  if (!newest) return null;
  const { tab: _tab, beat: _beat, visible: _visible, ...record } = newest;
  return record;
}

export function staleIn(now = Date.now()) {
  const s = storage();
  if (!s) return null;
  return attempt(() => {
    let next = null;
    for (let i = 0; i < s.length; i++) {
      const key = s.key(i);
      if (!key?.startsWith(RUN_PREFIX) || key === RUN_PREFIX + tab) continue;
      const beat = readRecord(s.getItem(key))?.beat ?? 0;
      const left = Math.max(0, beat + HEARTBEAT_MS - now);
      next = next == null ? left : Math.min(next, left);
    }
    return next;
  });
}

export function dismissUnfinished() {
  const s = storage();
  if (s) attempt(() => s.removeItem(UNFINISHED_KEY));
}
