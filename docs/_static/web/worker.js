// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

import init, {
  cancel,
  debug_fault,
  gpu_info,
  last_failure,
  register,
  view_info,
  view_slice,
} from "./pkg/fastwarp_web.js";
import { clip, featureList, limitsObject } from "./diag.js";

let ready = null;
let wasm = null;
let cancelRequested = false;
let failed = false;
const counts = {};

const FAULTS = new Set(["panic", "alloc", "lost", "lostoom", "oom", "internal", "stall"]);
const SPEC_LIMITS = {
  maxBufferSize: 268435456,
  maxStorageBufferBindingSize: 134217728,
  maxUniformBufferBindingSize: 65536,
  maxStorageBuffersPerShaderStage: 8,
  maxUniformBuffersPerShaderStage: 12,
  maxBindGroups: 4,
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65535,
};

function readDebug() {
  try {
    if (!["127.0.0.1", "localhost", "[::1]"].includes(self.location.hostname)) throw new Error("not loopback");
    const p = new URLSearchParams(self.location.search);
    const fault = p.get("fault");
    const limits = p.get("limits");
    let clamp = null;
    if (limits === "default") clamp = SPEC_LIMITS;
    else if (/^\d+$/.test(limits ?? "")) {
      const bytes = Number(limits) * 2 ** 20;
      clamp = { maxBufferSize: bytes, maxStorageBufferBindingSize: bytes };
    }
    return { fault: FAULTS.has(fault) ? fault : null, at: p.get("fault_at") || "rigid", clamp };
  } catch {
    return { fault: null, at: "rigid", clamp: null };
  }
}

const debug = readDebug();
const fault = { armed: false, fired: false, device: null, scopes: new WeakMap() };

function plain(value) {
  try {
    return JSON.parse(JSON.stringify(value ?? null));
  } catch {
    return null;
  }
}

function emit(kind, detail) {
  try {
    const key = detail?.kind ?? kind;
    counts[key] = (counts[key] ?? 0) + 1;
    const always = kind === "device" || kind === "adapter" || key === "lost";
    if (!always && counts[key] > 100) return;
    postMessage({ type: "diag", kind, detail: plain(detail) });
  } catch {
    counts.failed = (counts.failed ?? 0) + 1;
  }
}

function clampLimits() {
  const descriptor = Object.getOwnPropertyDescriptor(GPUAdapter.prototype, "limits");
  if (!descriptor?.get) return;
  Object.defineProperty(GPUAdapter.prototype, "limits", {
    configurable: true,
    get() {
      const real = descriptor.get.call(this);
      const out = {};
      for (const key in real) {
        const value = real[key];
        out[key] = typeof value === "number" && key in debug.clamp ? Math.min(value, debug.clamp[key]) : value;
      }
      return out;
    },
  });
}

function syntheticError(filter) {
  if (!fault.armed) return null;
  if (debug.fault === "oom" && filter === "out-of-memory") {
    return new GPUOutOfMemoryError("FastWarp fault injection: out of memory");
  }
  if (fault.fired) return null;
  if (debug.fault === "lostoom" && filter === "out-of-memory") {
    fault.fired = true;
    fault.device?.destroy();
    return new GPUOutOfMemoryError("FastWarp fault injection: out of memory before the device is lost");
  }
  if (debug.fault === "internal" && filter === "internal") {
    fault.fired = true;
    return new GPUInternalError("FastWarp fault injection: internal error");
  }
  return null;
}

function tapGpu() {
  if (typeof GPU !== "undefined") {
    const requestAdapter = GPU.prototype.requestAdapter;
    GPU.prototype.requestAdapter = async function (options) {
      try {
        const adapter = await requestAdapter.call(this, options);
        emit("adapter", {
          options,
          granted: Boolean(adapter),
          fallback: adapter ? Boolean(adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter) : null,
        });
        return adapter;
      } catch (err) {
        emit("gpu", { kind: "adapter-error", type: err?.name, message: clip(err?.message ?? err) });
        throw err;
      }
    };
  }
  if (typeof GPUAdapter !== "undefined") {
    if (debug.clamp) clampLimits();
    const requestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (descriptor) {
      let requested = {};
      try {
        requested = {
          requested_limits: plain(descriptor?.requiredLimits ?? {}),
          requested_features: plain([...(descriptor?.requiredFeatures ?? [])]),
        };
      } catch {
        requested = {};
      }
      let device;
      try {
        device = await requestDevice.call(this, descriptor);
      } catch (err) {
        emit("device", { ...requested, error: clip(`${err?.name}: ${err?.message ?? err}`) });
        throw err;
      }
      try {
        if (debug.fault) fault.device = device;
        emit("device", {
          ...requested,
          granted_limits: limitsObject(device.limits),
          granted_features: featureList(device.features),
        });
        device.addEventListener("uncapturederror", (e) => {
          emit("gpu", { kind: "uncaptured", type: e.error?.constructor?.name, message: clip(e.error?.message) });
        });
        device.lost.then((info) => {
          emit("gpu", { kind: "lost", reason: info?.reason ?? "unknown", message: clip(info?.message) });
        });
      } catch {
        counts.failed = (counts.failed ?? 0) + 1;
      }
      return device;
    };
  }
  if (typeof GPUDevice !== "undefined") {
    if (["oom", "lostoom", "internal"].includes(debug.fault)) {
      const pushErrorScope = GPUDevice.prototype.pushErrorScope;
      GPUDevice.prototype.pushErrorScope = function (filter) {
        const stack = fault.scopes.get(this) ?? [];
        stack.push(filter);
        fault.scopes.set(this, stack);
        return pushErrorScope.call(this, filter);
      };
    }
    const popErrorScope = GPUDevice.prototype.popErrorScope;
    GPUDevice.prototype.popErrorScope = async function () {
      const filter = fault.scopes.get(this)?.pop();
      let error = await popErrorScope.call(this);
      if (!error && debug.fault) error = syntheticError(filter);
      if (error) emit("gpu", { kind: "scope", type: error.constructor?.name, message: clip(error.message) });
      return error;
    };
  }
  const consoleError = console.error.bind(console);
  console.error = (...args) => {
    try {
      emit("console", { text: clip(args.map(String).join(" "), 1500) });
    } catch {
      counts.failed = (counts.failed ?? 0) + 1;
    }
    consoleError(...args);
  };
}

function boot() {
  ready ??= init().then((exports) => {
    wasm = exports;
    return exports;
  });
  return ready;
}

function memoryBytes() {
  try {
    return wasm?.memory?.buffer?.byteLength ?? null;
  } catch {
    return null;
  }
}

function lastFailure() {
  try {
    return wasm ? JSON.parse(last_failure()) : null;
  } catch {
    return null;
  }
}

function diagnosticsOf(err) {
  try {
    return typeof err?.diagnostics === "string" ? JSON.parse(err.diagnostics) : null;
  } catch {
    return null;
  }
}

function postFailure(err) {
  if (failed) return;
  if (err instanceof WebAssembly.RuntimeError) failed = true;
  postMessage({
    type: "error",
    name: err?.name ?? "Error",
    message: String(err?.message ?? err),
    stack: clip(err?.stack, 2000) || null,
    memory: memoryBytes(),
    failure: lastFailure(),
    diagnostics: diagnosticsOf(err),
  });
}

tapGpu();

self.addEventListener("error", (e) => {
  if (!(e.error instanceof WebAssembly.RuntimeError)) return;
  e.preventDefault();
  postFailure(e.error);
});

self.addEventListener("unhandledrejection", (e) => {
  emit("gpu", { kind: "rejection", type: e.reason?.name, message: clip(e.reason?.message ?? e.reason) });
  if (e.reason instanceof WebAssembly.RuntimeError) {
    e.preventDefault();
    postFailure(e.reason);
  }
});

function withDeviceMemory(options) {
  const gb = self.navigator?.deviceMemory;
  return typeof gb === "number" && Number.isFinite(gb) && gb > 0 ? { ...options, device_memory_gb: gb } : options;
}

function cancelled() {
  const err = new Error("the registration was cancelled");
  err.name = "Cancelled";
  return err;
}

function onProgress(json) {
  const progress = JSON.parse(json);
  if (debug.fault && progress.phase === debug.at && !fault.armed) {
    fault.armed = true;
    if (debug.fault === "lost") fault.device?.destroy();
  }
  if (debug.fault === "stall" && fault.armed) return;
  postMessage({ type: "progress", progress, memory: memoryBytes() });
}

self.onmessage = async ({ data }) => {
  if (data.type === "cancel") {
    cancelRequested = true;
    try {
      cancel();
    } catch {
      cancelRequested = true;
    }
    return;
  }
  if (data.type === "slice") {
    try {
      await boot();
      const bytes = view_slice(data.axis, data.index, data.layer);
      postMessage({ type: "slice", id: data.id, data: bytes.buffer }, [bytes.buffer]);
    } catch (err) {
      postMessage({ type: "slice-error", id: data.id, message: String(err?.message ?? err) });
    }
    return;
  }
  if (data.type === "run") cancelRequested = false;
  try {
    await boot();
    if (data.type === "info") {
      postMessage({ type: "info", info: JSON.parse(await gpu_info()), memory: memoryBytes() });
    } else if (data.type === "run") {
      if (cancelRequested) throw cancelled();
      if (debug.fault === "panic" || debug.fault === "alloc") debug_fault(debug.fault, debug.at);
      const zip = await register(
        new Uint8Array(data.fixed),
        new Uint8Array(data.moving),
        JSON.stringify(withDeviceMemory(data.options)),
        onProgress,
      );
      if (debug.fault === "stall" && fault.armed) return;
      const view = view_info();
      postMessage(
        { type: "done", zip: zip.buffer, memory: memoryBytes(), view: view ? JSON.parse(view) : null },
        [zip.buffer],
      );
    }
  } catch (err) {
    postFailure(err);
  }
};
