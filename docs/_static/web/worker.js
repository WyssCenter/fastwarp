// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

import init, { cancel, gpu_info, register, view_info, view_slice } from "./pkg/fastwarp_web.js";
import { clip, featureList, limitsObject } from "./diag.js";

let ready = null;
let wasm = null;
let cancelRequested = false;
let sent = 0;

function plain(value) {
  try {
    return JSON.parse(JSON.stringify(value ?? null));
  } catch {
    return null;
  }
}

function emit(kind, detail) {
  if (sent >= 200) return;
  sent += 1;
  postMessage({ type: "diag", kind, detail: plain(detail) });
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
    const requestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (descriptor) {
      const requested = {
        requested_limits: plain(descriptor?.requiredLimits ?? {}),
        requested_features: plain([...(descriptor?.requiredFeatures ?? [])]),
      };
      let device;
      try {
        device = await requestDevice.call(this, descriptor);
      } catch (err) {
        emit("device", { ...requested, error: clip(`${err?.name}: ${err?.message ?? err}`) });
        throw err;
      }
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
      return device;
    };
  }
  if (typeof GPUDevice !== "undefined") {
    const popErrorScope = GPUDevice.prototype.popErrorScope;
    GPUDevice.prototype.popErrorScope = async function () {
      const error = await popErrorScope.call(this);
      if (error) emit("gpu", { kind: "scope", type: error.constructor?.name, message: clip(error.message) });
      return error;
    };
  }
  const consoleError = console.error.bind(console);
  console.error = (...args) => {
    emit("console", { text: clip(args.map(String).join(" "), 1500) });
    consoleError(...args);
  };
  self.addEventListener("unhandledrejection", (e) => {
    emit("gpu", { kind: "rejection", type: e.reason?.name, message: clip(e.reason?.message ?? e.reason) });
  });
}

tapGpu();

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

function cancelled() {
  const err = new Error("the registration was cancelled");
  err.name = "Cancelled";
  return err;
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
      const zip = await register(
        new Uint8Array(data.fixed),
        new Uint8Array(data.moving),
        JSON.stringify(data.options),
        (json) => postMessage({ type: "progress", progress: JSON.parse(json), memory: memoryBytes() }),
      );
      const view = view_info();
      postMessage(
        { type: "done", zip: zip.buffer, memory: memoryBytes(), view: view ? JSON.parse(view) : null },
        [zip.buffer],
      );
    }
  } catch (err) {
    postMessage({
      type: "error",
      name: err?.name ?? "Error",
      message: String(err?.message ?? err),
      stack: clip(err?.stack, 2000) || null,
      memory: memoryBytes(),
    });
  }
};
