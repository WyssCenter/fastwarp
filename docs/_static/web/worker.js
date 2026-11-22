// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

import init, { cancel, gpu_info, register, view_info, view_slice } from "./pkg/fastwarp_web.js";

let ready = null;
let cancelRequested = false;

function boot() {
  ready ??= init();
  return ready;
}

function memoryBytes(wasm) {
  try {
    return wasm.memory.buffer.byteLength;
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
    const wasm = await boot();
    if (data.type === "info") {
      postMessage({ type: "info", info: JSON.parse(await gpu_info()) });
    } else if (data.type === "run") {
      if (cancelRequested) throw cancelled();
      const zip = await register(
        new Uint8Array(data.fixed),
        new Uint8Array(data.moving),
        JSON.stringify(data.options),
        (json) => postMessage({ type: "progress", progress: JSON.parse(json) }),
      );
      const view = view_info();
      postMessage(
        { type: "done", zip: zip.buffer, memory: memoryBytes(wasm), view: view ? JSON.parse(view) : null },
        [zip.buffer],
      );
    }
  } catch (err) {
    postMessage({
      type: "error",
      name: err?.name ?? "Error",
      message: String(err?.message ?? err),
    });
  }
};
