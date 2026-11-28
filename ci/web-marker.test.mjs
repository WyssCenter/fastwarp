// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { beforeEach, test } from "node:test";

class MemoryStorage {
  constructor() {
    this.items = new Map();
  }
  get length() {
    return this.items.size;
  }
  key(i) {
    return [...this.items.keys()][i] ?? null;
  }
  getItem(k) {
    return this.items.has(k) ? this.items.get(k) : null;
  }
  setItem(k, v) {
    this.items.set(k, String(v));
  }
  removeItem(k) {
    this.items.delete(k);
  }
  clear() {
    this.items.clear();
  }
}

const open = new Set();

class LocalChannel extends EventTarget {
  constructor(name) {
    super();
    this.name = name;
    this.onmessage = null;
    open.add(this);
  }
  postMessage(data) {
    for (const other of open) {
      if (other === this || other.name !== this.name) continue;
      setTimeout(() => {
        const event = new Event("message");
        event.data = data;
        other.onmessage?.(event);
        other.dispatchEvent(event);
      });
    }
  }
  close() {
    open.delete(this);
  }
}

const store = new MemoryStorage();
Object.defineProperty(globalThis, "localStorage", { value: store, configurable: true, writable: true });
Object.defineProperty(globalThis, "BroadcastChannel", { value: LocalChannel, configurable: true, writable: true });

const source = readFileSync(new URL("../docs/_static/web/diag.js", import.meta.url), "utf8");
const load = (tag) => import(`data:text/javascript,${encodeURIComponent(`${source}\n//${tag}`)}`);
const { beginRun, endRun, saveRun, takeUnfinished } = await load("main");

const RUN = "fastwarp-demo-run:";
const NOTE = "fastwarp-demo-unfinished";
const WAIT = 30;
const others = [];

function otherTab(id) {
  const c = new LocalChannel("fastwarp-demo");
  c.onmessage = ({ data }) => {
    if (data?.type === "ping") c.postMessage({ type: "alive", tab: id });
  };
  others.push(c);
}

function put(key, record) {
  store.setItem(key, JSON.stringify(record));
}

const keys = () => [...store.items.keys()];

beforeEach(() => {
  store.clear();
  while (others.length) others.pop().close();
});

test("a run that stopped replaces an older note", async () => {
  const now = Date.now();
  put(NOTE, { started_utc: "old", grid: "1mm", tab: "a", beat: now - 86400000 });
  put(RUN + "b", { started_utc: "new", grid: "1mm", stage: { phase: "syn" }, tab: "b", beat: now - 5000 });
  const previous = await takeUnfinished({ now, wait: WAIT });
  assert.equal(previous.started_utc, "new");
  assert.equal(previous.tab, undefined);
  assert.equal(previous.beat, undefined);
  assert.equal(JSON.parse(store.getItem(NOTE)).started_utc, "new");
  assert.deepEqual(keys(), [NOTE]);
});

test("the newest of several stopped runs becomes the note", async () => {
  const now = Date.now();
  put(RUN + "c", { started_utc: "newest", tab: "c", beat: now - 1000 });
  put(RUN + "b", { started_utc: "older", tab: "b", beat: now - 300000 });
  const previous = await takeUnfinished({ now, wait: WAIT });
  assert.equal(previous.started_utc, "newest");
  assert.deepEqual(keys(), [NOTE]);
});

test("the run of a tab that is still open is left alone", async () => {
  const now = Date.now();
  otherTab("live");
  put(RUN + "live", { started_utc: "running", tab: "live", beat: now - 1000 });
  assert.equal(await takeUnfinished({ now, wait: WAIT }), null);
  assert.deepEqual(keys(), [RUN + "live"]);
});

test("an older stopped run does not replace a newer note", async () => {
  const now = Date.now();
  put(NOTE, { started_utc: "note", tab: "a", beat: now - 1000 });
  put(RUN + "b", { started_utc: "stale", tab: "b", beat: now - 600000 });
  const previous = await takeUnfinished({ now, wait: WAIT });
  assert.equal(previous.started_utc, "note");
  assert.deepEqual(keys(), [NOTE]);
});

test("without BroadcastChannel a run with a recent heartbeat counts as live", async () => {
  const now = Date.now();
  Object.defineProperty(globalThis, "BroadcastChannel", { value: undefined, configurable: true, writable: true });
  try {
    const bare = await load("bare");
    put(RUN + "live", { started_utc: "running", tab: "live", beat: now - 1000 });
    put(RUN + "dead", { started_utc: "stopped", tab: "dead", beat: now - 600000 });
    const previous = await bare.takeUnfinished({ now, wait: WAIT });
    assert.equal(previous.started_utc, "stopped");
    assert.deepEqual(keys().sort(), [RUN + "live", NOTE]);
  } finally {
    Object.defineProperty(globalThis, "BroadcastChannel", { value: LocalChannel, configurable: true, writable: true });
  }
});

test("a note that cannot be written is still shown, and the run record is kept", async () => {
  const now = Date.now();
  put(RUN + "b", { started_utc: "new", tab: "b", beat: now - 5000 });
  const setItem = store.setItem;
  store.setItem = () => {
    throw new Error("QuotaExceededError");
  };
  try {
    const previous = await takeUnfinished({ now, wait: WAIT });
    assert.equal(previous.started_utc, "new");
    assert.deepEqual(keys(), [RUN + "b"]);
  } finally {
    store.setItem = setItem;
  }
});

test("a damaged run record is dropped", async () => {
  store.setItem(RUN + "x", "{not json");
  assert.equal(await takeUnfinished({ wait: WAIT }), null);
  assert.deepEqual(keys(), []);
});

test("starting a run replaces the note, ending it removes the run record", async () => {
  put(NOTE, { started_utc: "note", tab: "a", beat: 1 });
  beginRun({ started_utc: "now", grid: "2mm" });
  const own = keys();
  assert.equal(own.length, 1);
  assert.ok(own[0].startsWith(RUN));
  assert.equal(await takeUnfinished({ wait: WAIT }), null);
  saveRun({ started_utc: "now", grid: "2mm", stage: { phase: "rigid" } });
  assert.equal(JSON.parse(store.getItem(own[0])).stage.phase, "rigid");
  endRun(false);
  assert.deepEqual(keys(), []);
});

test("a finished run removes the note", async () => {
  beginRun({ started_utc: "now" });
  put(NOTE, { started_utc: "note", tab: "a", beat: 1 });
  endRun(true);
  assert.deepEqual(keys(), []);
});
