// worker.js runs the compiled stochadex simulation inside a Web Worker and
// brokers between three parties:
//
//   1. The wasm module (which exports a global `stepSimulation(callback,
//      actionBytes-or-null)` function — see pkg/simio/step.go).
//   2. The page (which postMessages a 'start' control message at boot and
//      may postMessage further driver-specific messages thereafter).
//   3. A driver (the pluggable bit) — one of the files under runtime/drivers/.
//      The driver decides when to advance the simulation and where actions
//      come from.
//
// Lifecycle:
//
//   page → worker:
//     { action: 'start', wasmBinary, driver: { kind, options } }
//
//   worker (this file):
//     1. loadWasm(wasmBinary) → registers `stepSimulation`.
//     2. importScripts('drivers/' + driver.kind + '.js') → defines
//        self.createDriver(env, options).
//     3. driver = self.createDriver(env, driver.options); driver.start().
//
//   worker → page (continuously):
//     { type: 'partitionState', data: { partitionName, timesteps, state: {values} } }
//     { type: 'status', data: <string> }
//     { type: 'error',  data: <string> }
//
// All driver-specific behaviour (network connections, page-input handling,
// pacing) lives in the driver. This file knows nothing about either.

self.importScripts('wasm_exec.js');
self.importScripts('google-protobuf.js');
self.importScripts('partition_state_pb.js');
self.importScripts('action_state_pb.js');

let go;
let wasmReady = false;

// step(actionBytes | null) advances the simulation by one tick.
//   - On every call, the wasm side sees `handlePartitionState` as its
//     output callback (re-registered each time so a driver could swap it,
//     though none currently does).
//   - actionBytes may be a Uint8Array of serialised ActionState (which
//     the wasm side decodes and dispatches via ApplyActionState) or null
//     (no action input — partitions keep their previous action_state_values).
function step(actionBytes) {
    if (!wasmReady) return;
    self.stepSimulation(handlePartitionState, actionBytes);
}

// Subscribers to every PartitionState the wasm emits. The first subscriber
// is registered by this file (it forwards to the page); drivers can add
// further subscribers (the websocket driver, for example, forwards selected
// partitions to its socket).
const partitionStateSubscribers = [];

function onPartitionState(callback) {
    partitionStateSubscribers.push(callback);
}

// Called by the wasm side once per output partition per step. The raw bytes
// are deserialised once here and the decoded view is shared with all
// subscribers (alongside the bytes, in case a subscriber wants to forward
// them onward without re-encoding).
function handlePartitionState(bytes) {
    const message = proto.PartitionState.deserializeBinary(bytes);
    const partitionName = message.getPartitionName();
    for (let i = 0; i < partitionStateSubscribers.length; i++) {
        partitionStateSubscribers[i](bytes, partitionName, message);
    }
}

// Default subscriber: forward every partition state to the page.
onPartitionState(function (bytes, partitionName, message) {
    self.postMessage({
        type: 'partitionState',
        data: {
            timesteps: message.getCumulativeTimesteps(),
            partitionName: partitionName,
            state: { values: message.getStateList() },
        },
    });
});

function postToPage(msg) {
    self.postMessage(msg);
}

// Page-message subscribers. The 'start' message is handled directly in
// onmessage below; every other message is fanned out to whatever the
// loaded driver subscribed during start().
const pageMessageSubscribers = [];

function onPageMessage(callback) {
    pageMessageSubscribers.push(callback);
}

let driver = null;
let started = false;

self.onmessage = async function (event) {
    const msg = event.data || {};

    if (!started && msg.action === 'start') {
        started = true;
        await loadWasm(msg.wasmBinary);
        loadDriver(msg.driver || { kind: 'websocket', options: {} });
        return;
    }

    if (started) {
        for (let i = 0; i < pageMessageSubscribers.length; i++) {
            pageMessageSubscribers[i](msg);
        }
    }
};

async function loadWasm(wasmBinary) {
    try {
        go = new Go();
        const result = await WebAssembly.instantiateStreaming(
            fetch(wasmBinary), go.importObject);
        go.run(result.instance);
        wasmReady = true;
    } catch (err) {
        postToPage({ type: 'error', data: 'wasm load failed: ' + err.message });
        throw err;
    }
}

function loadDriver(spec) {
    const kind = spec.kind || 'websocket';
    try {
        self.importScripts('drivers/' + kind + '.js');
    } catch (err) {
        postToPage({ type: 'error', data: 'driver load failed: ' + err.message });
        throw err;
    }
    const env = { step, onPartitionState, onPageMessage, postToPage };
    driver = self.createDriver(env, spec.options || {});
    driver.start();
}
