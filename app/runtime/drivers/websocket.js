// Websocket action driver.
//
// Connects to an external action source over a WebSocket (the dexact Python
// server is the canonical one). The driver:
//
//   - kicks off the simulation by calling step(null) on connection open,
//   - forwards each emitted PartitionState whose name is in
//     options.forwardPartitions to the socket as raw protobuf bytes,
//   - on each inbound socket message, treats the payload as ActionState
//     bytes and advances the simulation by one step with those actions,
//   - reconnects with exponential backoff (capped) on close/error.
//
// `options`:
//   url               default 'ws://localhost:2112'
//   forwardPartitions array of partition names whose state should be sent
//                     to the server. Default: empty (no forwarding).

self.createDriver = function (env, options) {
    const url = (options && options.url) || 'ws://localhost:2112';
    const forwardPartitions = (options && options.forwardPartitions) || [];

    let socket = null;
    let connected = false;
    let reconnectDelay = 0;
    let stopped = false;

    function connect() {
        socket = new WebSocket(url);
        socket.binaryType = 'arraybuffer';

        socket.onopen = function () {
            connected = true;
            reconnectDelay = 0;
            env.postToPage({ type: 'status', data: 'connected to ' + url });
            // Kick off the simulation. The first step has no incoming
            // actions; subsequent steps are driven by socket.onmessage.
            env.step(null);
        };

        socket.onmessage = function (event) {
            env.step(new Uint8Array(event.data));
        };

        socket.onclose = function () {
            connected = false;
            if (!stopped) scheduleReconnect();
        };
        socket.onerror = function () {
            connected = false;
            if (!stopped) scheduleReconnect();
        };
    }

    function scheduleReconnect() {
        if (reconnectDelay === 0) {
            reconnectDelay = 100;
        } else {
            reconnectDelay = Math.min(reconnectDelay * 1.5, 2000);
        }
        setTimeout(function () { if (!stopped) connect(); }, reconnectDelay);
    }

    return {
        start: function () {
            env.onPartitionState(function (bytes, partitionName) {
                if (forwardPartitions.indexOf(partitionName) >= 0 &&
                    socket && socket.readyState === WebSocket.OPEN) {
                    try { socket.send(bytes); } catch (e) { /* ignore */ }
                }
            });
            connect();
        },
        stop: function () {
            stopped = true;
            if (socket) socket.close();
        },
    };
};
