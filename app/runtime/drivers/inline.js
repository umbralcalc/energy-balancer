// Inline action driver.
//
// Drives the simulation from in-page state (slider values, button clicks,
// keyboard input — anything the page chooses to surface) rather than from
// an external network connection. This is the driver the decision-support
// dashboard use case uses.
//
// Pacing: the driver advances the simulation at a fixed wall-clock interval
// (options.intervalMs, default 33 ms ≈ 30 Hz). Each tick uses the most
// recent action values the page has posted; if the page hasn't posted any,
// the tick happens with null actions and partitions keep their previous
// action_state_values.
//
// Page → worker message protocol used by this driver:
//   { action: 'setActions', partitions: { partitionName: [v0, v1, ...], ... } }
//
// `options`:
//   intervalMs  tick interval in ms. Default 33.

self.createDriver = function (env, options) {
    const intervalMs = (options && options.intervalMs) || 33;

    // Most recent ActionState bytes assembled from page input. Set to null
    // after consumption so partitions retain their previous action_state_values
    // unless the page actively republishes them.
    let latestActionBytes = null;
    let timerId = null;
    let stopped = false;

    function encodeActionState(partitions) {
        const msg = new proto.ActionState();
        const map = msg.getPartitionsMap();
        for (const name in partitions) {
            if (!Object.prototype.hasOwnProperty.call(partitions, name)) continue;
            const values = partitions[name];
            const av = new proto.ActionValues();
            av.setValuesList(values);
            map.set(name, av);
        }
        return msg.serializeBinary();
    }

    function tick() {
        if (stopped) return;
        const bytes = latestActionBytes;
        latestActionBytes = null;
        env.step(bytes);
    }

    return {
        start: function () {
            env.onPageMessage(function (msg) {
                if (msg && msg.action === 'setActions' && msg.partitions) {
                    latestActionBytes = encodeActionState(msg.partitions);
                }
            });
            env.postToPage({ type: 'status', data: 'inline driver ready' });
            // Kick off the first step (no actions yet) immediately so the
            // renderer has something to draw, then settle into the timer.
            env.step(null);
            timerId = setInterval(tick, intervalMs);
        },
        stop: function () {
            stopped = true;
            if (timerId !== null) clearInterval(timerId);
        },
    };
};
