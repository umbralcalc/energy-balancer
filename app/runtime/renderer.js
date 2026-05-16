// Generic canvas renderer for stochadex simulations.
//
// Exposes two globals consumed by the per-example shell:
//   initializeRenderer(canvas, config) — bind to a canvas + visualization config
//   updateVisualization(partitionState) — feed in one partition's latest state
//
// `config` shape:
//   {
//     canvasWidth, canvasHeight, backgroundColor, updateIntervalMs,
//     renderers: [
//       { type, partitionName, properties: {...} },
//       ...
//     ]
//   }
//
// `partitionState` shape (from the wasm OutputFunction):
//   { partitionName, state: { values: number[] }, timesteps }

class GenericRenderer {
    constructor(canvas, config) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.config = config;
        this.state = {};
        this.history = {};
    }

    update(partitionState) {
        this.state[partitionState.partitionName] = partitionState.state.values;

        if (!this.history[partitionState.partitionName]) {
            this.history[partitionState.partitionName] = [];
        }
        this.history[partitionState.partitionName].push({
            value: partitionState.state.values[0] || 0,
            time: partitionState.cumulativeTimesteps || 0
        });
        if (this.history[partitionState.partitionName].length > 100) {
            this.history[partitionState.partitionName].shift();
        }
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.config.renderers.forEach(renderer => {
            this.renderElement(renderer);
        });
    }

    renderElement(renderer) {
        const state = this.state[renderer.partitionName];
        if (!state && renderer.partitionName !== '') return;

        switch (renderer.type) {
            case 'text':         this.renderText(renderer, state); break;
            case 'circle':       this.renderCircle(renderer, state); break;
            case 'rectangle':    this.renderRectangle(renderer, state); break;
            case 'rectangleSet': this.renderRectangleSet(renderer, state); break;
            case 'line':         this.renderLine(renderer, state); break;
            case 'barChart':     this.renderBarChart(renderer, state); break;
            case 'lineChart':    this.renderLineChart(renderer, state); break;
            case 'progressBar':  this.renderProgressBar(renderer, state); break;
            case 'image':        this.renderImage(renderer, state); break;
            case 'playerSet':
            case 'pointSet':     this.renderPointSet(renderer, state); break;
        }
    }

    renderText(renderer, state) {
        // Honor caller-supplied color/font/alignment from the
        // properties object. Defaults match the previous hardcoded
        // behaviour (white, 16px Arial, centered) so existing widgets
        // render unchanged unless they set their own values via the
        // TextOptions emitted by pkg/dashboard's AddText helper.
        const fontSize = renderer.properties.fontSize || 16;
        const fontFamily = renderer.properties.fontFamily || 'Arial';
        this.ctx.fillStyle = renderer.properties.color || '#ffffff';
        this.ctx.font = `${fontSize}px ${fontFamily}`;
        this.ctx.textAlign = renderer.properties.textAlign || 'center';
        let text = renderer.properties.text || '{value}';
        // Guard against undefined/empty state for renderers bound to
        // an empty partitionName (static labels).
        const v = (state && state.length > 0) ? state[0] : 0;
        text = text.replace('{value}', Math.floor(v || 0));
        this.ctx.fillText(text,
            renderer.properties.x || this.canvas.width / 2,
            renderer.properties.y || this.canvas.height / 2);
    }

    renderCircle(renderer, state) {
        const x = renderer.properties.x || this.canvas.width / 2;
        const y = renderer.properties.y || this.canvas.height / 2;
        const radius = renderer.properties.radius || 10;
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
        if (renderer.properties.fillColor) {
            this.ctx.fillStyle = renderer.properties.fillColor;
            this.ctx.fill();
        }
        if (renderer.properties.strokeColor) {
            this.ctx.strokeStyle = renderer.properties.strokeColor;
            this.ctx.lineWidth = renderer.properties.strokeWidth || 1;
            this.ctx.stroke();
        }
        if (!renderer.properties.fillColor && !renderer.properties.strokeColor) {
            this.ctx.fillStyle = renderer.properties.color || '#ffffff';
            this.ctx.fill();
        }
    }

    renderRectangle(renderer, state) {
        const x = renderer.properties.x || 0;
        const y = renderer.properties.y || 0;
        const width = renderer.properties.width || 50;
        const height = renderer.properties.height || 50;
        if (renderer.properties.fillColor) {
            this.ctx.fillStyle = renderer.properties.fillColor;
            this.ctx.fillRect(x, y, width, height);
        }
        if (renderer.properties.strokeColor) {
            this.ctx.strokeStyle = renderer.properties.strokeColor;
            this.ctx.lineWidth = renderer.properties.strokeWidth || 1;
            this.ctx.strokeRect(x, y, width, height);
        }
        if (!renderer.properties.fillColor && !renderer.properties.strokeColor) {
            this.ctx.fillStyle = renderer.properties.color || '#ffffff';
            this.ctx.fillRect(x, y, width, height);
        }
    }

    renderRectangleSet(renderer, state) {
        const defaultWidth = renderer.properties.defaultWidth || 12;
        const defaultHeight = renderer.properties.defaultHeight || 8;
        const fill = renderer.properties.fillColor || renderer.properties.color || '#ffffff';
        const stroke = renderer.properties.strokeColor;
        const strokeWidth = renderer.properties.strokeWidth || 1;
        const topLeftAnchor = renderer.properties.anchor === 'topLeft';

        for (let i = 0; i + 3 < state.length; i += 4) {
            const x = state[i];
            const y = state[i + 1];
            const rawWidth = state[i + 2];
            const rawHeight = state[i + 3];
            const width = Number.isFinite(rawWidth) ? rawWidth : defaultWidth;
            const height = Number.isFinite(rawHeight) ? rawHeight : defaultHeight;
            if (!width || !height || width <= 0 || height <= 0) continue;
            if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

            const drawWidth = Math.abs(width);
            const drawHeight = Math.abs(height);
            const left = topLeftAnchor ? x : x - drawWidth / 2;
            const top = topLeftAnchor ? y : y - drawHeight / 2;

            this.ctx.fillStyle = fill;
            this.ctx.fillRect(left, top, drawWidth, drawHeight);
            if (stroke) {
                this.ctx.strokeStyle = stroke;
                this.ctx.lineWidth = strokeWidth;
                this.ctx.strokeRect(left, top, drawWidth, drawHeight);
            }
        }
    }

    renderLine(renderer, state) {
        const x1 = renderer.properties.x1 || 0;
        const y1 = renderer.properties.y1 || 0;
        const x2 = renderer.properties.x2 || 50;
        const y2 = renderer.properties.y2 || 50;
        this.ctx.beginPath();
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x2, y2);
        this.ctx.strokeStyle = renderer.properties.color || '#ffffff';
        this.ctx.lineWidth = renderer.properties.width || 1;
        this.ctx.stroke();
    }

    renderBarChart(renderer, state) {
        const x = renderer.properties.x || 0;
        const y = renderer.properties.y || 0;
        const width = renderer.properties.width || 50;
        const height = renderer.properties.height || 50;
        const maxValue = renderer.properties.maxValue || 100;
        const value = state[0] || 0;
        const normalizedValue = Math.min(value / maxValue, 1.0);

        this.ctx.fillStyle = renderer.properties.color || 'rgba(255,255,255,0.3)';
        this.ctx.fillRect(x, y, width, height);
        this.ctx.fillStyle = renderer.properties.color || '#4CAF50';
        this.ctx.fillRect(x, y + height * (1 - normalizedValue), width, height * normalizedValue);

        if (renderer.properties.showLabels) {
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '12px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(Math.floor(value), x + width / 2, y + height / 2);
        }
    }

    renderLineChart(renderer, state) {
        const history = this.history[renderer.partitionName];
        if (!history || history.length < 2) return;

        const x = renderer.properties.x || 0;
        const y = renderer.properties.y || 0;
        const width = renderer.properties.width || 50;
        const height = renderer.properties.height || 50;

        let minVal = Infinity, maxVal = -Infinity;
        history.forEach(point => {
            minVal = Math.min(minVal, point.value);
            maxVal = Math.max(maxVal, point.value);
        });
        const range = Math.max(maxVal - minVal, 0.1);

        this.ctx.strokeStyle = renderer.properties.color || '#4CAF50';
        this.ctx.lineWidth = renderer.properties.lineWidth || 2;
        this.ctx.beginPath();
        history.forEach((point, i) => {
            const px = x + (i / (history.length - 1)) * width;
            const py = y + height - ((point.value - minVal) / range) * height;
            if (i === 0) this.ctx.moveTo(px, py);
            else this.ctx.lineTo(px, py);
        });
        this.ctx.stroke();
    }

    renderProgressBar(renderer, state) {
        const x = renderer.properties.x || 0;
        const y = renderer.properties.y || 0;
        const width = renderer.properties.width || 100;
        const height = renderer.properties.height || 20;
        const maxValue = renderer.properties.maxValue || 100;
        const value = Math.max(0, Math.min(state[0] || 0, maxValue));
        const normalizedValue = value / maxValue;

        this.ctx.fillStyle = renderer.properties.backgroundColor || 'rgba(255,255,255,0.3)';
        this.ctx.fillRect(x, y, width, height);
        this.ctx.fillStyle = renderer.properties.foregroundColor || '#4CAF50';
        this.ctx.fillRect(x, y, width * normalizedValue, height);
        if (renderer.properties.borderColor) {
            this.ctx.strokeStyle = renderer.properties.borderColor;
            this.ctx.lineWidth = renderer.properties.borderWidth || 1;
            this.ctx.strokeRect(x, y, width, height);
        }
        if (renderer.properties.showLabel) {
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '12px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(Math.floor(value) + '%', x + width / 2, y + height / 2 + 4);
        }
    }

    renderImage(renderer, state) {
        const imagePath = renderer.properties.imagePath;
        if (!imagePath) return;
        const x = renderer.properties.x || 0;
        const y = renderer.properties.y || 0;
        this.ctx.fillStyle = 'rgba(255,255,255,0.5)';
        this.ctx.fillRect(x, y,
            renderer.properties.width || 32,
            renderer.properties.height || 32);
    }

    renderPointSet(renderer, state) {
        const radius = renderer.properties.radius || 8;
        const fill = renderer.properties.fillColor || renderer.properties.color || '#ffffff';
        const stroke = renderer.properties.strokeColor;
        const strokeWidth = renderer.properties.strokeWidth || 1;

        for (let i = 0; i < state.length; i += 2) {
            const x = state[i];
            const y = state[i + 1];
            if (typeof x !== 'number' || typeof y !== 'number') continue;

            this.ctx.beginPath();
            this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
            this.ctx.fillStyle = fill;
            this.ctx.fill();
            if (stroke) {
                this.ctx.strokeStyle = stroke;
                this.ctx.lineWidth = strokeWidth;
                this.ctx.stroke();
            }
        }
    }
}

// Expose the renderer constructor under a single global namespace so each
// widget on a page can instantiate its own. No module-level singleton —
// multiple widgets must not share renderer state.
self.dexetera = self.dexetera || {};
self.dexetera.createRenderer = function (canvas, config) {
    return new GenericRenderer(canvas, config);
};
