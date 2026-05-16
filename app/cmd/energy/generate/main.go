// generate emits the energy-balancer widget shell (widget.html,
// test.html, build.sh) into app/energy/. Re-run whenever the
// dashboard.Config in pkg/energydash changes shape (controls,
// partitions, visualisation).
//
//	cd app && go run ./cmd/energy/generate
//
// After codegen the emitted HTML is post-processed to:
//   - recolour the slider accent + readout text in the explainer-series'
//     magenta so the controls read as "what the reader does"
//   - replace the dexetera-emitted policy and scenario sliders with
//     two rows of radio buttons, so the categorical choices get
//     categorical controls
//   - wrap each pair of threshold sliders in a div that's only visible
//     when its corresponding policy is selected, so the reader only
//     sees the thresholds that apply to their current choice — same
//     pattern as the AMR widget's per-policy parameters
//   - patch the dexetera-emitted IIFE so the worker is terminated and
//     a status message is posted once the SimSteps horizon is reached
//     (otherwise the inline driver ticks forever even after iterations
//     freeze)
//   - re-publish slider values on the 'inline driver ready' status
//     message so the very-first action message lands after the worker
//     has its driver loaded
//   - load the worker via a same-origin Blob URL so cross-origin CDN
//     hosting works
//   - wire the radio buttons to the hidden sliders + the Reset button
//     so a policy/scenario change restarts the simulation
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/umbralcalc/dexetera/pkg/dashboard"
	"github.com/umbralcalc/energy-balancer/app/pkg/energydash"
)

// actionColor is the magenta from the Acting on Simulated Systems
// collection — used to signal "this is what the reader controls".
// Replaces dexetera's default blue (#3c78d8) on the slider track and
// readout text. Kept in sync with energydash.actionColorHex.
const actionColor = "#b0447a"

// policyChoices lists the radio buttons that replace the policy
// slider. Indices match the energydash.Policy* constants. ThresholdGroup
// names the conditional-slider wrapper that should be visible when
// this policy is selected.
var policyChoices = []struct {
	Value          int
	Label          string
	ThresholdGroup string
}{
	{energydash.PolicyPrice, "Price threshold (£ arbitrage)", "price"},
	{energydash.PolicyCarbon, "Carbon threshold (gCO₂ signal)", "carbon"},
}

// scenarioChoices lists the radio buttons that replace the scenario
// slider. Indices match the energydash.Scenario* constants.
var scenarioChoices = []struct {
	Value int
	Label string
}{
	{energydash.Scenario2025, "2025 grid (current renewables)"},
	{energydash.Scenario2030, "2030 grid (Holistic Transition)"},
}

// thresholdGroups maps the policy-group key from policyChoices to the
// pair of slider names that should be visible for it. Each group's
// sliders are wrapped in a single .policy-conditional div by
// wrapConditionalThresholds.
var thresholdGroups = map[string][]string{
	"price":  {"price_high", "price_low"},
	"carbon": {"carbon_high", "carbon_low"},
}

func main() {
	runtimeURL := flag.String("runtime-url", "",
		"absolute URL the blog will serve dexetera's runtime/ folder from "+
			"(e.g. https://example.com/assets/dexetera/runtime/). "+
			"Leave empty for local preview via test.html.")
	wasmURL := flag.String("wasm-url", "",
		"absolute URL the blog will serve main.wasm from. "+
			"Leave empty for local preview.")
	flag.Parse()

	cfg := energydash.NewConfig()
	dashboard.MustGenerateWidget(cfg, dashboard.WidgetOptions{
		RuntimeBaseURL: *runtimeURL,
		WasmURL:        *wasmURL,
	})

	for _, name := range []string{"widget.html", "test.html"} {
		path := filepath.Join(cfg.Name, name)
		if err := postProcess(path); err != nil {
			fmt.Fprintf(os.Stderr, "post-process %s: %v\n", path, err)
			os.Exit(1)
		}
	}
}

func postProcess(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	out := string(data)
	widgetID := extractWidgetID(out)

	for _, step := range []func(string, string) (string, error){
		recolorControls,
		injectScopedStyles,
		replaceSliderWithRadios("policy", "energy-policy", "Dispatch policy", policyChoiceLabels()),
		replaceSliderWithRadios("scenario", "energy-scenario", "Grid scenario", scenarioChoiceLabels()),
		wrapConditionalThresholds,
		fixIntegerReadoutDecimals,
		injectTerminationHalt,
		injectActionResend,
		injectCrossOriginWorkerShim,
		injectControlScript,
	} {
		out, err = step(out, widgetID)
		if err != nil {
			return err
		}
	}
	return os.WriteFile(path, []byte(out), 0644)
}

type choiceLabel struct {
	Value int
	Label string
}

func policyChoiceLabels() []choiceLabel {
	out := make([]choiceLabel, len(policyChoices))
	for i, c := range policyChoices {
		out[i] = choiceLabel{Value: c.Value, Label: c.Label}
	}
	return out
}

func scenarioChoiceLabels() []choiceLabel {
	out := make([]choiceLabel, len(scenarioChoices))
	for i, c := range scenarioChoices {
		out[i] = choiceLabel{Value: c.Value, Label: c.Label}
	}
	return out
}

// recolorControls swaps dexetera's default blue on the slider accent
// and readout for the action-colour magenta. Anchored on enough
// surrounding CSS text to avoid touching unrelated occurrences of the
// colour.
func recolorControls(html, _ string) (string, error) {
	pairs := [][2]string{
		{"accent-color: #3c78d8", "accent-color: " + actionColor},
		{".slider-readout { grid-area: readout; text-align: right; color: #3c78d8;",
			".slider-readout { grid-area: readout; text-align: right; color: " + actionColor + ";"},
	}
	return applyPairs(html, pairs)
}

// injectScopedStyles appends CSS rules for the radio-button rows
// (policy and scenario selectors) and the conditional-threshold
// wrapper. All rules are prefixed with #<widgetID> so they don't leak
// out of the widget shell.
func injectScopedStyles(html, widgetID string) (string, error) {
	const marker = `</style>`
	extra := strings.ReplaceAll(scopedStylesTemplate, "{{.WidgetID}}", widgetID)
	if !strings.Contains(html, marker) {
		return "", fmt.Errorf("</style> marker not found")
	}
	return strings.Replace(html, marker, extra+marker, 1), nil
}

const scopedStylesTemplate = `#{{.WidgetID}} .energy-selector { display: flex; flex-direction: column; gap: 0.4em; font-size: 1rem; margin-bottom: 0.6em; }` +
	`#{{.WidgetID}} .energy-selector-label { color: #2c3e50; font-weight: 600; }` +
	`#{{.WidgetID}} .energy-options { display: flex; flex-direction: column; gap: 0.3em; }` +
	`#{{.WidgetID}} .energy-options label { display: flex; align-items: center; gap: 0.4em; color: #2c3e50; cursor: pointer; }` +
	`#{{.WidgetID}} .energy-options input[type="radio"] { accent-color: ` + actionColor + `; }` +
	`#{{.WidgetID}} .policy-conditional { display: none; }` +
	`#{{.WidgetID}} .policy-conditional.is-active { display: block; }` +
	`#{{.WidgetID}} .policy-conditional-title { color: #2c3e50; font-weight: 600; font-size: 0.95rem; margin: 0.3em 0 0.2em; }`

// replaceSliderWithRadios returns a post-process step that rewrites the
// dexetera-emitted slider for the given sliderName into a radio-button
// group with the given heading. Same pattern as the flood widget — the
// slider input itself is kept (display:none) so dexetera's
// slider→worker publish mechanism still picks it up; our injected JS
// keeps the hidden slider's value in sync with the selected radio.
func replaceSliderWithRadios(sliderName, radioName, heading string, choices []choiceLabel) func(string, string) (string, error) {
	return func(html, _ string) (string, error) {
		startTag := `<label class="slider">`
		endTag := `</label>`
		dataAttr := `data-slider="` + sliderName + `"`

		idx := strings.Index(html, dataAttr)
		if idx == -1 {
			return "", fmt.Errorf("%s slider not found", sliderName)
		}
		start := strings.LastIndex(html[:idx], startTag)
		if start == -1 {
			return "", fmt.Errorf("%s slider <label> not found", sliderName)
		}
		end := strings.Index(html[idx:], endTag)
		if end == -1 {
			return "", fmt.Errorf("%s slider </label> not found", sliderName)
		}
		end += idx + len(endTag)

		var b strings.Builder
		b.WriteString(`<div class="energy-selector">`)
		fmt.Fprintf(&b, `<span class="energy-selector-label">%s</span>`, heading)
		b.WriteString(`<div class="energy-options">`)
		for _, c := range choices {
			checked := ""
			if c.Value == 0 {
				checked = " checked"
			}
			fmt.Fprintf(&b,
				`<label><input type="radio" name="%s" value="%d"%s>%s</label>`,
				radioName, c.Value, checked, c.Label,
			)
		}
		b.WriteString(`</div>`)
		fmt.Fprintf(&b,
			`<input type="range" data-slider="%s" min="0" max="%d" step="1" value="0" style="display:none">`,
			sliderName, len(choices)-1,
		)
		fmt.Fprintf(&b, `<span data-slider-readout="%s" style="display:none"></span>`, sliderName)
		b.WriteString(`</div>`)

		return html[:start] + b.String() + html[end:], nil
	}
}

// wrapConditionalThresholds wraps each pair of threshold sliders in a
// .policy-conditional div whose visibility is gated on the currently-
// selected policy. injectControlScript toggles the .is-active class
// when a radio button is clicked.
//
// Each group also gets a small DOM heading ("Price-threshold settings"
// or "Carbon-threshold settings") so the reader sees clearly which
// units are in play; this is the mitigation noted in the plan against
// the conditional-slider-confusion risk.
func wrapConditionalThresholds(html, _ string) (string, error) {
	type group struct {
		Key     string
		Policy  int
		Title   string
		Sliders []string
	}
	groups := []group{
		{Key: "price", Policy: energydash.PolicyPrice, Title: "Price-threshold settings", Sliders: thresholdGroups["price"]},
		{Key: "carbon", Policy: energydash.PolicyCarbon, Title: "Carbon-threshold settings", Sliders: thresholdGroups["carbon"]},
	}

	for _, g := range groups {
		startTag := `<label class="slider">`
		first := strings.Index(html, `data-slider="`+g.Sliders[0]+`"`)
		if first == -1 {
			return "", fmt.Errorf("threshold slider %q not found", g.Sliders[0])
		}
		groupStart := strings.LastIndex(html[:first], startTag)
		if groupStart == -1 {
			return "", fmt.Errorf("threshold slider %q <label> not found", g.Sliders[0])
		}
		last := strings.Index(html, `data-slider="`+g.Sliders[len(g.Sliders)-1]+`"`)
		if last == -1 {
			return "", fmt.Errorf("threshold slider %q not found", g.Sliders[len(g.Sliders)-1])
		}
		endRel := strings.Index(html[last:], `</label>`)
		if endRel == -1 {
			return "", fmt.Errorf("threshold slider %q </label> not found", g.Sliders[len(g.Sliders)-1])
		}
		groupEnd := last + endRel + len(`</label>`)

		wrapper := fmt.Sprintf(
			`<div class="policy-conditional" data-policy="%d"><p class="policy-conditional-title">%s</p>%s</div>`,
			g.Policy, g.Title, html[groupStart:groupEnd],
		)
		html = html[:groupStart] + wrapper + html[groupEnd:]
	}
	return html, nil
}

// fixIntegerReadoutDecimals rewrites the marshalled gameConfig JSON so
// the readouts we intend to format as integers (hour count, carbon
// tonnes, EFC count) render at zero decimal places where appropriate.
// dexetera's marshalGameConfig substitutes any dashboard.Readout.Decimals
// == 0 with its default of 2 — so we set Decimals=1 in NewConfig and
// patch here only when we want a tighter format.
func fixIntegerReadoutDecimals(html, _ string) (string, error) {
	// No-op for now — the default Decimals=1 from energydash.NewConfig
	// produces readable output across all surfaced metrics. Kept as a
	// hook for future refinement (e.g. integer EFCs).
	return html, nil
}

// injectTerminationHalt patches the dexetera-emitted IIFE so that when
// a partition state arrives with cumulativeTimesteps >= SimSteps + 1
// (one extra step to flush the final stats through the readouts), the
// worker is terminated and a status message is posted. Without this
// the inline driver ticks forever — even with frozen iterations, the
// renderer keeps re-drawing identical state every 30 ms.
func injectTerminationHalt(html, _ string) (string, error) {
	const oldTail = `if (el) el.textContent = applyReadout(r.template, r.decimals, msg.data);
                }
            } else if (msg.type === 'status') {`
	totalDays := int(float64(energydash.SimSteps) * energydash.Stepsize / 24.0)
	newTail := fmt.Sprintf(`if (el) el.textContent = applyReadout(r.template, r.decimals, msg.data);
                }
                if (worker && msg.data.timesteps >= %d) { worker.terminate(); worker = null; setStatus('%d-day simulation complete. Use Reset to rerun.'); }
            } else if (msg.type === 'status') {`,
		energydash.SimSteps, totalDays)
	if !strings.Contains(html, oldTail) {
		return "", fmt.Errorf("partitionState block anchor not found for termination halt")
	}
	return strings.Replace(html, oldTail, newTail, 1), nil
}

// injectActionResend patches the dexetera-emitted IIFE so that the page
// re-publishes the current slider values once the driver reports that
// it's ready. Same rationale as in the AMR and flood widgets: the
// initial setActions message races the worker's driver-load, so the
// first step (or several) can pick up stale defaults on Reset.
func injectActionResend(html, _ string) (string, error) {
	const oldStatus = `} else if (msg.type === 'status') {
                setStatus(msg.data);`
	const newStatus = `} else if (msg.type === 'status') {
                setStatus(msg.data);
                if (msg.data === 'inline driver ready') publishActions();`
	if !strings.Contains(html, oldStatus) {
		return "", fmt.Errorf("status handler anchor not found for action resend")
	}
	return strings.Replace(html, oldStatus, newStatus, 1), nil
}

// injectCrossOriginWorkerShim wraps the dexetera-emitted worker
// creation so the worker.js script can be loaded from a different
// origin (e.g. the blog's R2 CDN) while the page itself is served
// from GitHub Pages. Mirrors the pattern used by AMR, flood, and
// rugby; lift this into dexetera proper once it stabilises.
func injectCrossOriginWorkerShim(html, _ string) (string, error) {
	const oldNewWorker = `worker = new Worker(RUNTIME_BASE + 'worker.js');`
	const newNewWorker = `ensureWorkerUrl().then(function (workerUrl) {
        worker = new Worker(workerUrl);`
	if !strings.Contains(html, oldNewWorker) {
		return "", fmt.Errorf("worker creation anchor not found for cross-origin shim")
	}
	html = strings.Replace(html, oldNewWorker, newNewWorker, 1)

	const oldEnd = `        publishActions();
    }

    ensureRenderer().then(function () {`
	const newEnd = `        publishActions();
        }).catch(function (err) {
            console.error(err);
            setStatus('Failed to load dexetera worker: ' + err.message);
        });
    }

    ensureRenderer().then(function () {`
	if !strings.Contains(html, oldEnd) {
		return "", fmt.Errorf("startWorker tail anchor not found for cross-origin shim")
	}
	html = strings.Replace(html, oldEnd, newEnd, 1)

	const startWorkerSig = `function startWorker(renderer) {`
	const ensureWorkerUrlFn = `function ensureWorkerUrl() {
        if (self.__dexeteraWorkerUrl) return Promise.resolve(self.__dexeteraWorkerUrl);
        if (self.__dexeteraWorkerLoading) return self.__dexeteraWorkerLoading;
        var BASE_ABS = new URL(RUNTIME_BASE, document.baseURI).href;
        self.__dexeteraWorkerLoading = fetch(BASE_ABS + 'worker.js')
            .then(function (r) {
                if (!r.ok) throw new Error('failed to fetch worker.js: ' + r.status);
                return r.text();
            })
            .then(function (src) {
                var shim = '(function(){var BASE=' + JSON.stringify(BASE_ABS) +
                    ';var orig=self.importScripts;self.importScripts=function(){' +
                    'var args=Array.prototype.map.call(arguments,function(u){' +
                    'return new URL(u,BASE).href;});return orig.apply(self,args);};})();\n';
                var blob = new Blob([shim, src], { type: 'application/javascript' });
                self.__dexeteraWorkerUrl = URL.createObjectURL(blob);
                return self.__dexeteraWorkerUrl;
            });
        return self.__dexeteraWorkerLoading;
    }

    `
	if !strings.Contains(html, startWorkerSig) {
		return "", fmt.Errorf("startWorker signature not found for cross-origin shim")
	}
	html = strings.Replace(html, startWorkerSig, ensureWorkerUrlFn+startWorkerSig, 1)
	return html, nil
}

// injectControlScript appends an IIFE that wires the radio buttons
// (policy and scenario groups) to their hidden sliders, toggles the
// conditional-threshold wrappers, and clicks the Reset button on every
// discrete-choice change so the simulation restarts from t=0.
func injectControlScript(html, widgetID string) (string, error) {
	script := strings.ReplaceAll(controlScriptTemplate, "{{.WidgetID}}", widgetID)
	return html + script, nil
}

const controlScriptTemplate = `
<script>
(function () {
    var widget = document.getElementById('{{.WidgetID}}');
    if (!widget) return;
    var resetBtn = widget.querySelector('[data-reset]');

    function wireGroup(radioName, sliderName, onChange) {
        var radios = widget.querySelectorAll('input[name="' + radioName + '"]');
        var slider = widget.querySelector('[data-slider="' + sliderName + '"]');
        function applyValue(value) {
            if (slider) {
                slider.value = String(value);
                slider.dispatchEvent(new Event('input', { bubbles: true }));
            }
            if (onChange) onChange(value);
        }
        for (var i = 0; i < radios.length; i++) {
            radios[i].addEventListener('change', function (e) {
                applyValue(parseInt(e.target.value, 10));
                if (resetBtn) resetBtn.click();
            });
        }
        var initial = 0;
        for (var j = 0; j < radios.length; j++) {
            if (radios[j].checked) { initial = parseInt(radios[j].value, 10); break; }
        }
        // Initial state — sync slider + apply onChange but do NOT click
        // Reset (the dexetera IIFE handles the very first sim start;
        // clicking Reset on init would race the renderer load).
        applyValue(initial);
    }

    function applyPolicyConditionals(value) {
        var conditionals = widget.querySelectorAll('.policy-conditional');
        for (var i = 0; i < conditionals.length; i++) {
            var c = conditionals[i];
            if (c.getAttribute('data-policy') === String(value)) {
                c.classList.add('is-active');
            } else {
                c.classList.remove('is-active');
            }
        }
    }

    wireGroup('energy-policy', 'policy', applyPolicyConditionals);
    wireGroup('energy-scenario', 'scenario', null);
})();
</script>
`

func applyPairs(html string, pairs [][2]string) (string, error) {
	for _, p := range pairs {
		if !strings.Contains(html, p[0]) {
			return "", fmt.Errorf("expected fragment not found: %q", p[0])
		}
		html = strings.Replace(html, p[0], p[1], 1)
	}
	return html, nil
}

// extractWidgetID picks the widget root's id out of the generated HTML
// so the styles and script we inject can scope to the same element as
// the rest of the dexetera CSS.
func extractWidgetID(html string) string {
	const marker = `id="`
	i := strings.Index(html, marker)
	if i < 0 {
		return "dexetera"
	}
	i += len(marker)
	end := strings.Index(html[i:], `"`)
	if end < 0 {
		return "dexetera"
	}
	return html[i : i+end]
}
