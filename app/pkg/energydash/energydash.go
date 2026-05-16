// Package energydash is the dexetera dashboard for the energy-balancer
// post — "Energy demand response optimisation for the national grid".
// The simulator under the hood is the project's stochadex grid model
// (NESO demand replay → residual demand → imbalance price → dispatch
// policy → battery), running 14 days of half-hour steps at a time over
// a bundled October 2024 slice of NESO open data.
//
// The controls are two discrete selectors (dispatch policy: price vs
// carbon threshold; grid scenario: 2025 current vs 2030 Holistic
// Transition) and a conditional pair of threshold sliders per policy.
// The visualisation has two stacked live line charts (battery state of
// charge over the simulated window, cumulative net value in £k) plus a
// pair of DOM readouts that surface revenue, degradation cost, carbon
// savings, equivalent full cycles, and participation rate.
//
// See app/cmd/energy/{register_step,generate} for the wasm entry-point
// and the codegen that emits the widget shell respectively.
package energydash

import (
	"fmt"

	"github.com/umbralcalc/dexetera/pkg/dashboard"
)

// actionColorHex is the magenta the Acting on Simulated Systems
// collection uses to signal "this is what the reader controls". Kept
// in sync with the recolouring constant in cmd/energy/generate so the
// canvas markers and the HTML radio/slider accents match.
const actionColorHex = "#b0447a"

// referenceColorHex is the slate grey used for static reference
// elements (axis lines, neutral panel dividers). Same hue as the AMR
// reference bars and the flood reference dots so the visual language
// carries across the collection.
const referenceColorHex = "#7d8aa1"

const (
	// Canvas dimensions and panel layout. The canvas natively renders
	// at 640×400 but is CSS-scaled to fit the panel — typically
	// ~410px wide at standard blog embeddings — so font sizes need to
	// be set in canvas-space pixels that stay readable after up to a
	// 2× scale-down.
	CanvasWidth  = 640
	CanvasHeight = 400

	// SoC line chart — top half of the canvas.
	socChartX      = 60
	socChartY      = 50
	socChartWidth  = 520
	socChartHeight = 130

	// Cumulative net value line chart — bottom half.
	valueChartX      = 60
	valueChartY      = 240
	valueChartWidth  = 520
	valueChartHeight = 130

	// Font sizes for canvas labels. 22 reads at ~13px after the
	// standard ~0.6× embed scale-down; 18 lands at ~11px.
	titleFontSize = 22
	axisFontSize  = 18
)

// NewConfig returns the dashboard.Config for the energy-balancer
// widget. Declaration order of renderers matters: later renderers draw
// on top of earlier ones, so static frame elements (panel borders,
// axis lines, text labels) are added first and the partition-bound
// line charts on top.
func NewConfig() *dashboard.Config {
	vb := dashboard.NewVisualizationBuilder().
		WithCanvas(CanvasWidth, CanvasHeight).
		WithBackground("#fafafa").
		WithUpdateInterval(0)

	// --- SoC panel (top half) ---

	vb = vb.AddText("", "Battery state of charge (%)",
		socChartX, socChartY-20,
		&dashboard.TextOptions{
			Color:     "#2c3e50",
			FontSize:  titleFontSize,
			TextAlign: "left",
		})

	// Panel baseline + left edge so the chart sits inside a frame
	// even before the line itself starts populating.
	vb = vb.AddLine("",
		socChartX, socChartY+socChartHeight,
		socChartX+socChartWidth, socChartY+socChartHeight,
		&dashboard.LineOptions{Color: "#2c3e50", Width: 1}).
		AddLine("",
			socChartX, socChartY,
			socChartX, socChartY+socChartHeight,
			&dashboard.LineOptions{Color: "#2c3e50", Width: 1})

	// SoC chart — blue (simulation output), bound to soc_display state[0]
	// (percent). The renderer's rolling 100-sample window happens to be
	// exactly two days of half-hour cadence — enough texture for the
	// reader to see the battery cycling without overplotting.
	vb = vb.AddLineChart("soc_display",
		socChartX, socChartY, socChartWidth, socChartHeight,
		&dashboard.ChartOptions{
			Color:     "#3c78d8",
			LineWidth: 2,
		})

	// --- Net value panel (bottom half) ---

	vb = vb.AddText("", "Cumulative net value (£k)",
		valueChartX, valueChartY-20,
		&dashboard.TextOptions{
			Color:     "#2c3e50",
			FontSize:  titleFontSize,
			TextAlign: "left",
		})

	vb = vb.AddLine("",
		valueChartX, valueChartY+valueChartHeight,
		valueChartX+valueChartWidth, valueChartY+valueChartHeight,
		&dashboard.LineOptions{Color: "#2c3e50", Width: 1}).
		AddLine("",
			valueChartX, valueChartY,
			valueChartX, valueChartY+valueChartHeight,
			&dashboard.LineOptions{Color: "#2c3e50", Width: 1})

	// Cumulative net value line chart — same blue as SoC. Bound to
	// outcomes state[0] (£k); state[1..5] feed the readouts below.
	vb = vb.AddLineChart("outcomes",
		valueChartX, valueChartY, valueChartWidth, valueChartHeight,
		&dashboard.ChartOptions{
			Color:     "#3c78d8",
			LineWidth: 2,
		})

	// --- Static section divider between the two panels. ---
	vb = vb.AddLine("",
		socChartX, valueChartY-30,
		socChartX+socChartWidth, valueChartY-30,
		&dashboard.LineOptions{Color: "#e3e6ec", Width: 1})

	vis := vb.Build()

	cfg := dashboard.NewConfigBuilder("energy").
		WithDescription("Grid battery dispatch policy support: pick a dispatch strategy and grid scenario, tune the trigger thresholds; the simulator (fitted to NESO half-hourly demand data) shows the resulting battery operation, revenue, and carbon savings over a 14-day window. This is a research model fitted to open data, not a trading or operational tool.").
		WithServerPartition("soc_display").
		WithServerPartition("outcomes").
		WithServerPartition("display_progress").
		WithActionStatePartition("policy_action").
		WithVisualization(vis).
		WithSimulation(BuildEnergySimulation)

	// The two top-level discrete sliders are replaced by radio buttons
	// in cmd/energy/generate. They're kept in the data model so the
	// slider→worker action publication mechanism still carries the
	// values to wasm. The labels below are what generate.go uses to
	// find and hide them.
	cfg = cfg.
		WithSlider(dashboard.Slider{
			Name:       "policy",
			Label:      "Policy (radio-controlled)",
			Partition:  "policy_action",
			ValueIndex: PAIdxPolicy,
			Min:        0,
			Max:        NumPolicies - 1,
			Step:       1,
			Default:    PolicyPrice,
			Decimals:   0,
		}).
		WithSlider(dashboard.Slider{
			Name:       "scenario",
			Label:      "Scenario (radio-controlled)",
			Partition:  "policy_action",
			ValueIndex: PAIdxScenario,
			Min:        0,
			Max:        NumScenarios - 1,
			Step:       1,
			Default:    Scenario2025,
			Decimals:   0,
		}).
		// Threshold sliders. cmd/energy/generate wraps each in a
		// conditional block so only the pair relevant to the selected
		// policy is visible — same pattern as AMR's per-policy params.
		WithSlider(dashboard.Slider{
			Name:       "price_high",
			Label:      "Discharge threshold (£/MWh)",
			Partition:  "policy_action",
			ValueIndex: PAIdxPriceHigh,
			Min:        20,
			Max:        100,
			Step:       1,
			Default:    DefaultPriceHigh,
			Decimals:   0,
		}).
		WithSlider(dashboard.Slider{
			Name:       "price_low",
			Label:      "Charge threshold (£/MWh)",
			Partition:  "policy_action",
			ValueIndex: PAIdxPriceLow,
			Min:        0,
			Max:        60,
			Step:       1,
			Default:    DefaultPriceLow,
			Decimals:   0,
		}).
		WithSlider(dashboard.Slider{
			Name:       "carbon_high",
			Label:      "Discharge threshold (gCO₂/kWh)",
			Partition:  "policy_action",
			ValueIndex: PAIdxCarbonHigh,
			Min:        100,
			Max:        400,
			Step:       5,
			Default:    DefaultCarbonHigh,
			Decimals:   0,
		}).
		WithSlider(dashboard.Slider{
			Name:       "carbon_low",
			Label:      "Charge threshold (gCO₂/kWh)",
			Partition:  "policy_action",
			ValueIndex: PAIdxCarbonLow,
			Min:        0,
			Max:        250,
			Step:       5,
			Default:    DefaultCarbonLow,
			Decimals:   0,
		})

	// Readouts. The first surfaces simulation progress + live SoC; the
	// second the headline cumulative outcomes once a run has had time
	// to accumulate. Templates use {vN} tokens — N indexes the
	// partition's state vector, see iteration.go for the layout.
	cfg = cfg.
		WithReadout(dashboard.Readout{
			Partition: "display_progress",
			Template:  fmt.Sprintf("hour {v%d} of %d · SoC {v%d}%%", 0, int(float64(SimSteps)*Stepsize), 1),
			Decimals:  1,
		}).
		WithReadout(dashboard.Readout{
			Partition: "outcomes",
			Template: fmt.Sprintf(
				"net £{v%d}k · revenue £{v%d}k · degradation £{v%d}k · carbon {v%d} tCO₂ · {v%d} EFC · {v%d} active",
				0, 1, 2, 3, 4, 5,
			),
			Decimals: 1,
		}).
		WithResetButton().
		WithInlineDriver(15)

	return cfg.Build()
}
