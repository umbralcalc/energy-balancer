package energydash

import (
	_ "embed"
	"encoding/csv"
	"math"
	"strconv"
	"strings"

	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// SimSteps is the number of half-hour outer steps the dashboard runs
// before halting — 672 = 14 days. The inline driver ticks once per
// outer step, so SimSteps × the driver interval (~15 ms) sets the total
// wall-clock simulation time. Chosen so the SoC trace shows ~2 weeks of
// real dispatch dynamics (enough for the reader to see the battery
// breathe at half-hour cadence under each policy) while keeping the run
// under ~10 seconds on every reset.
const SimSteps = 672

// Stepsize is the per-step time increment in hours. 0.5 = half-hourly,
// matching NESO settlement-period cadence and the offline cmd/simulate
// configuration.
const Stepsize = 0.5

// Action vector layout. The slider/radio panel writes to
// action_state_values in this order; PolicyActionIteration latches it
// onto state and adds derived wind/solar scale factors.
const (
	PAIdxPolicy     = 0
	PAIdxScenario   = 1
	PAIdxPriceHigh  = 2
	PAIdxPriceLow   = 3
	PAIdxCarbonHigh = 4
	PAIdxCarbonLow  = 5
	PAIdxWindScale  = 6
	PAIdxSolarScale = 7
	PolicyActionLen = 8
)

// Policy and scenario indices. Match the project's two named policies
// (price-threshold, carbon-threshold) and two named grid scenarios
// (2025 current, 2030 Holistic Transition).
const (
	PolicyPrice  = 0
	PolicyCarbon = 1
	NumPolicies  = 2

	Scenario2025 = 0
	Scenario2030 = 1
	NumScenarios = 2
)

// Wind and solar capacity scale factors per scenario. 2030 numbers
// match the project's NESO Holistic Transition defaults.
var (
	WindScales  = []float64{1.0, 2.1}
	SolarScales = []float64{1.0, 2.0}
)

// Default threshold values — same as the project's cmd/simulate flags.
const (
	DefaultPriceHigh  = 45.0
	DefaultPriceLow   = 25.0
	DefaultCarbonHigh = 250.0
	DefaultCarbonLow  = 100.0
)

// Battery hardware defaults — same as the project's cmd/simulate flags.
const (
	CapacityMWh      = 200.0
	PowerRatingMW    = 100.0
	ChargeEfficiency = 0.92
	MinSoCFraction   = 0.1
	MaxSoCFraction   = 0.9
)

// CostPerEFC is the assumed degradation cost in £ per equivalent full
// cycle. Matches the project's default and is the multiplier behind
// the net-value-vs-revenue distinction the post hangs on.
const CostPerEFC = 8000.0

// PriceSlope and PriceIntercept reproduce the project's structural
// price model: price = slope × residual_demand + intercept + noise.
const (
	PriceSlope     = 0.002
	PriceIntercept = -10.0
)

// Price-noise OU parameters — same as the project's cmd/simulate.
const (
	PriceNoiseTheta = 2.0
	PriceNoiseSigma = 5.0
)

//go:embed data/oct2024.csv
var embeddedDataCSV string

// gridData holds the parsed [national_demand_mw, embedded_wind_mw,
// embedded_solar_mw, carbon_intensity_gco2_kwh] series. Populated once
// per process by parseEmbeddedData on first Configure.
var (
	embeddedDemand [][3]float64
	embeddedCarbon []float64
)

func parseEmbeddedData() {
	if len(embeddedDemand) > 0 {
		return
	}
	r := csv.NewReader(strings.NewReader(embeddedDataCSV))
	records, err := r.ReadAll()
	if err != nil {
		panic("energydash: parse embedded data: " + err.Error())
	}
	embeddedDemand = make([][3]float64, len(records))
	embeddedCarbon = make([]float64, len(records))
	for i, row := range records {
		nd, _ := strconv.ParseFloat(strings.TrimSpace(row[0]), 64)
		wind, _ := strconv.ParseFloat(strings.TrimSpace(row[1]), 64)
		solar, _ := strconv.ParseFloat(strings.TrimSpace(row[2]), 64)
		carbon, _ := strconv.ParseFloat(strings.TrimSpace(row[3]), 64)
		embeddedDemand[i] = [3]float64{nd, wind, solar}
		embeddedCarbon[i] = carbon
	}
}

// terminated reports whether the simulation has reached SimSteps. Once
// hit, custom iterations freeze their state so radio/slider changes
// after completion don't silently rewind the displayed values — the
// reader uses Reset to rerun.
func terminated(timestepsHistory *simulator.CumulativeTimestepsHistory) bool {
	return timestepsHistory.Values.AtVec(0) >= float64(SimSteps)
}

// PolicyActionIteration is the slider/radio-driven action partition.
// It echoes the most recent action_state_values vector as state, then
// computes the wind_scale / solar_scale derived from the scenario
// index so downstream partitions can read them from the same upstream
// state vector.
//
// State width: PolicyActionLen.
type PolicyActionIteration struct{}

func (p *PolicyActionIteration) Configure(int, *simulator.Settings) {}

func (p *PolicyActionIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	if terminated(timestepsHistory) {
		return stateHistories[partitionIndex].CopyStateRow(0)
	}
	out := make([]float64, PolicyActionLen)
	if actions, ok := params.GetOk("action_state_values"); ok {
		for i := 0; i < PolicyActionLen && i < len(actions); i++ {
			out[i] = actions[i]
		}
	} else {
		prev := stateHistories[partitionIndex].CopyStateRow(0)
		copy(out, prev[:PolicyActionLen])
	}
	scenario := clampInt(int(math.Round(out[PAIdxScenario])), 0, NumScenarios-1)
	out[PAIdxWindScale] = WindScales[scenario]
	out[PAIdxSolarScale] = SolarScales[scenario]
	return out
}

// EmbeddedGridDataIteration replays the bundled NESO half-hourly demand
// slice (ND, embedded wind, embedded solar) so the wasm build can run
// the same residual-demand calculation as the offline simulator without
// reading a CSV at runtime.
//
// State: [national_demand_mw, embedded_wind_mw, embedded_solar_mw].
type EmbeddedGridDataIteration struct{}

func (e *EmbeddedGridDataIteration) Configure(int, *simulator.Settings) {
	parseEmbeddedData()
}

func (e *EmbeddedGridDataIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	step := timestepsHistory.CurrentStepNumber
	if step < 0 {
		step = 0
	}
	if step >= len(embeddedDemand) {
		step = len(embeddedDemand) - 1
	}
	row := embeddedDemand[step]
	return []float64{row[0], row[1], row[2]}
}

// EmbeddedCarbonDataIteration replays the bundled half-hourly carbon
// intensity slice, exposing only the "actual" channel (the project's
// upstream CarbonDataIteration also exposes a forecast channel that
// neither dispatch policy reads).
//
// State: [actual_gco2_kwh].
type EmbeddedCarbonDataIteration struct{}

func (e *EmbeddedCarbonDataIteration) Configure(int, *simulator.Settings) {
	parseEmbeddedData()
}

func (e *EmbeddedCarbonDataIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	step := timestepsHistory.CurrentStepNumber
	if step < 0 {
		step = 0
	}
	if step >= len(embeddedCarbon) {
		step = len(embeddedCarbon) - 1
	}
	return []float64{embeddedCarbon[step]}
}

// SwitchingDispatchIteration emits the battery's dispatch signal under
// whichever of the two threshold policies is selected. The discrete
// policy choice lives in the upstream policy_action partition; the
// reader's threshold sliders feed the same partition.
//
// This collapses the project's two separate PriceThresholdDispatch and
// CarbonThresholdDispatch iterations into one, so the dashboard graph
// stays the same shape across both policy modes.
//
// State: [dispatch_mw].
type SwitchingDispatchIteration struct{}

func (s *SwitchingDispatchIteration) Configure(int, *simulator.Settings) {}

func (s *SwitchingDispatchIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	action := params.Get("policy_action")
	policyIdx := clampInt(int(math.Round(action[PAIdxPolicy])), 0, NumPolicies-1)
	rating := params.GetIndex("power_rating_mw", 0)

	switch policyIdx {
	case PolicyCarbon:
		carbonIdx := int(params.GetIndex("carbon_partition", 0))
		carbon := stateHistories[carbonIdx].Values.At(0, 0)
		high := action[PAIdxCarbonHigh]
		low := action[PAIdxCarbonLow]
		switch {
		case carbon > high:
			return []float64{rating}
		case carbon < low:
			return []float64{-rating}
		default:
			return []float64{0}
		}
	default:
		priceIdx := int(params.GetIndex("price_partition", 0))
		price := stateHistories[priceIdx].Values.At(0, 0)
		high := action[PAIdxPriceHigh]
		low := action[PAIdxPriceLow]
		switch {
		case price > high:
			return []float64{rating}
		case price < low:
			return []float64{-rating}
		default:
			return []float64{0}
		}
	}
}

// SoCDisplayIteration projects the battery's state-of-charge into a
// percentage (0–100) that AddLineChart can pick up from state[0]. The
// raw battery state is in MWh; the line chart auto-scales but reads
// more clearly with a percentage on the y-axis.
//
// State: [soc_percent, soc_mwh].
type SoCDisplayIteration struct{}

func (s *SoCDisplayIteration) Configure(int, *simulator.Settings) {}

func (s *SoCDisplayIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	batteryIdx := int(params.GetIndex("battery_partition", 0))
	capacity := params.GetIndex("energy_capacity_mwh", 0)
	socMWh := stateHistories[batteryIdx].Values.At(0, 0)
	pct := 0.0
	if capacity > 0 {
		pct = 100.0 * socMWh / capacity
	}
	return []float64{pct, socMWh}
}

// OutcomesIteration accumulates the dashboard's headline outcome
// metrics. State[0] is cumulative net value in £k so AddLineChart on
// this partition picks up the net-value trace; the other state slots
// carry the readout-visible breakdown.
//
// State layout:
//
//	0: cumulative net value (£k)            — revenue − £CostPerEFC × EFC
//	1: cumulative gross revenue (£k)
//	2: cumulative degradation cost (£k)
//	3: cumulative carbon savings (tCO₂)
//	4: cumulative equivalent full cycles (EFC)
//	5: fraction of periods active (0–1)
//	6: dispatched-period count (raw; for state[5])
//
// One-step lag — reads the previous step's battery, price, and carbon
// states (same convention as the project's RevenueIteration /
// CarbonSavingsIteration / BatteryDegradationIteration).
type OutcomesIteration struct{}

func (o *OutcomesIteration) Configure(int, *simulator.Settings) {}

func (o *OutcomesIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	prev := stateHistories[partitionIndex].CopyStateRow(0)
	if terminated(timestepsHistory) {
		return prev
	}

	batteryIdx := int(params.GetIndex("battery_partition", 0))
	priceIdx := int(params.GetIndex("price_partition", 0))
	carbonIdx := int(params.GetIndex("carbon_partition", 0))
	capacity := params.GetIndex("energy_capacity_mwh", 0)
	dt := timestepsHistory.NextIncrement

	actualDispatch := stateHistories[batteryIdx].Values.At(0, 1)
	price := stateHistories[priceIdx].Values.At(0, 0)
	carbon := stateHistories[carbonIdx].Values.At(0, 0)

	revenueGBP := actualDispatch * price * dt
	efc := math.Abs(actualDispatch*dt) / (2.0 * capacity)
	dischargeMWh := math.Max(actualDispatch, 0) * dt
	carbonTonnes := dischargeMWh * carbon / 1000.0
	active := 0.0
	if actualDispatch != 0 {
		active = 1.0
	}

	cumRevenue := prev[1] + revenueGBP/1000.0       // store in £k
	cumEFC := prev[4] + efc
	cumDegCost := prev[2] + (efc*CostPerEFC)/1000.0 // £k
	cumNetValue := cumRevenue - cumDegCost
	cumCarbon := prev[3] + carbonTonnes
	dispatchedCount := prev[6] + active

	step := timestepsHistory.Values.AtVec(0) + 1
	fracActive := 0.0
	if step > 0 {
		fracActive = dispatchedCount / step
	}

	return []float64{
		cumNetValue,
		cumRevenue,
		cumDegCost,
		cumCarbon,
		cumEFC,
		fracActive,
		dispatchedCount,
	}
}

// DisplayProgressIteration surfaces the simulation's wall-clock-ish
// progress and the live SoC for the inline readout, anchoring the
// reader's eye while the line charts scroll.
//
// State: [hours_elapsed, soc_percent].
type DisplayProgressIteration struct{}

func (d *DisplayProgressIteration) Configure(int, *simulator.Settings) {}

func (d *DisplayProgressIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	if terminated(timestepsHistory) {
		return stateHistories[partitionIndex].CopyStateRow(0)
	}
	socIdx := int(params.GetIndex("soc_partition", 0))
	socPct := stateHistories[socIdx].Values.At(0, 0)
	hours := timestepsHistory.Values.AtVec(0) * Stepsize
	return []float64{hours, socPct}
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
