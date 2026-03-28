package grid

import (
	"math"

	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// RevenueIteration accumulates the revenue earned by battery dispatch against
// the imbalance price. Revenue is positive when discharging (selling) and
// negative when charging (buying).
//
// Revenue per step (£) = actual_dispatch_mw × price_gbp_per_mwh × dt
//
// NOTE: one-step lag — reads the previous step's battery and price states.
//
// Params:
//
//	battery_partition [index] - partition index of the battery state
//	price_partition   [index] - partition index of the imbalance price state
//
// State: [cumulative_revenue_gbp]
type RevenueIteration struct{}

func (r *RevenueIteration) Configure(
	partitionIndex int,
	settings *simulator.Settings,
) {
}

func (r *RevenueIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	batteryIdx := int(params.Map["battery_partition"][0])
	priceIdx := int(params.Map["price_partition"][0])
	dt := timestepsHistory.NextIncrement

	actualDispatch := stateHistories[batteryIdx].Values.At(0, 1)  // state[1]
	price := stateHistories[priceIdx].Values.At(0, 0)
	prevRevenue := stateHistories[partitionIndex].Values.At(0, 0)

	return []float64{prevRevenue + actualDispatch*price*dt}
}

// CarbonSavingsIteration accumulates the carbon displaced by battery
// discharging. When the battery discharges it reduces demand on dispatchable
// (typically gas) generation, saving carbon proportional to the current grid
// carbon intensity.
//
// Carbon saved per step (tCO₂) = max(actual_dispatch, 0) × dt × actual_gco2_kwh / 1000
//   - actual_dispatch in MW, dt in hours, actual_gco2_kwh in gCO₂/kWh
//   - Discharge MWh × 1000 kWh/MWh × gCO₂/kWh = gCO₂; divide by 1e6 → tCO₂
//   - Simplified: dispatch_mw × dt × actual_gco2_kwh / 1000
//
// NOTE: one-step lag — reads the previous step's battery and carbon states.
//
// Params:
//
//	battery_partition [index] - partition index of the battery state
//	carbon_partition  [index] - partition index of the carbon data state
//
// State: [cumulative_carbon_tco2]
type CarbonSavingsIteration struct{}

func (c *CarbonSavingsIteration) Configure(
	partitionIndex int,
	settings *simulator.Settings,
) {
}

func (c *CarbonSavingsIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	batteryIdx := int(params.Map["battery_partition"][0])
	carbonIdx := int(params.Map["carbon_partition"][0])
	dt := timestepsHistory.NextIncrement

	actualDispatch := stateHistories[batteryIdx].Values.At(0, 1)         // state[1]
	actualGco2Kwh := stateHistories[carbonIdx].Values.At(0, 0)           // state[0]
	prevCarbon := stateHistories[partitionIndex].Values.At(0, 0)

	dischargeMwh := math.Max(actualDispatch, 0) * dt
	savedTco2 := dischargeMwh * actualGco2Kwh / 1000.0

	return []float64{prevCarbon + savedTco2}
}
