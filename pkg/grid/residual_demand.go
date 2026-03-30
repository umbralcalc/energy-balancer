package grid

import (
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// ResidualDemandIteration computes residual demand from an upstream
// grid data partition. Residual demand is the demand that must be met
// by dispatchable generation and storage after subtracting wind and solar.
//
// It reads from an upstream partition whose state vector is:
//
//	[national_demand_mw, embedded_wind_mw, embedded_solar_mw]
//
// And outputs:
//
//	[residual_demand_mw]
//
// where residual_demand = national_demand - wind*wind_scale - solar*solar_scale.
//
// Optional params:
//
//	wind_scale  [dimensionless] - multiply embedded wind by this factor (default 1.0)
//	solar_scale [dimensionless] - multiply embedded solar by this factor (default 1.0)
//
// Set wind_scale > 1 and solar_scale > 1 to simulate future grid mixes with
// higher renewable penetration (e.g. 2030 Holistic Transition scenario: 2.1, 2.0).
type ResidualDemandIteration struct {
	upstreamPartitionIndex int
}

func (r *ResidualDemandIteration) Configure(
	partitionIndex int,
	settings *simulator.Settings,
) {
	r.upstreamPartitionIndex = int(
		settings.Iterations[partitionIndex].Params.Map["upstream_partition"][0],
	)
}

func (r *ResidualDemandIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	windScale := 1.0
	if v, ok := params.Map["wind_scale"]; ok {
		windScale = v[0]
	}
	solarScale := 1.0
	if v, ok := params.Map["solar_scale"]; ok {
		solarScale = v[0]
	}

	upstream := stateHistories[r.upstreamPartitionIndex]
	demand := upstream.Values.At(0, 0)
	wind := upstream.Values.At(0, 1)
	solar := upstream.Values.At(0, 2)
	residual := demand - wind*windScale - solar*solarScale
	return []float64{residual}
}
