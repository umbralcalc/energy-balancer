package custom

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
// where residual_demand = national_demand - embedded_wind - embedded_solar.
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
	upstream := stateHistories[r.upstreamPartitionIndex]
	demand := upstream.Values.At(0, 0)
	wind := upstream.Values.At(0, 1)
	solar := upstream.Values.At(0, 2)
	residual := demand - wind - solar
	return []float64{residual}
}
