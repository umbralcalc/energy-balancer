package grid

import "github.com/umbralcalc/stochadex/pkg/simulator"

// LaggedValuesIteration outputs the state from one timestep ago of the
// source partition identified by params "source_partition" (resolved via
// ParamsAsPartitions).
type LaggedValuesIteration struct {
	sourceIdx int
}

func (l *LaggedValuesIteration) Configure(
	partitionIndex int,
	settings *simulator.Settings,
) {
	l.sourceIdx = int(
		settings.Iterations[partitionIndex].Params.GetIndex("source_partition", 0))
}

func (l *LaggedValuesIteration) Iterate(
	params *simulator.Params,
	_ int,
	stateHistories []*simulator.StateHistory,
	_ *simulator.CumulativeTimestepsHistory,
) []float64 {
	h := stateHistories[l.sourceIdx]
	// Iterate is called before UpdateHistory, so Row(0) already holds the
	// PREVIOUS step's output — exactly the 1-step lag we need.
	src := h.Values.RawRowView(0)
	out := make([]float64, len(src))
	copy(out, src)
	return out
}
