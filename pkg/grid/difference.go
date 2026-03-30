package grid

import "github.com/umbralcalc/stochadex/pkg/simulator"

// ScalarDifferenceIteration outputs the scalar difference a - b where a and b
// are single-element values wired in via ParamsFromUpstream keys "a" and "b".
// Use this to build intermediate partitions for OLS regression inputs such as
// ΔX = X_next - X_prev or d = mu - X_prev.
type ScalarDifferenceIteration struct{}

func (s *ScalarDifferenceIteration) Configure(_ int, _ *simulator.Settings) {}

func (s *ScalarDifferenceIteration) Iterate(
	params *simulator.Params,
	_ int,
	_ []*simulator.StateHistory,
	_ *simulator.CumulativeTimestepsHistory,
) []float64 {
	return []float64{params.GetIndex("a", 0) - params.GetIndex("b", 0)}
}
