package grid

import (
	"testing"

	"github.com/umbralcalc/stochadex/pkg/inference"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// Regression: OU inference sampler must draw independent marginals on
// [log(theta), log(sigma)] (diagonal variance), not perfectly correlated pairs.
func TestOUSampler_NormalLikelihood_GenerateNewSamplesIndependent(t *testing.T) {
	n := &inference.NormalLikelihoodDistribution{}
	settings := &simulator.Settings{
		Iterations: []simulator.IterationSettings{
			{Seed: 42},
		},
	}
	n.SetSeed(0, settings)

	params := simulator.NewParams(map[string][]float64{
		"mean":     {-0.5, 7.0},
		"variance": {0.25, 1.0},
	})
	n.SetParams(&params, 0, nil, nil)

	var a, b float64
	for range 200 {
		s := n.GenerateNewSamples()
		if len(s) != 2 {
			t.Fatalf("len=%d", len(s))
		}
		if s[0] != s[1] {
			a, b = s[0], s[1]
			return
		}
	}
	t.Fatalf("200 draws all had s[0]==s[1], last=%g,%g", a, b)
}
