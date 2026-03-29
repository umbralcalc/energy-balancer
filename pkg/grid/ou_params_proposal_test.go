package grid

import (
	"testing"

	"github.com/umbralcalc/stochadex/pkg/simulator"
	"gonum.org/v1/gonum/mat"
)

func TestOUParamsProposal_GenerateNewSamplesIndependent(t *testing.T) {
	p := &OUParamsProposal{}
	settings := &simulator.Settings{
		Iterations: []simulator.IterationSettings{
			{Seed: 42},
		},
	}
	p.SetSeed(0, settings)

	h := &simulator.StateHistory{
		Values:            mat.NewDense(1, 2, []float64{-0.5, 7.0}),
		StateWidth:        2,
		StateHistoryDepth: 1,
	}

	params := simulator.NewParams(map[string][]float64{
		"posterior_mean_partition_index": {0},
		"proposal_std_log_theta":         {0.5},
		"proposal_std_log_sigma":         {1.0},
	})

	p.SetParams(&params, 0, []*simulator.StateHistory{h}, nil)

	var a, b float64
	for range 200 {
		s := p.GenerateNewSamples()
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
