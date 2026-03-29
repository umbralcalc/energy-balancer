package grid

import (
	"math/rand/v2"

	"github.com/umbralcalc/stochadex/pkg/inference"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// OUParamsProposal implements inference.LikelihoodDistribution for use with
// inference.DataGenerationIteration only (GenerateNewSamples). It draws
// independent normals:
//
//	log(theta)    ~ N(mean[0], stdLogTheta^2)
//	log(sigma)    ~ N(mean[1], stdLogSigma^2)
//
// Mean is read from stateHistories[idx] where idx = int(params
// "posterior_mean_partition_index"[0]). (Do not use ParamsAsPartitions
// "mean_partition" + partition names: storage.GetNames() iteration order is
// map-randomised while name→index maps are still correct, and we observed
// mis-resolved indices pointing at the sampler partition, duplicating the
// first proposal component into the second.)
//
// Proposal scales are "proposal_std_log_theta" and "proposal_std_log_sigma".
//
// This avoids distmv.Normal + mat.SymDense backing-store issues seen when the
// multivariate normal path started producing perfectly correlated samples
// mid-run.
type OUParamsProposal struct {
	rnd         *rand.Rand
	mean        []float64
	stdLogTheta float64
	stdLogSigma float64
}

func (o *OUParamsProposal) SetSeed(partitionIndex int, settings *simulator.Settings) {
	seed := settings.Iterations[partitionIndex].Seed
	// Distinct PCG words — rand.NewPCG(seed, seed) is unnecessarily symmetric.
	o.rnd = rand.New(rand.NewPCG(seed, seed+1))
}

func (o *OUParamsProposal) SetParams(
	params *simulator.Params,
	_ int,
	stateHistories []*simulator.StateHistory,
	_ *simulator.CumulativeTimestepsHistory,
) {
	idx := int(params.GetIndex("posterior_mean_partition_index", 0))
	o.stdLogTheta = params.GetIndex("proposal_std_log_theta", 0)
	o.stdLogSigma = params.GetIndex("proposal_std_log_sigma", 0)
	row := stateHistories[idx].CopyStateRow(0)
	o.mean = append(o.mean[:0], row...)
}

func (o *OUParamsProposal) EvaluateLogLike([]float64) float64 {
	panic("grid.OUParamsProposal: EvaluateLogLike is not used for DataGenerationIteration")
}

func (o *OUParamsProposal) GenerateNewSamples() []float64 {
	return []float64{
		o.mean[0] + o.stdLogTheta*o.rnd.NormFloat64(),
		o.mean[1] + o.stdLogSigma*o.rnd.NormFloat64(),
	}
}

var _ inference.LikelihoodDistribution = (*OUParamsProposal)(nil)
