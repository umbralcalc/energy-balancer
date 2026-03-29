package grid

import (
	"math"

	"github.com/umbralcalc/stochadex/pkg/simulator"
	"gonum.org/v1/gonum/stat/distuv"
)

// OUTransitionLikelihood evaluates the exact Ornstein-Uhlenbeck
// transition log-likelihood: log P(X_next | X_prev, theta, sigma, mu).
//
// Uses the exact OU transition density (not Euler-Maruyama):
//
//	mean = mu + (X_prev - mu) * exp(-theta * dt)
//	var  = sigma^2 / (2*theta) * (1 - exp(-2*theta*dt))
//
// This is numerically stable for any theta > 0, including large values where
// the EM approximation (which requires theta*dt << 1) breaks down catastrophically.
// For theta <= 0 (explosive/non-mean-reverting) returns -Inf.
//
// Required params (set via ParamsFromUpstream):
//
//	"thetas"         - OU mean-reversion speed (scalar, must be > 0)
//	"sigmas"         - log(sigma), OU volatility in log scale
//	"mus"            - OU long-run mean (scalar)
//	"previous_state" - X at the previous timestep (scalar)
type OUTransitionLikelihood struct {
	mean float64
	std  float64
}

func (o *OUTransitionLikelihood) SetSeed(_ int, _ *simulator.Settings) {}

func (o *OUTransitionLikelihood) SetParams(
	params *simulator.Params,
	_ int,
	_ []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) {
	theta := params.GetIndex("thetas", 0)
	// "sigmas" param is log(sigma) — keeps sigma > 0 and puts it on a
	// similar scale to theta, preventing covariance corruption.
	sigma := math.Exp(params.GetIndex("sigmas", 0))
	mu := params.GetIndex("mus", 0)
	xPrev := params.GetIndex("previous_state", 0)
	dt := timestepsHistory.NextIncrement

	if theta <= 0 {
		// Non-mean-reverting: mark as invalid so EvaluateLogLike returns -Inf.
		o.std = -1
		return
	}

	// Exact OU transition: stable for all theta > 0, no theta*dt << 1 requirement.
	expNeg := math.Exp(-theta * dt)
	o.mean = mu + (xPrev-mu)*expNeg
	variance := (sigma * sigma / (2 * theta)) * (1 - expNeg*expNeg)
	if variance <= 0 {
		o.std = -1
		return
	}
	o.std = math.Sqrt(variance)
}

func (o *OUTransitionLikelihood) EvaluateLogLike(data []float64) float64 {
	if o.std <= 0 || math.IsInf(o.mean, 0) || math.IsNaN(o.mean) {
		return math.Inf(-1)
	}
	dist := distuv.Normal{Mu: o.mean, Sigma: o.std}
	return dist.LogProb(data[0])
}

func (o *OUTransitionLikelihood) GenerateNewSamples() []float64 {
	return []float64{o.mean}
}
