package grid

import (
	"math"

	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// ODEMLEFromDemandStorage estimates (theta, sigma) for the exact OU transition
//
//	X_next ~ N( μ + (X_prev-μ)e^{-θΔ},  σ²/(2θ) (1 - e^{-2θΔ}) )
//
// by regressing ΔX = X_next − X_prev on d = μ − X_prev. The Gaussian AR(1)
// form gives E[ΔX | X_prev] = (1 − e^{-θΔ}) d, so β := Cov(ΔX,d)/Var(d) ≈
// 1 − e^{-θΔ} and θ = −ln(1−β)/Δ. Then σ is recovered from the innovation
// variance Var(ΔX − β d) ≈ σ²/(2θ)(1 − e^{-2θΔ}).
//
// rdValues, laggedValues, muValues must align on the same time index (same
// length); typically lagged[i] ≈ rd[i-1] from LaggedValuesIteration.
func ODEMLEFromDemandStorage(
	rdValues, laggedValues, muValues [][]float64,
	dt float64,
) (theta, sigma float64) {
	n := len(rdValues)
	if n < 3 || len(laggedValues) != n || len(muValues) != n {
		return math.NaN(), math.NaN()
	}
	var sumDeltaD, sumDD float64
	count := 0
	for i := 1; i < n; i++ {
		x := rdValues[i][0]
		xPrev := laggedValues[i][0]
		mu := muValues[i][0]
		if math.IsNaN(x) || math.IsNaN(xPrev) || math.IsNaN(mu) {
			continue
		}
		delta := x - xPrev
		d := mu - xPrev
		sumDeltaD += delta * d
		sumDD += d * d
		count++
	}
	if count < 2 || sumDD <= 0 {
		return math.NaN(), math.NaN()
	}
	beta := sumDeltaD / sumDD
	if beta <= 1e-9 || beta >= 1.0-1e-12 {
		return math.NaN(), math.NaN()
	}
	theta = -math.Log(1-beta) / dt

	var sumR2 float64
	for i := 1; i < n; i++ {
		x := rdValues[i][0]
		xPrev := laggedValues[i][0]
		mu := muValues[i][0]
		if math.IsNaN(x) || math.IsNaN(xPrev) || math.IsNaN(mu) {
			continue
		}
		delta := x - xPrev
		d := mu - xPrev
		r := delta - beta*d
		sumR2 += r * r
	}
	varR := sumR2 / float64(count)
	den := 1.0 - math.Exp(-2*theta*dt)
	if den <= 0 || varR < 0 {
		return theta, math.NaN()
	}
	sigma = math.Sqrt(2 * theta * varR / den)
	if math.IsNaN(sigma) || sigma <= 0 {
		return theta, math.NaN()
	}
	return theta, sigma
}

// ODEMLEFromStateTimeStorage reads aligned series from storage by partition name.
func ODEMLEFromStateTimeStorage(
	store *simulator.StateTimeStorage,
	rdName, laggedName, muName string,
	dt float64,
) (theta, sigma float64) {
	return ODEMLEFromDemandStorage(
		store.GetValues(rdName),
		store.GetValues(laggedName),
		store.GetValues(muName),
		dt,
	)
}
