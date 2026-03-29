package grid

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestODEMLEFromDemandStorage_NoNoiseExact(t *testing.T) {
	dt := 0.5
	theta := 0.2
	mu := 20000.0
	phi := math.Exp(-theta * dt)
	beta := 1 - phi
	x := 20500.0
	n := 100
	rd := make([][]float64, n)
	lag := make([][]float64, n)
	mus := make([][]float64, n)
	rd[0] = []float64{x}
	lag[0] = []float64{x}
	mus[0] = []float64{mu}
	for i := 1; i < n; i++ {
		xPrev := rd[i-1][0]
		d := mu - xPrev
		delta := beta * d
		x = xPrev + delta
		rd[i] = []float64{x}
		lag[i] = []float64{xPrev}
		mus[i] = []float64{mu}
	}
	th, sg := ODEMLEFromDemandStorage(rd, lag, mus, dt)
	if math.Abs(th-theta) > 1e-9 {
		t.Fatalf("theta got %g want %g", th, theta)
	}
	if sg > 1e-8 || math.IsNaN(sg) {
		t.Fatalf("sigma with zero innovations want ~0, got %g", sg)
	}
}

func TestODEMLEFromDemandStorage_WithNoise(t *testing.T) {
	dt := 0.5
	theta := 0.2
	sigma := 1000.0
	mu := 20000.0
	// Start far from μ so (μ−X_prev) stays informative for many steps.
	x := 28000.0
	n := 8000
	rng := rand.New(rand.NewPCG(42, 43))
	rd := make([][]float64, n)
	lag := make([][]float64, n)
	mus := make([][]float64, n)
	rd[0] = []float64{x}
	lag[0] = []float64{x}
	mus[0] = []float64{mu}
	for i := 1; i < n; i++ {
		xPrev := rd[i-1][0]
		exp := math.Exp(-theta * dt)
		m := mu + (xPrev-mu)*exp
		v := (sigma * sigma / (2 * theta)) * (1 - math.Exp(-2*theta*dt))
		x = m + math.Sqrt(v)*rng.NormFloat64()
		rd[i] = []float64{x}
		lag[i] = []float64{xPrev}
		mus[i] = []float64{mu}
	}
	th, sg := ODEMLEFromDemandStorage(rd, lag, mus, dt)
	if math.Abs(th-theta) > 0.02 {
		t.Fatalf("theta got %g want ~%g", th, theta)
	}
	if math.Abs(sg-sigma) > 200 {
		t.Fatalf("sigma got %g want ~%g", sg, sigma)
	}
}
