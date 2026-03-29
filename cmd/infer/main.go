package main

import (
	"flag"
	"fmt"
	"log"
	"math"

	"github.com/umbralcalc/energy-balancer/pkg/grid"
	"github.com/umbralcalc/stochadex/pkg/analysis"
	"github.com/umbralcalc/stochadex/pkg/general"
	"github.com/umbralcalc/stochadex/pkg/inference"
	"github.com/umbralcalc/stochadex/pkg/kernels"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

func main() {
	dataPath := flag.String("data", "dat/demand.csv", "Path to NESO demand CSV")
	steps := flag.Int("steps", 2016, "Number of steps to infer over (default: ~6 weeks)")
	ouTheta := flag.Float64("theta", 0.5, "Initial OU theta guess (mean-reversion speed per half-hour)")
	ouSigma := flag.Float64("sigma", 1000.0, "Initial OU sigma guess (MW per sqrt-half-hour); stored internally as log(sigma)")
	windowDepth := flag.Int("window", 100, "Rolling window depth for mean/variance estimation")
	pastDiscount := flag.Float64("discount", 1.0, "Past discounting factor for posterior (1.0 = full memory)")
	flag.Parse()

	// Step 1: Replay observed demand data to populate storage.
	// Also computes lagged_residual_demand = residual_demand[t-1], used in
	// the OU transition likelihood as the previous state.
	log.Println("Step 1: Replaying observed demand data...")
	storage := analysis.NewStateTimeStorageFromPartitions(
		[]*simulator.PartitionConfig{
			{
				Name:              "grid_data",
				Iteration:         &grid.GridDataIteration{CsvPath: *dataPath},
				Params:            simulator.NewParams(make(map[string][]float64)),
				InitStateValues:   []float64{22000, 1500, 0},
				StateHistoryDepth: 1,
				Seed:              0,
			},
			{
				Name:      "residual_demand",
				Iteration: &grid.ResidualDemandIteration{},
				Params:    simulator.NewParams(make(map[string][]float64)),
				ParamsAsPartitions: map[string][]string{
					"upstream_partition": {"grid_data"},
				},
				InitStateValues:   []float64{20500},
				StateHistoryDepth: 1,
				Seed:              0,
			},
			{
				Name:              "conditional_mean",
				Iteration:         &grid.ConditionalMeanIteration{CsvPath: *dataPath},
				Params:            simulator.NewParams(make(map[string][]float64)),
				InitStateValues:   []float64{20500},
				StateHistoryDepth: 1,
				Seed:              0,
			},
			{
				Name:      "lagged_residual_demand",
				Iteration: &grid.LaggedValuesIteration{},
				Params:    simulator.NewParams(make(map[string][]float64)),
				ParamsAsPartitions: map[string][]string{
					"source_partition": {"residual_demand"},
				},
				InitStateValues:   []float64{20500},
				StateHistoryDepth: 1,
				Seed:              0,
			},
		},
		&simulator.NumberOfStepsTerminationCondition{MaxNumberOfSteps: *steps},
		&simulator.ConstantTimestepFunction{Stepsize: 0.5},
		0.0,
	)
	log.Printf("  %d steps stored", len(storage.GetTimes()))

	// Step 2: Compute rolling mean and variance of observed residual demand.
	log.Println("Step 2: Computing rolling mean and variance of residual demand...")
	meanPartition := analysis.NewVectorMeanPartition(
		analysis.AppliedAggregation{
			Name:   "residual_demand_mean",
			Data:   analysis.DataRef{PartitionName: "residual_demand"},
			Kernel: &kernels.ExponentialIntegrationKernel{},
		},
		storage,
	)
	meanPartition.Params.Set(
		"exponential_weighting_timescale", []float64{float64(*windowDepth)})
	storage = analysis.AddPartitionsToStateTimeStorage(
		storage,
		[]*simulator.PartitionConfig{meanPartition},
		map[string]int{"residual_demand": *windowDepth},
	)

	varPartition := analysis.NewVectorVariancePartition(
		analysis.DataRef{PartitionName: "residual_demand_mean"},
		analysis.AppliedAggregation{
			Name:         "residual_demand_var",
			Data:         analysis.DataRef{PartitionName: "residual_demand"},
			Kernel:       &kernels.ExponentialIntegrationKernel{},
			DefaultValue: 1e6,
		},
		storage,
	)
	varPartition.Params.Set(
		"exponential_weighting_timescale", []float64{float64(*windowDepth)})
	storage = analysis.AddPartitionsToStateTimeStorage(
		storage,
		[]*simulator.PartitionConfig{varPartition},
		map[string]int{
			"residual_demand":      *windowDepth,
			"residual_demand_mean": 1,
		},
	)
	log.Println("  Rolling statistics computed")

	// Step 3: Infer OU parameters [theta, sigma] using posterior estimation
	// with the exact OU transition likelihood.
	//
	// At each outer step, the sampler draws candidate [theta, sigma] from the
	// current posterior. Inside the comparison window, we evaluate the cumulative
	// OU transition log-likelihood over windowDepth observed steps:
	//
	//   sum_n log P(X(t-n+1) | X(t-n), theta, sigma, mu(t-n+1))
	//
	// using the Euler-Maruyama density: X_next ~ N(X_prev + theta*(mu-X_prev)*dt, sigma^2*dt).
	//
	// This is numerically stable for all theta (no inner simulation to diverge),
	// so the posterior correctly weights negative theta as near-zero probability.
	log.Println("Step 3: Inferring OU parameters via posterior estimation...")

	// Convert sigma to log scale: OUTransitionLikelihood interprets "sigmas" as log(sigma).
	// Both theta and log(sigma) are ~O(1), preventing covariance corruption from scale mismatch.
	logSigma := math.Log(*ouSigma)

	// Passthrough partition: exposes the current sampler's [theta, log(sigma)] as
	// inner-simulation state, so comparison.ParamsFromUpstream can wire them.
	ouParamsPassthrough := &simulator.PartitionConfig{
		Name:              "ou_params_passthrough",
		Iteration:         &general.ParamValuesIteration{},
		Params:            simulator.NewParams(map[string][]float64{"param_values": {*ouTheta, logSigma}}),
		InitStateValues:   []float64{*ouTheta, logSigma},
		StateHistoryDepth: 1,
		Seed:              0,
	}

	// Prior covariance: diagonal [theta_var, log_sigma_var].
	//   theta    ~ N(0.5, 0.25)  → std 0.5
	//   log(sigma) ~ N(log(1000), 1.0) → std 1.0 (covers ~e^-1 to e^1 multiplicative range)
	priorCov := []float64{0.25, 0.0, 0.0, 1.0}

	posteriorPartitions := analysis.NewPosteriorEstimationPartitions(
		analysis.AppliedPosteriorEstimation{
			LogNorm: analysis.PosteriorLogNorm{
				Name:    "ou_log_norm",
				Default: 0.0,
			},
			Mean: analysis.PosteriorMean{
				Name:    "ou_posterior_mean",
				Default: []float64{*ouTheta, logSigma},
			},
			Covariance: analysis.PosteriorCovariance{
				Name:    "ou_posterior_cov",
				Default: priorCov,
			},
			Sampler: analysis.PosteriorSampler{
				Name:    "ou_params_sampler",
				Default: []float64{*ouTheta, logSigma},
				Distribution: analysis.ParameterisedModel{
					Likelihood: &inference.NormalLikelihoodDistribution{},
					Params: simulator.NewParams(map[string][]float64{
						"default_covariance": priorCov,
					}),
					ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
						"mean":              {Upstream: "ou_posterior_mean"},
						"covariance_matrix": {Upstream: "ou_posterior_cov"},
					},
				},
			},
			Comparison: analysis.AppliedLikelihoodComparison{
				Name: "ou_loglikelihood",
				// Model evaluates the OU transition log-likelihood using the
				// Euler-Maruyama density. Inside the window, residual_demand and
				// lagged_residual_demand replay historical X(t) and X(t-1);
				// conditional_mean replays historical mu(t). The sampler's
				// [theta, sigma] are forwarded via ou_params_passthrough.
				Model: analysis.ParameterisedModel{
					Likelihood: &grid.OUTransitionLikelihood{},
					Params:     simulator.NewParams(make(map[string][]float64)),
					ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
						"previous_state": {Upstream: "lagged_residual_demand"},
						"mus":            {Upstream: "conditional_mean"},
						"thetas":         {Upstream: "ou_params_passthrough", Indices: []int{0}},
						"sigmas":         {Upstream: "ou_params_passthrough", Indices: []int{1}},
					},
				},
				Data: analysis.DataRef{PartitionName: "residual_demand"},
				Window: analysis.WindowedPartitions{
					Partitions: []analysis.WindowedPartition{{
						Partition: ouParamsPassthrough,
						OutsideUpstreams: map[string]simulator.NamedUpstreamConfig{
							"param_values": {Upstream: "ou_params_sampler"},
						},
					}},
					Data: []analysis.DataRef{
						{PartitionName: "residual_demand"},
						{PartitionName: "lagged_residual_demand"},
						{PartitionName: "conditional_mean"},
					},
					Depth: *windowDepth,
				},
			},
			PastDiscount: *pastDiscount,
			MemoryDepth:  *windowDepth,
			Seed:         1234,
		},
		storage,
	)

	storage = analysis.AddPartitionsToStateTimeStorage(
		storage,
		posteriorPartitions,
		map[string]int{
			"residual_demand":        *windowDepth,
			"lagged_residual_demand": *windowDepth,
			"conditional_mean":       *windowDepth,
			"residual_demand_mean":   *windowDepth,
			"residual_demand_var":    *windowDepth,
		},
	)

	posteriorMeanValues := storage.GetValues("ou_posterior_mean")
	finalParams := posteriorMeanValues[len(posteriorMeanValues)-1]

	// Debug: print a sample of posterior mean values over time.
	loglikeValues := storage.GetValues("ou_loglikelihood")
	samplerValues := storage.GetValues("ou_params_sampler")
	logNormValues := storage.GetValues("ou_log_norm")
	covValues := storage.GetValues("ou_posterior_cov")
	fmt.Println()
	fmt.Println("Step | theta_mean | logσ_mean | sampler_θ | sampler_σ | loglike_last |  cov00 |  cov01 |  cov11")
	stride := len(posteriorMeanValues) / 30
	if stride < 1 {
		stride = 1
	}
	for i := 0; i < len(posteriorMeanValues); i += stride {
		ll := loglikeValues[i][len(loglikeValues[i])-1]
		_ = logNormValues
		fmt.Printf("%4d | %10.4f | %9.4f | %9.4f | %9.4f | %12.2f | %6.3f | %6.3f | %6.3f\n",
			i,
			posteriorMeanValues[i][0],
			posteriorMeanValues[i][1],
			samplerValues[i][0],
			samplerValues[i][1],
			ll,
			covValues[i][0],
			covValues[i][1],
			covValues[i][3],
		)
	}

	fmt.Println()
	fmt.Println("Inferred OU parameters for residual demand model:")
	fmt.Printf("  theta (mean-reversion speed, per half-hour): %.4f\n", finalParams[0])
	fmt.Printf("  sigma (volatility, MW/sqrt(half-hour)):      %.2f\n", math.Exp(finalParams[1]))
	fmt.Println()
	fmt.Printf("Update cmd/simulate price_noise with these values:\n")
	fmt.Printf("  thetas: [%.4f]  sigmas: [%.2f]\n", finalParams[0], math.Exp(finalParams[1]))

	log.Println("Done.")
}
