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
	ouTheta := flag.Float64("theta", 0.2, "Initial OU theta guess (per half-hour); near OLS ~0.17 on typical demand")
	ouSigma := flag.Float64("sigma", 1500.0, "Initial OU sigma guess (MW per sqrt-half-hour)")
	windowDepth := flag.Int("window", 100, "Rolling window depth for mean/variance estimation")
	pastDiscount := flag.Float64("discount", 1.0, "Past discounting factor for posterior (1.0 = full memory)")
	runSBI := flag.Bool("sbi", false, "Run stochadex analysis posterior + rolling stats (slow); default is OLS/MLE only")
	showAnalysisPosterior := flag.Bool("show-analysis-posterior", false, "With -sbi, print online posterior mean (often unstable; OLS is recommended)")
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

	demandStorage := storage
	dt := 0.5
	olsTheta, olsSigma := grid.ODEMLEFromStateTimeStorage(
		demandStorage,
		"residual_demand",
		"lagged_residual_demand",
		"conditional_mean",
		dt,
	)

	if !*runSBI {
		fmt.Println()
		if math.IsNaN(olsTheta) || math.IsNaN(olsSigma) {
			log.Println("Warning: OLS/MLE fit failed (insufficient or invalid data).")
		} else {
			fmt.Println("MLE/OLS on observed demand (recommended for cmd/simulate):")
			fmt.Printf("  theta (mean-reversion speed, per half-hour): %.4f\n", olsTheta)
			fmt.Printf("  sigma (volatility, MW/sqrt(half-hour)):      %.2f\n", olsSigma)
		}
		fmt.Println()
		if !math.IsNaN(olsTheta) && !math.IsNaN(olsSigma) {
			fmt.Println("Update cmd/simulate price_noise with:")
			fmt.Printf("  thetas: [%.4f]  sigmas: [%.2f]\n", olsTheta, olsSigma)
		}
		log.Println("Done. (Pass -sbi to run analysis-package posterior estimation.)")
		return
	}

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

	// Step 3: Posterior estimation (analysis.NewPosteriorEstimationPartitions).
	// Latent state is [log(theta), log(sigma)]; OUTransitionLikelihood exp()s them.
	// The sampler uses inference.NormalLikelihoodDistribution with fixed diagonal
	// variance (independent marginals). Mean comes from ou_posterior_mean; we do
	// not wire covariance_matrix from ou_posterior_cov (dense MVN + streamed matrix
	// previously produced perfectly correlated proposals mid-run).
	log.Println("Step 3: Inferring OU parameters via posterior estimation...")

	logTheta := math.Log(*ouTheta)
	logSigma := math.Log(*ouSigma)

	// Passthrough: inner window reads [log(theta), log(sigma)] from the sampler.
	ouParamsPassthrough := &simulator.PartitionConfig{
		Name:              "ou_params_passthrough",
		Iteration:         &general.ParamValuesIteration{},
		Params:            simulator.NewParams(map[string][]float64{"param_values": {logTheta, logSigma}}),
		InitStateValues:   []float64{logTheta, logSigma},
		StateHistoryDepth: 1,
		Seed:              0,
	}

	// Prior / initial covariance for ou_posterior_cov (online estimate; not used by sampler).
	priorCov := []float64{0.25, 0.0, 0.0, 1.0}

	posteriorPartitions := analysis.NewPosteriorEstimationPartitions(
		analysis.AppliedPosteriorEstimation{
			LogNorm: analysis.PosteriorLogNorm{
				Name:    "ou_log_norm",
				Default: 0.0,
			},
			Mean: analysis.PosteriorMean{
				Name:    "ou_posterior_mean",
				Default: []float64{logTheta, logSigma},
			},
			Covariance: analysis.PosteriorCovariance{
				Name:    "ou_posterior_cov",
				Default: priorCov,
			},
			Sampler: analysis.PosteriorSampler{
				Name:    "ou_params_sampler",
				Default: []float64{logTheta, logSigma},
				Distribution: analysis.ParameterisedModel{
					Likelihood: &inference.NormalLikelihoodDistribution{},
					Params: simulator.NewParams(map[string][]float64{
						// Diagonal proposal: match former OUParamsProposal stds 0.5, 1.0.
						"variance": {0.25, 1.0},
					}),
					ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
						"mean": {Upstream: "ou_posterior_mean"},
					},
				},
			},
			Comparison: analysis.AppliedLikelihoodComparison{
				Name: "ou_loglikelihood",
				// Exact OU transition density. Inside the window, residual_demand and
				// lagged_residual_demand replay historical X(t) and X(t-1);
				// conditional_mean replays historical mu(t). The sampler's
				// [theta, sigma] are forwarded via ou_params_passthrough.
				Model: analysis.ParameterisedModel{
					Likelihood: &grid.OUTransitionLikelihood{},
					Params:     simulator.NewParams(make(map[string][]float64)),
					ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
						"previous_state": {Upstream: "lagged_residual_demand"},
						"mus":            {Upstream: "conditional_mean"},
						"thetas":         {Upstream: "ou_params_passthrough", Indices: []int{0}}, // log(theta)
						"sigmas":         {Upstream: "ou_params_passthrough", Indices: []int{1}}, // log(sigma)
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
	thetaHat := math.Exp(finalParams[0])
	sigmaHat := math.Exp(finalParams[1])

	fmt.Println()
	if math.IsNaN(olsTheta) || math.IsNaN(olsSigma) {
		log.Println("Warning: OLS/MLE fit failed (insufficient or invalid data).")
	} else {
		fmt.Println("MLE/OLS on observed demand (recommended for cmd/simulate):")
		fmt.Printf("  theta (mean-reversion speed, per half-hour): %.4f\n", olsTheta)
		fmt.Printf("  sigma (volatility, MW/sqrt(half-hour)):      %.2f\n", olsSigma)
	}
	fmt.Println()
	if *showAnalysisPosterior {
		fmt.Println("Online posterior mean (stochadex analysis — diagnostic only):")
		fmt.Printf("  theta: %.4f\n", thetaHat)
		fmt.Printf("  sigma: %.2f\n", sigmaHat)
		fmt.Println()
	}
	if !math.IsNaN(olsTheta) && !math.IsNaN(olsSigma) {
		fmt.Println("Update cmd/simulate price_noise with the MLE/OLS row:")
		fmt.Printf("  thetas: [%.4f]  sigmas: [%.2f]\n", olsTheta, olsSigma)
	} else {
		fmt.Println("Update cmd/simulate using the diagnostic row if OLS failed.")
		fmt.Printf("  thetas: [%.4f]  sigmas: [%.2f]\n", thetaHat, sigmaHat)
	}

	log.Println("Done.")
}
