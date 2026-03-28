package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/umbralcalc/energy-balancer/pkg/grid"
	"github.com/umbralcalc/stochadex/pkg/analysis"
	"github.com/umbralcalc/stochadex/pkg/continuous"
	"github.com/umbralcalc/stochadex/pkg/inference"
	"github.com/umbralcalc/stochadex/pkg/kernels"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

func main() {
	dataPath := flag.String("data", "dat/demand.csv", "Path to NESO demand CSV")
	steps := flag.Int("steps", 2016, "Number of steps to infer over (default: ~6 weeks)")
	ouTheta := flag.Float64("theta", 0.5, "Initial OU theta guess (mean-reversion speed per half-hour)")
	ouSigma := flag.Float64("sigma", 1000.0, "Initial OU sigma guess (MW per sqrt-half-hour)")
	windowDepth := flag.Int("window", 100, "Rolling window depth for mean/variance estimation")
	pastDiscount := flag.Float64("discount", 1.0, "Past discounting factor for posterior (1.0 = full memory)")
	flag.Parse()

	// Step 1: Replay observed demand data to populate storage.
	// Partitions: grid_data, residual_demand, conditional_mean.
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
				// ParamsAsPartitions resolves "grid_data" to its integer index,
				// which ResidualDemandIteration.Configure reads as upstream_partition.
				ParamsAsPartitions: map[string][]string{
					"upstream_partition": {"grid_data"},
				},
				InitStateValues:   []float64{20500},
				StateHistoryDepth: *windowDepth,
				Seed:              0,
			},
			{
				Name:              "conditional_mean",
				Iteration:         &grid.ConditionalMeanIteration{CsvPath: *dataPath},
				Params:            simulator.NewParams(make(map[string][]float64)),
				InitStateValues:   []float64{20500},
				StateHistoryDepth: *windowDepth,
				Seed:              0,
			},
		},
		&simulator.NumberOfStepsTerminationCondition{MaxNumberOfSteps: *steps},
		&simulator.ConstantTimestepFunction{Stepsize: 0.5},
		0.0,
	)
	log.Printf("  %d steps stored", len(storage.GetTimes()))

	// Step 2: Compute rolling mean and variance of observed residual demand.
	// These characterise the empirical distribution the OU model must match.
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

	// Step 3: Infer OU parameters [theta, sigma] using posterior estimation.
	//
	// Pattern follows simulation_inference_test.go in the stochadex test suite:
	//   - ou_params_sampler  generates candidate [theta, sigma] from the current posterior
	//   - comparison window  simulates OU(theta, sigma) with mus=conditional_mean
	//   - loglikelihood      evaluates how well the synthetic OU output matches the
	//                        empirical distribution of observed residual demand
	//   - ou_posterior_mean  tracks the likelihood-weighted mean of [theta, sigma]
	//
	// The inferred parameters converge to the [theta, sigma] that make the OU
	// simulation statistically consistent with the observed residual demand.
	log.Println("Step 3: Inferring OU parameters via posterior estimation...")

	ouSyntheticConfig := &simulator.PartitionConfig{
		Name: "ou_synthetic",
		Iteration: &continuous.OrnsteinUhlenbeckIteration{},
		Params: simulator.NewParams(map[string][]float64{
			"thetas": {*ouTheta},
			"sigmas": {*ouSigma},
			"mus":    {20500},
		}),
		InitStateValues:   []float64{20500},
		StateHistoryDepth: 1,
		Seed:              99,
	}

	// Prior covariance: diagonal [theta_var, sigma_var].
	// Chosen to cover plausible OU parameter ranges:
	//   theta ~ [0.01, 10] per half-hour → variance 4.0
	//   sigma ~ [100, 5000] MW/sqrt(hh) → variance 2.5e6
	priorCov := []float64{4.0, 0.0, 0.0, 2.5e6}

	posteriorPartitions := analysis.NewPosteriorEstimationPartitions(
		analysis.AppliedPosteriorEstimation{
			LogNorm: analysis.PosteriorLogNorm{
				Name:    "ou_log_norm",
				Default: 0.0,
			},
			Mean: analysis.PosteriorMean{
				Name:    "ou_posterior_mean",
				Default: []float64{*ouTheta, *ouSigma},
			},
			Covariance: analysis.PosteriorCovariance{
				Name:    "ou_posterior_cov",
				Default: priorCov,
			},
			Sampler: analysis.PosteriorSampler{
				Name:    "ou_params_sampler",
				Default: []float64{*ouTheta, *ouSigma},
				Distribution: analysis.ParameterisedModel{
					Likelihood: &inference.NormalLikelihoodDistribution{},
					Params: simulator.NewParams(map[string][]float64{
						"default_covariance": priorCov,
						"cov_burn_in_steps":  {float64(*windowDepth)},
					}),
					ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
						"mean":              {Upstream: "ou_posterior_mean"},
						"covariance_matrix": {Upstream: "ou_posterior_cov"},
					},
				},
			},
			Comparison: analysis.AppliedLikelihoodComparison{
				Name: "ou_loglikelihood",
				// Model evaluates how likely the OU synthetic output is given
				// the empirical distribution of observed residual demand.
				Model: analysis.ParameterisedModel{
					Likelihood: &inference.NormalLikelihoodDistribution{},
					Params:     simulator.NewParams(make(map[string][]float64)),
					ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
						"mean":     {Upstream: "residual_demand_mean"},
						"variance": {Upstream: "residual_demand_var"},
					},
				},
				// Compare synthetic OU output against the empirical distribution.
				Data: analysis.DataRef{PartitionName: "ou_synthetic"},
				Window: analysis.WindowedPartitions{
					Partitions: []analysis.WindowedPartition{{
						Partition: ouSyntheticConfig,
						// Wire theta and sigma from the sampler; mus from conditional_mean.
						OutsideUpstreams: map[string]simulator.NamedUpstreamConfig{
							"thetas": {Upstream: "ou_params_sampler", Indices: []int{0}},
							"sigmas": {Upstream: "ou_params_sampler", Indices: []int{1}},
							"mus":    {Upstream: "conditional_mean"},
						},
					}},
					Data: []analysis.DataRef{
						{PartitionName: "residual_demand_mean"},
						{PartitionName: "residual_demand_var"},
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
			"residual_demand":      *windowDepth,
			"conditional_mean":     *windowDepth,
			"residual_demand_mean": *windowDepth,
			"residual_demand_var":  *windowDepth,
		},
	)

	// Extract inferred parameters from the final posterior mean.
	posteriorMeanValues := storage.GetValues("ou_posterior_mean")
	finalParams := posteriorMeanValues[len(posteriorMeanValues)-1]

	fmt.Println()
	fmt.Println("Inferred OU parameters for residual demand model:")
	fmt.Printf("  theta (mean-reversion speed, per half-hour): %.4f\n", finalParams[0])
	fmt.Printf("  sigma (volatility, MW/sqrt(half-hour)):      %.2f\n", finalParams[1])
	fmt.Println()
	fmt.Printf("Update cmd/simulate price_noise with these values:\n")
	fmt.Printf("  thetas: [%.4f]  sigmas: [%.2f]\n", finalParams[0], finalParams[1])

	log.Println("Done.")
}
