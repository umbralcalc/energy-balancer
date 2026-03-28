package main

import (
	"flag"
	"log"

	"github.com/umbralcalc/energy-balancer/pkg/grid"
	"github.com/umbralcalc/stochadex/pkg/continuous"
	"github.com/umbralcalc/stochadex/pkg/general"
	"github.com/umbralcalc/stochadex/pkg/inference"
	"github.com/umbralcalc/stochadex/pkg/kernels"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// Partition indices
const (
	idxGridData       = 0
	idxResidualDemand = 1
	idxCondMean       = 2
	idxResidualOU     = 3
	idxRollingMean    = 4
	idxRollingCov     = 5
	idxLogLikelihood  = 6
)

func main() {
	dataPath := flag.String("data", "dat/demand.csv", "Path to NESO demand CSV")
	steps := flag.Int("steps", 2016, "Number of steps to run (default: 6 weeks = 2016 half-hours)")
	outPath := flag.String("out", "dat/inference.log", "Output JSON log path")
	// OU parameters for the residual demand model — adjust to fit
	ouTheta := flag.Float64("theta", 0.5, "OU mean-reversion speed (per hour)")
	ouSigma := flag.Float64("sigma", 1000.0, "OU volatility (MW per sqrt-hour)")
	flag.Parse()

	// Rolling window depth for mean/covariance estimation
	const windowDepth = 100

	settings := &simulator.Settings{
		Iterations: []simulator.IterationSettings{
			{
				// 0: replay observed ND, embedded wind, embedded solar
				Name:              "grid_data",
				Params:            simulator.NewParams(map[string][]float64{}),
				InitStateValues:   []float64{22000, 1500, 0},
				StateWidth:        3,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				// 1: observed residual demand = ND - wind - solar
				Name: "residual_demand",
				Params: simulator.NewParams(map[string][]float64{
					"upstream_partition": {idxGridData},
				}),
				InitStateValues:   []float64{20500},
				StateWidth:        1,
				StateHistoryDepth: windowDepth,
				Seed:              0,
			},
			{
				// 2: conditional mean by settlement period × month
				Name:              "conditional_mean",
				Params:            simulator.NewParams(map[string][]float64{}),
				InitStateValues:   []float64{20500},
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				// 3: OU residual demand model — mus wired from conditional mean
				Name: "residual_ou",
				Params: simulator.NewParams(map[string][]float64{
					"thetas": {*ouTheta},
					"sigmas": {*ouSigma},
					"mus":    {20500}, // overridden by params_from_upstream below
				}),
				ParamsFromUpstream: map[string]simulator.UpstreamConfig{
					"mus": {Upstream: idxCondMean},
				},
				InitStateValues:   []float64{20500},
				StateWidth:        1,
				StateHistoryDepth: windowDepth,
				Seed:              99,
			},
			{
				// 4: rolling mean of the OU simulated residual demand
				Name: "rolling_mean",
				Params: simulator.NewParams(map[string][]float64{
					"exponential_weighting_timescale": {float64(windowDepth)},
					"data_values_partition":           {idxResidualOU},
				}),
				ParamsFromUpstream: map[string]simulator.UpstreamConfig{
					"latest_data_values": {Upstream: idxResidualOU},
				},
				InitStateValues:   []float64{20500},
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				// 5: rolling covariance of the OU simulated residual demand
				Name: "rolling_cov",
				Params: simulator.NewParams(map[string][]float64{
					"exponential_weighting_timescale": {float64(windowDepth)},
					"data_values_partition":           {idxResidualOU},
				}),
				ParamsFromUpstream: map[string]simulator.UpstreamConfig{
					"latest_data_values": {Upstream: idxResidualOU},
					"mean":               {Upstream: idxRollingMean},
				},
				InitStateValues:   []float64{1e6}, // initial variance guess
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				// 6: log-likelihood of observed residual demand given the OU model
				Name: "log_likelihood",
				Params: simulator.NewParams(map[string][]float64{
					"burn_in_steps": {float64(windowDepth)},
					"cumulative":    {1},
				}),
				ParamsFromUpstream: map[string]simulator.UpstreamConfig{
					"latest_data_values": {Upstream: idxResidualDemand},
					"mean":               {Upstream: idxRollingMean},
					"covariance_matrix":  {Upstream: idxRollingCov},
				},
				InitStateValues:   []float64{0},
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
		},
		InitTimeValue:         0.0,
		TimestepsHistoryDepth: windowDepth,
	}

	gridIter := &grid.GridDataIteration{CsvPath: *dataPath}
	condMeanIter := &grid.ConditionalMeanIteration{CsvPath: *dataPath}

	iterations := []simulator.Iteration{
		gridIter,
		&grid.ResidualDemandIteration{},
		condMeanIter,
		&continuous.OrnsteinUhlenbeckIteration{},
		&general.ValuesFunctionVectorMeanIteration{
			Function: general.DataValuesFunction,
			Kernel:   &kernels.ExponentialIntegrationKernel{},
		},
		&general.ValuesFunctionVectorCovarianceIteration{
			Function: general.DataValuesFunction,
			Kernel:   &kernels.ExponentialIntegrationKernel{},
		},
		&inference.DataComparisonIteration{
			Likelihood: &inference.NormalLikelihoodDistribution{},
		},
	}
	for i, iter := range iterations {
		iter.Configure(i, settings)
	}

	numSteps := *steps
	if numSteps == 0 {
		numSteps = gridIter.DataLen() - 1
	}

	implementations := &simulator.Implementations{
		Iterations:      iterations,
		OutputCondition: &simulator.EveryStepOutputCondition{},
		OutputFunction:  simulator.NewJsonLogOutputFunction(*outPath),
		TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
			MaxNumberOfSteps: numSteps,
		},
		TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 0.5},
	}

	log.Printf("Running inference: %d steps → %s", numSteps, *outPath)
	simulator.NewPartitionCoordinator(settings, implementations).Run()
	log.Println("Done.")
}
