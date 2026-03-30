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
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

func main() {
	dataPath := flag.String("data", "dat/demand.csv", "Path to NESO demand CSV")
	steps := flag.Int("steps", 2016, "Number of steps to infer over (default: ~6 weeks of half-hours)")
	runSMC := flag.Bool("smc", false, "Run SMC Bayesian inference in addition to OLS")
	numParticles := flag.Int("particles", 200, "Number of SMC particles")
	numRounds := flag.Int("rounds", 20, "Number of SMC rounds")
	flag.Parse()

	// -------------------------------------------------------------------------
	// Step 1: Replay observed demand data.
	// Produces: residual_demand, conditional_mean, lagged_residual_demand.
	// -------------------------------------------------------------------------
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

	// -------------------------------------------------------------------------
	// Step 2: OLS via streaming scalar regression.
	//
	// Regresses ΔX = X_next - X_prev (ou_delta_rd) on d = mu - X_prev (ou_d_mean)
	// with no intercept. The Gaussian AR(1) form gives:
	//   E[ΔX | X_prev] = (1 - exp(-θΔ)) · d
	//   β = Cov(ΔX, d) / Var(d) ≈ 1 - exp(-θΔ)
	//   θ = -ln(1-β) / Δ
	//   σ = sqrt(2θ · Var(residual) / (1 - exp(-2θΔ)))
	//
	// Uses analysis.NewScalarRegressionStatsPartition for the streaming summation.
	// -------------------------------------------------------------------------
	log.Println("Step 2: OLS estimation of OU parameters...")

	// Intermediate difference partitions (wired via ParamsFromUpstream so they
	// see the current step's upstream outputs, not the one-step-lagged history).
	deltaRdPartition := &simulator.PartitionConfig{
		Name:            "ou_delta_rd",
		Iteration:       &grid.ScalarDifferenceIteration{},
		Params:          simulator.NewParams(make(map[string][]float64)),
		ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
			"a": {Upstream: "residual_demand"},
			"b": {Upstream: "lagged_residual_demand"},
		},
		InitStateValues:   []float64{0},
		StateHistoryDepth: 1,
		Seed:              0,
	}
	dMeanPartition := &simulator.PartitionConfig{
		Name:            "ou_d_mean",
		Iteration:       &grid.ScalarDifferenceIteration{},
		Params:          simulator.NewParams(make(map[string][]float64)),
		ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
			"a": {Upstream: "conditional_mean"},
			"b": {Upstream: "lagged_residual_demand"},
		},
		InitStateValues:   []float64{0},
		StateHistoryDepth: 1,
		Seed:              0,
	}

	// Add difference partitions first so the OLS partition can reference them.
	storage = analysis.AddPartitionsToStateTimeStorage(
		storage,
		[]*simulator.PartitionConfig{deltaRdPartition, dMeanPartition},
		nil,
	)

	olsPartition := analysis.NewScalarRegressionStatsPartition(
		analysis.AppliedScalarRegressionStats{
			Name:              "ou_ols",
			Y:                 analysis.DataRef{PartitionName: "ou_delta_rd"},
			X:                 analysis.DataRef{PartitionName: "ou_d_mean"},
			Intercept:         false,
			Mode:              analysis.RegressionStatsCumulative,
			StateHistoryDepth: 1,
		},
		storage,
	)
	storage = analysis.AddPartitionsToStateTimeStorage(
		storage,
		[]*simulator.PartitionConfig{olsPartition},
		nil,
	)

	// Extract OLS estimates from the final state.
	// State layout (no intercept, cumulative): [Sxx, Sxy, Syy, n, beta, sigma2]
	dt := 0.5
	olsState := storage.GetValues("ou_ols")
	finalOLS := olsState[len(olsState)-1]
	beta := finalOLS[4]
	sigma2 := finalOLS[5]

	var olsTheta, olsSigma float64
	olsOK := beta > 1e-9 && beta < 1.0-1e-12 && sigma2 >= 0
	if olsOK {
		olsTheta = -math.Log(1-beta) / dt
		den := 1.0 - math.Exp(-2*olsTheta*dt)
		if den > 0 {
			olsSigma = math.Sqrt(2 * olsTheta * sigma2 / den)
		} else {
			olsOK = false
		}
	}

	if !olsOK || math.IsNaN(olsTheta) || math.IsNaN(olsSigma) {
		log.Println("Warning: OLS fit failed (insufficient or degenerate data).")
	} else {
		fmt.Println()
		fmt.Println("OLS/MLE estimates (recommended for cmd/simulate):")
		fmt.Printf("  theta (mean-reversion speed, /half-hour): %.4f\n", olsTheta)
		fmt.Printf("  sigma (volatility, MW/sqrt(half-hour)):   %.2f\n", olsSigma)
		fmt.Printf("  → thetas: [%.4f]  sigmas: [%.2f]\n", olsTheta, olsSigma)
	}

	if !*runSMC {
		log.Println("Done. (Pass -smc to run SMC Bayesian inference.)")
		return
	}

	// -------------------------------------------------------------------------
	// Step 3: SMC Bayesian inference of log(theta) and log(sigma).
	//
	// Builds N particles each evaluated against the exact OU transition
	// log-likelihood over all T steps of replayed demand data.
	// The OUTransitionLikelihood interprets "thetas" as log(theta) and
	// "sigmas" as log(sigma), so both params are on a similar scale and the
	// SMC covariance regularisation works well.
	// -------------------------------------------------------------------------
	log.Printf("Step 3: SMC Bayesian inference (%d particles, %d rounds)...",
		*numParticles, *numRounds)

	// Centre priors on OLS estimates if available; fall back to wide defaults.
	priorLogTheta := math.Log(0.2)
	priorLogSigma := math.Log(1500.0)
	if olsOK && olsTheta > 0 && olsSigma > 0 {
		priorLogTheta = math.Log(olsTheta)
		priorLogSigma = math.Log(olsSigma)
	}

	// Load the replayed series for embedding in the inner SMC simulation.
	rdData := storage.GetValues("residual_demand")
	lagData := storage.GetValues("lagged_residual_demand")
	muData := storage.GetValues("conditional_mean")
	T := len(rdData)

	N := *numParticles
	result := analysis.RunSMCInference(analysis.AppliedSMCInference{
		ProposalName:  "ou_proposal",
		SimName:       "ou_sim",
		PosteriorName: "ou_posterior",
		NumParticles:  N,
		NumRounds:     *numRounds,
		ParamNames:    []string{"log_theta", "log_sigma"},
		Priors: []inference.Prior{
			// log(theta): truncated normal centred on OLS, ±2 log-units wide
			&inference.TruncatedNormalPrior{
				Mu:    priorLogTheta,
				Sigma: 1.5,
				Lo:    math.Log(1e-4),
				Hi:    math.Log(20.0),
			},
			// log(sigma): truncated normal centred on OLS, ±2 log-units wide
			&inference.TruncatedNormalPrior{
				Mu:    priorLogSigma,
				Sigma: 1.5,
				Lo:    math.Log(10.0),
				Hi:    math.Log(1e5),
			},
		},
		Seed:    42,
		Verbose: true,
		Model: analysis.SMCParticleModel{
			Build: func(nParticles, nParams int) *analysis.SMCInnerSimConfig {
				return buildOUInnerSim(nParticles, rdData, lagData, muData, T)
			},
		},
	})

	if result == nil {
		log.Println("SMC inference returned no result.")
		return
	}

	smcTheta := math.Exp(result.PosteriorMean[0])
	smcSigma := math.Exp(result.PosteriorMean[1])
	smcThetaStd := smcTheta * result.PosteriorStd[0] // delta method
	smcSigmaStd := smcSigma * result.PosteriorStd[1]

	fmt.Println()
	fmt.Println("SMC Bayesian posterior estimates:")
	fmt.Printf("  theta: %.4f ± %.4f /half-hour\n", smcTheta, smcThetaStd)
	fmt.Printf("  sigma: %.2f ± %.2f MW/sqrt(half-hour)\n", smcSigma, smcSigmaStd)
	fmt.Printf("  log marginal likelihood: %.2f\n", result.LogMarginalLik)
	fmt.Printf("  → thetas: [%.4f]  sigmas: [%.2f]\n", smcTheta, smcSigma)

	log.Println("Done.")
}

// buildOUInnerSim constructs the SMC inner simulation that evaluates N particles
// against the exact OU transition log-likelihood over T-1 data steps.
//
// Each particle p has params [log_theta_p, log_sigma_p] forwarded from the
// proposal partition (flat index p*2, p*2+1).
func buildOUInnerSim(
	N int,
	rdData, lagData, muData [][]float64,
	T int,
) *analysis.SMCInnerSimConfig {
	partitions := make([]*simulator.PartitionConfig, 0, 3+2*N)
	loglikePartitions := make([]string, N)
	paramForwarding := make(map[string][]int, N)

	// Shared data replay partitions.
	partitions = append(partitions,
		&simulator.PartitionConfig{
			Name:              "residual_demand",
			Iteration:         &general.FromStorageIteration{Data: rdData},
			Params:            simulator.NewParams(make(map[string][]float64)),
			InitStateValues:   rdData[0],
			StateHistoryDepth: 1,
			Seed:              0,
		},
		&simulator.PartitionConfig{
			Name:      "lagged_residual_demand",
			Iteration: &grid.LaggedValuesIteration{},
			Params:    simulator.NewParams(make(map[string][]float64)),
			ParamsAsPartitions: map[string][]string{
				"source_partition": {"residual_demand"},
			},
			InitStateValues:   lagData[0],
			StateHistoryDepth: 1,
			Seed:              0,
		},
		&simulator.PartitionConfig{
			Name:              "conditional_mean",
			Iteration:         &general.FromStorageIteration{Data: muData},
			Params:            simulator.NewParams(make(map[string][]float64)),
			InitStateValues:   muData[0],
			StateHistoryDepth: 1,
			Seed:              0,
		},
	)

	// Per-particle params passthrough and loglike accumulator.
	for p := range N {
		paramsName := fmt.Sprintf("ou_params_%d", p)
		loglikeName := fmt.Sprintf("ou_loglike_%d", p)

		partitions = append(partitions, &simulator.PartitionConfig{
			Name:            paramsName,
			Iteration:       &general.ParamValuesIteration{},
			Params:          simulator.NewParams(map[string][]float64{"param_values": {0, 0}}),
			InitStateValues: []float64{0, 0},
			StateHistoryDepth: 1,
			Seed:            0,
		})

		partitions = append(partitions, &simulator.PartitionConfig{
			Name: loglikeName,
			Iteration: &inference.DataComparisonIteration{
				Likelihood: &grid.OUTransitionLikelihood{},
			},
			Params: simulator.NewParams(map[string][]float64{
				"cumulative":    {1},
				"burn_in_steps": {0},
			}),
			ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
				"latest_data_values": {Upstream: "residual_demand"},
				"previous_state":     {Upstream: "lagged_residual_demand"},
				"mus":                {Upstream: "conditional_mean"},
				"thetas":             {Upstream: paramsName, Indices: []int{0}},
				"sigmas":             {Upstream: paramsName, Indices: []int{1}},
			},
			InitStateValues:   []float64{0.0},
			StateHistoryDepth: 1,
			Seed:              0,
		})

		loglikePartitions[p] = loglikeName
		paramForwarding[paramsName+"/param_values"] = []int{p * 2, p*2 + 1}
	}

	return &analysis.SMCInnerSimConfig{
		Partitions: partitions,
		Simulation: &simulator.SimulationConfig{
			OutputCondition: &simulator.NilOutputCondition{},
			OutputFunction:  &simulator.NilOutputFunction{},
			TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
				MaxNumberOfSteps: T - 1,
			},
			TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 0.5},
			InitTimeValue:    0.0,
		},
		LoglikePartitions: loglikePartitions,
		ParamForwarding:   paramForwarding,
	}
}
