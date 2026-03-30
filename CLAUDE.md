# Stochadex SDK — Project Conventions

This project uses the [stochadex](https://github.com/umbralcalc/stochadex) SDK to build and run simulations.

## The Iteration Interface

Every simulation component implements `simulator.Iteration`:

```go
type Iteration interface {
    Configure(partitionIndex int, settings *Settings)
    Iterate(params *Params, partitionIndex int, stateHistories []*StateHistory,
            timestepsHistory *CumulativeTimestepsHistory) []float64
}
```

**Rules:**
- `Configure` is called once at setup. Use it to seed RNGs, read fixed config, allocate buffers. All mutable state must be re-initializable here (no statefulness residues between runs).
- `Iterate` is called each step. It must NOT mutate `params`. It returns the next state as `[]float64` with length equal to `StateWidth`.
- `stateHistories` gives access to all partitions' rolling state windows. `stateHistories[i].Values.At(row, col)` where row=0 is the latest state.
- `timestepsHistory.Values.AtVec(0)` is the current time. `timestepsHistory.NextIncrement` is the upcoming time step.
- Partitions communicate by wiring one partition's output state into another's params via `params_from_upstream` in config.

## YAML Config Format (API Code-Generation Path)

Simulations are defined in YAML and run via the stochadex CLI, which generates and executes Go code.

```yaml
main:
  partitions:
  - name: my_partition              # unique name
    iteration: myVar                # references a variable from extra_vars
    params:
      some_param: [1.0, 2.0]       # all param values are []float64
    params_from_upstream:           # wire upstream partition output → this partition's params
      latest_values:
        upstream: other_partition   # name of the upstream partition
    params_as_partitions:           # reference partition names as param values (resolved to indices)
      data_partition: [some_partition]
    init_state_values: [0.0, 0.0]  # initial state (determines state_width)
    state_history_depth: 1          # rolling window size
    seed: 1234                      # RNG seed (0 = no randomness needed)
    extra_packages:                 # Go import paths
    - github.com/umbralcalc/stochadex/pkg/continuous
    extra_vars:                     # Go variable declarations
    - myVar: "&continuous.WienerProcessIteration{}"

  simulation:
    output_condition: "&simulator.EveryStepOutputCondition{}"
    output_function: "&simulator.StdoutOutputFunction{}"
    termination_condition: "&simulator.NumberOfStepsTerminationCondition{MaxNumberOfSteps: 100}"
    timestep_function: "&simulator.ConstantTimestepFunction{Stepsize: 1.0}"
    init_time_value: 0.0
```

### Common Output Functions
- `&simulator.StdoutOutputFunction{}` — print to stdout
- `simulator.NewJsonLogOutputFunction("./data.log")` — write JSON log file
- `&simulator.NilOutputFunction{}` — no output (for embedded sims)

### Common Termination Conditions
- `&simulator.NumberOfStepsTerminationCondition{MaxNumberOfSteps: N}`
- `&simulator.TimeElapsedTerminationCondition{MaxTimeElapsed: T}`

### Common Timestep Functions
- `&simulator.ConstantTimestepFunction{Stepsize: 0.1}`
- `&simulator.ExponentialDistributionTimestepFunction{RateLambda: 1.0}`

## Build & Run

```bash
go build ./...                                    # compile this project
go test -count=1 ./...                            # run all tests
go run github.com/umbralcalc/stochadex/cmd/stochadex --config cfg/builtin_example.yaml
```

## Testing Conventions

- Unit tests live alongside source as `*_test.go` files.
- Use `t.Run("description", func(t *testing.T) { ... })` subtests.
- Always include a subtest using `simulator.RunWithHarnesses(settings, implementations)` — this checks for NaN outputs, wrong state widths, params mutation, state history integrity, and statefulness residues.
- Load settings from a colocated YAML file (e.g., `my_iteration_settings.yaml` next to `my_iteration_test.go`).
- Use `gonum.org/v1/gonum/floats` for float comparisons, never raw `==`.
- No mocking — use real implementations.

## Built-In Iterations Reference

### continuous (github.com/umbralcalc/stochadex/pkg/continuous)

| Iteration | Params | Description |
|-----------|--------|-------------|
| `WienerProcessIteration` | `variances` | Brownian motion |
| `GeometricBrownianMotionIteration` | `variances` | Multiplicative Brownian motion |
| `OrnsteinUhlenbeckIteration` | `thetas`, `mus`, `sigmas` | Mean-reverting process |
| `DriftDiffusionIteration` | `drift_coefficients`, `diffusion_coefficients` | General drift-diffusion SDE |
| `DriftJumpDiffusionIteration` | `drift_coefficients`, `diffusion_coefficients`, `jump_rates` | Drift-diffusion with Poisson jumps |
| `CompoundPoissonProcessIteration` | `rates` | Compound Poisson process |
| `GradientDescentIteration` | `gradient`, `learning_rate`, `ascent` (optional) | Gradient-based optimization |
| `CumulativeTimeIteration` | (none) | Outputs cumulative simulation time |

### discrete (github.com/umbralcalc/stochadex/pkg/discrete)

| Iteration | Params | Description |
|-----------|--------|-------------|
| `PoissonProcessIteration` | `rates` | Poisson counting process |
| `BernoulliProcessIteration` | `state_value_observation_probs` | Binary outcomes |
| `BinomialObservationProcessIteration` | `observed_values`, `state_value_observation_probs`, `state_value_observation_indices` | Binomial draws |
| `CoxProcessIteration` | `rates` | Doubly-stochastic Poisson |
| `HawkesProcessIteration` | `intensity` | Self-exciting point process |
| `HawkesProcessIntensityIteration` | `background_rates` | Hawkes intensity function |
| `CategoricalStateTransitionIteration` | `transitions_from_N`, `transition_rates` | State machine |

### general (github.com/umbralcalc/stochadex/pkg/general)

| Iteration | Params | Description |
|-----------|--------|-------------|
| `ConstantValuesIteration` | (none) | Returns unchanged initial state |
| `CopyValuesIteration` | `partitions`, `partition_state_values` | Copies values from other partitions |
| `ParamValuesIteration` | `param_values` | Injects param values as state |
| `ValuesFunctionIteration` | (varies by Function) | User-defined function of state |
| `CumulativeIteration` | (none, wraps another) | Accumulates wrapped iteration output |
| `FromStorageIteration` | (none, uses Data field) | Streams pre-computed data |
| `FromHistoryIteration` | `latest_data_values` | Replays state history data |
| `EmbeddedSimulationRunIteration` | (varies) | Runs nested simulation each step |
| `ValuesCollectionIteration` | `values_state_width`, `empty_value` | Rolling collection of values |
| `ValuesChangingEventsIteration` | `default_values` | Routes by event value |
| `ValuesWeightedResamplingIteration` | `log_weight_partitions`, `data_values_partitions`, `past_discounting_factor` | Weighted resampling |
| `ValuesFunctionVectorMeanIteration` | `data_values_partition`, `latest_data_values` | Kernel-weighted rolling mean |
| `ValuesFunctionVectorCovarianceIteration` | `data_values_partition`, `latest_data_values`, `mean` | Kernel-weighted rolling covariance |

### inference (github.com/umbralcalc/stochadex/pkg/inference)

| Iteration | Params | Description |
|-----------|--------|-------------|
| `DataGenerationIteration` | `steps_per_resample`, `correlation_with_previous` (optional) | Synthetic data generation |
| `DataComparisonIteration` | `latest_data_values`, `cumulative`, `burn_in_steps` | Log-likelihood evaluation |
| `PosteriorMeanIteration` | `loglike_partitions`, `param_partitions`, `posterior_log_normalisation` | Posterior mean estimation |
| `PosteriorCovarianceIteration` | `loglike_partitions`, `param_partitions`, `posterior_log_normalisation`, `mean` | Posterior covariance estimation |
| `PosteriorLogNormalisationIteration` | `loglike_partitions`, `past_discounting_factor` | Log-normalisation tracking |

### kernels (github.com/umbralcalc/stochadex/pkg/kernels)

Kernels are not iterations — they implement `IntegrationKernel` and are used by iterations like `ValuesFunctionVectorMeanIteration`. Available: `ConstantIntegrationKernel`, `ExponentialIntegrationKernel`, `PeriodicIntegrationKernel`, `GaussianStateIntegrationKernel`, `TDistributionStateIntegrationKernel`, `BinnedIntegrationKernel`, `ProductIntegrationKernel`, `InstantaneousIntegrationKernel`.

## Project-Specific Iterations (`pkg/grid`)

| Iteration | Params | Description |
|-----------|--------|-------------|
| `GridDataIteration` | (reads CSV at `CsvPath`) | Streams NESO demand CSV: [total_demand, wind_solar, period_index] |
| `ResidualDemandIteration` | `upstream_partition` (via ParamsAsPartitions) | Computes residual demand = total_demand − wind_solar |
| `ConditionalMeanIteration` | (reads CSV at `CsvPath`) | Returns historical half-hourly mean demand for the current settlement period |
| `LaggedValuesIteration` | `source_partition` (via ParamsAsPartitions) | Returns the previous step's value of the source partition (reads row 0 of stateHistories before UpdateHistory runs) |
| `ScalarDifferenceIteration` | `a`, `b` (via ParamsFromUpstream) | Returns `[a − b]`. Stateless. Used to build OLS regression inputs ΔX and d. |
| `OUTransitionLikelihood` | `thetas` (log θ), `sigmas` (log σ), `mus`, `previous_state` (via ParamsFromUpstream) | Exact OU transition log-likelihood. Params are **log-scale**: internally does exp(). Guards invalid params with std=−1 → −∞ loglike. |
| `BatteryIteration` | `capacity_mwh`, `max_power_mw`, `efficiency`, dispatch threshold params | Battery SoC tracker with charge/discharge logic |
| `ImbalancePriceIteration` | `upstream_partition` (via ParamsAsPartitions) | Imbalance price as function of residual demand |
| `OutcomesIteration` | various upstream partitions | Computes per-step balancing cost, carbon, curtailment outcomes |

## Analysis Package Tools Used

**stochadex version:** `v0.0.0-20260330061034-1555b7d4e430`

### `analysis.NewStateTimeStorageFromPartitions`
Runs a complete simulation and stores all partition state histories. Used in `cmd/infer` Step 1 to replay observed demand data.

### `analysis.AddPartitionsToStateTimeStorage`
Adds new partitions to an existing storage, replaying over stored data. Used to build OLS intermediate partitions (`ou_delta_rd`, `ou_d_mean`) and the regression stats partition on top of replayed demand data.

### `analysis.NewScalarRegressionStatsPartition`
Builds a streaming scalar OLS regression partition. State layout (no-intercept, cumulative): `[Sxx, Sxy, Syy, n, beta, sigma2]`. Used in `cmd/infer` Step 2.

```go
olsPartition := analysis.NewScalarRegressionStatsPartition(
    analysis.AppliedScalarRegressionStats{
        Name:      "ou_ols",
        Y:         analysis.DataRef{PartitionName: "ou_delta_rd"},
        X:         analysis.DataRef{PartitionName: "ou_d_mean"},
        Intercept: false,
        Mode:      analysis.RegressionStatsCumulative,
    }, storage)
// beta = finalOLS[4], sigma2 = finalOLS[5]
// theta = -log(1-beta) / dt
// sigma = sqrt(2*theta*sigma2 / (1 - exp(-2*theta*dt)))
```

### `analysis.RunSMCInference`
Batch sequential Monte Carlo. N particles, each evaluated via an embedded `SMCInnerSimConfig`. Used in `cmd/infer` Step 3. Priors are in log-space (log θ, log σ); results are exponentiated via delta method.

```go
result := analysis.RunSMCInference(analysis.AppliedSMCInference{
    NumParticles: N, NumRounds: rounds,
    ParamNames:   []string{"log_theta", "log_sigma"},
    Priors:       []inference.Prior{...TruncatedNormalPrior...},
    Model: analysis.SMCParticleModel{
        Build: func(nParticles, nParams int) *analysis.SMCInnerSimConfig {
            return buildOUInnerSim(nParticles, rdData, lagData, muData, T)
        },
    },
})
smcTheta := math.Exp(result.PosteriorMean[0])
smcSigma := math.Exp(result.PosteriorMean[1])
```

The inner sim (`buildOUInnerSim`) uses `general.FromStorageIteration{Data: ...}` to embed pre-computed data, `general.ParamValuesIteration` for per-particle param passthrough, and `inference.DataComparisonIteration{Likelihood: &grid.OUTransitionLikelihood{}}` for per-particle log-likelihood accumulation.

## Command Reference

| Command | Usage | Description |
|---------|-------|-------------|
| `cmd/ingest` | `go run ./cmd/ingest` | Downloads NESO demand CSV to `dat/demand.csv` |
| `cmd/simulate` | `go run ./cmd/simulate -thetas 0.24 -sigmas 1591` | Runs stochastic grid simulation |
| `cmd/infer` | `go run ./cmd/infer -data dat/demand.csv [-smc -particles 200 -rounds 20]` | Infers OU parameters; OLS always, SMC optional |
| `cmd/evaluate` | `go run ./cmd/evaluate` | Evaluates battery dispatch policy outcomes |

## Key Architectural Notes

- **`ParamsFromUpstream` vs `stateHistories`**: `ParamsFromUpstream` reflects the **current step's** upstream output. `stateHistories[i].Values.At(0, 0)` is **one step behind** (updated after all Iterate calls). Use `ParamsFromUpstream` when you need same-step values (e.g., `ScalarDifferenceIteration`); use `LaggedValuesIteration` (via `ParamsAsPartitions`) to get the previous step's value intentionally.
- **OUTransitionLikelihood log-parameterisation**: "thetas" and "sigmas" params are log(θ) and log(σ). This ensures θ,σ > 0 always and keeps SMC covariance regularisation numerically stable.
- **SMC inner sim data embedding**: Use `general.FromStorageIteration{Data: storage.GetValues("partition_name")}` to replay historical series inside the SMC inner simulation. The inner sim runs T−1 steps with `ConstantTimestepFunction{Stepsize: 0.5}` matching the original data cadence.
