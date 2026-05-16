package energydash

import (
	"github.com/umbralcalc/energy-balancer/pkg/grid"
	"github.com/umbralcalc/stochadex/pkg/continuous"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// BuildEnergySimulation constructs the stochadex generator for the
// energy-balancer dashboard.
//
// The partition graph mirrors the offline cmd/simulate graph (NESO
// demand replay → residual demand → imbalance price → dispatch policy
// → battery dispatch + degradation), with three dashboard-specific
// additions:
//
//   - policy_action: an action partition carrying the discrete policy +
//     scenario choices, the four threshold sliders, and the scenario-
//     derived wind/solar capacity scale factors. Downstream partitions
//     read these via ParamsFromUpstream so a slider change takes effect
//     on the next half-hour step.
//   - dispatch_policy: a switching iteration that collapses the
//     project's separate price- and carbon-threshold iterations into a
//     single partition. The graph shape stays the same in both modes;
//     policy_action[0] picks which branch fires per step.
//   - outcomes / soc_display / display_progress: dashboard read-out
//     partitions that compute the headline metrics (cumulative net
//     value, revenue, degradation cost, carbon savings, EFCs, fraction
//     of periods active) and the SoC percentage the line chart picks
//     up from state[0].
func BuildEnergySimulation() *simulator.ConfigGenerator {
	parseEmbeddedData()

	// --- Controls (slider/radio-driven). ---
	policyAction := &simulator.PartitionConfig{
		Name:      "policy_action",
		Iteration: &PolicyActionIteration{},
		Params: simulator.NewParams(map[string][]float64{
			"action_state_values": {
				PolicyPrice,
				Scenario2025,
				DefaultPriceHigh,
				DefaultPriceLow,
				DefaultCarbonHigh,
				DefaultCarbonLow,
				WindScales[Scenario2025],
				SolarScales[Scenario2025],
			},
		}),
		InitStateValues: []float64{
			PolicyPrice,
			Scenario2025,
			DefaultPriceHigh,
			DefaultPriceLow,
			DefaultCarbonHigh,
			DefaultCarbonLow,
			WindScales[Scenario2025],
			SolarScales[Scenario2025],
		},
		StateHistoryDepth: 1,
		Seed:              0,
	}

	// --- Grid + carbon data replay. ---
	gridData := &simulator.PartitionConfig{
		Name:              "grid_data",
		Iteration:         &EmbeddedGridDataIteration{},
		Params:            simulator.NewParams(map[string][]float64{}),
		InitStateValues:   []float64{22000, 1500, 0},
		StateHistoryDepth: 2,
		Seed:              0,
	}
	carbonData := &simulator.PartitionConfig{
		Name:              "carbon_data",
		Iteration:         &EmbeddedCarbonDataIteration{},
		Params:            simulator.NewParams(map[string][]float64{}),
		InitStateValues:   []float64{180.0},
		StateHistoryDepth: 2,
		Seed:              0,
	}

	// --- Residual demand: ND - wind*windScale - solar*solarScale. ---
	residualDemand := &simulator.PartitionConfig{
		Name:      "residual_demand",
		Iteration: &grid.ResidualDemandIteration{},
		Params:    simulator.NewParams(map[string][]float64{}),
		ParamsAsPartitions: map[string][]string{
			"upstream_partition": {"grid_data"},
		},
		ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
			"wind_scale":  {Upstream: "policy_action", Indices: []int{PAIdxWindScale}},
			"solar_scale": {Upstream: "policy_action", Indices: []int{PAIdxSolarScale}},
		},
		InitStateValues:   []float64{20500},
		StateHistoryDepth: 2,
		Seed:              0,
	}

	// --- Imbalance price: structural linear response + OU noise. ---
	priceNoise := &simulator.PartitionConfig{
		Name:      "price_noise",
		Iteration: &continuous.OrnsteinUhlenbeckIteration{},
		Params: simulator.NewParams(map[string][]float64{
			"thetas": {PriceNoiseTheta},
			"mus":    {0.0},
			"sigmas": {PriceNoiseSigma},
		}),
		InitStateValues:   []float64{0.0},
		StateHistoryDepth: 2,
		Seed:              42,
	}
	imbalancePrice := &simulator.PartitionConfig{
		Name:      "imbalance_price",
		Iteration: &grid.ImbalancePriceIteration{},
		Params: simulator.NewParams(map[string][]float64{
			"demand_slope":     {PriceSlope},
			"demand_intercept": {PriceIntercept},
		}),
		ParamsAsPartitions: map[string][]string{
			"demand_partition": {"residual_demand"},
			"noise_partition":  {"price_noise"},
		},
		InitStateValues:   []float64{31.0},
		StateHistoryDepth: 2,
		Seed:              0,
	}

	// --- Dispatch policy: switches between price/carbon thresholds. ---
	dispatchPolicy := &simulator.PartitionConfig{
		Name:      "dispatch_policy",
		Iteration: &SwitchingDispatchIteration{},
		Params: simulator.NewParams(map[string][]float64{
			"power_rating_mw": {PowerRatingMW},
		}),
		ParamsAsPartitions: map[string][]string{
			"price_partition":  {"imbalance_price"},
			"carbon_partition": {"carbon_data"},
		},
		ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
			"policy_action": {Upstream: "policy_action"},
		},
		InitStateValues:   []float64{0.0},
		StateHistoryDepth: 2,
		Seed:              0,
	}

	// --- Battery: state-of-charge tracker. ---
	battery := &simulator.PartitionConfig{
		Name:      "battery",
		Iteration: &grid.BatteryIteration{},
		Params: simulator.NewParams(map[string][]float64{
			"energy_capacity_mwh":  {CapacityMWh},
			"power_rating_mw":      {PowerRatingMW},
			"charge_efficiency":    {ChargeEfficiency},
			"discharge_efficiency": {ChargeEfficiency},
			"min_soc_fraction":     {MinSoCFraction},
			"max_soc_fraction":     {MaxSoCFraction},
		}),
		ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
			"dispatch_mw": {Upstream: "dispatch_policy"},
		},
		InitStateValues:   []float64{CapacityMWh * 0.5, 0},
		StateHistoryDepth: 2,
		Seed:              0,
	}

	// --- SoC display: percentage form, primary y for the line chart. ---
	socDisplay := &simulator.PartitionConfig{
		Name:      "soc_display",
		Iteration: &SoCDisplayIteration{},
		Params: simulator.NewParams(map[string][]float64{
			"energy_capacity_mwh": {CapacityMWh},
		}),
		ParamsAsPartitions: map[string][]string{
			"battery_partition": {"battery"},
		},
		InitStateValues:   []float64{50.0, CapacityMWh * 0.5},
		StateHistoryDepth: 1,
		Seed:              0,
	}

	// --- Outcomes: cumulative net value, revenue, deg cost, carbon, EFCs, %active. ---
	outcomes := &simulator.PartitionConfig{
		Name:      "outcomes",
		Iteration: &OutcomesIteration{},
		Params: simulator.NewParams(map[string][]float64{
			"energy_capacity_mwh": {CapacityMWh},
		}),
		ParamsAsPartitions: map[string][]string{
			"battery_partition": {"battery"},
			"price_partition":   {"imbalance_price"},
			"carbon_partition":  {"carbon_data"},
		},
		InitStateValues:   []float64{0, 0, 0, 0, 0, 0, 0},
		StateHistoryDepth: 1,
		Seed:              0,
	}

	// --- Progress readout. ---
	displayProgress := &simulator.PartitionConfig{
		Name:      "display_progress",
		Iteration: &DisplayProgressIteration{},
		Params:    simulator.NewParams(map[string][]float64{}),
		ParamsAsPartitions: map[string][]string{
			"soc_partition": {"soc_display"},
		},
		InitStateValues:   []float64{0, 50.0},
		StateHistoryDepth: 1,
		Seed:              0,
	}

	gen := simulator.NewConfigGenerator()
	for _, p := range []*simulator.PartitionConfig{
		policyAction,
		gridData,
		carbonData,
		residualDemand,
		priceNoise,
		imbalancePrice,
		dispatchPolicy,
		battery,
		socDisplay,
		outcomes,
		displayProgress,
	} {
		gen.SetPartition(p)
	}

	gen.SetSimulation(&simulator.SimulationConfig{
		OutputCondition: &simulator.EveryStepOutputCondition{},
		TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
			MaxNumberOfSteps: SimSteps,
		},
		TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: Stepsize},
		InitTimeValue:    0.0,
	})
	return gen
}
