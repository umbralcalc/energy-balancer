package main

import (
	"flag"
	"log"

	"github.com/umbralcalc/energy-balancer/pkg/grid"
	"github.com/umbralcalc/stochadex/pkg/continuous"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// Partition indices — shared across both policy modes.
// Carbon policy inserts a carbon_data partition between grid_data and
// residual_demand, shifting all downstream indices up by 1.
const (
	idxGridData       = 0
	idxResidualDemand = 1
	idxPriceNoise     = 2
	idxPrice          = 3
	idxDispatch       = 4
	idxBattery        = 5

	// Carbon policy replaces idxDispatch; carbon data sits at the same index
	// as dispatch in the price policy (4), with dispatch at 5 and battery at 6.
	idxCarbonData        = 4
	idxCarbonDispatch    = 5
	idxCarbonBattery     = 6

	// Degradation sits after battery in each policy.
	idxDegradation       = 6 // price policy: degradation at index 6
	idxCarbonDegradation = 7 // carbon policy: degradation at index 7
)

func main() {
	dataPath := flag.String("data", "dat/demand.csv", "Path to NESO demand CSV")
	carbonPath := flag.String("carbon", "dat/carbon_intensity.csv", "Path to carbon intensity CSV")
	steps := flag.Int("steps", 0, "Number of steps to simulate (0 = all available data)")
	outPath := flag.String("out", "dat/simulation.log", "Output JSON log path")
	capacityMWh := flag.Float64("capacity", 200.0, "Battery energy capacity in MWh")
	ratingMW := flag.Float64("rating", 100.0, "Battery power rating in MW")
	policy := flag.String("policy", "price", "Dispatch policy: price or carbon")
	priceHigh := flag.Float64("price-high", 45.0, "Price policy: discharge threshold £/MWh")
	priceLow := flag.Float64("price-low", 25.0, "Price policy: charge threshold £/MWh")
	carbonHigh := flag.Float64("carbon-high", 250.0, "Carbon policy: discharge threshold gCO₂/kWh")
	carbonLow := flag.Float64("carbon-low", 100.0, "Carbon policy: charge threshold gCO₂/kWh")
	flag.Parse()

	switch *policy {
	case "price":
		runPricePolicy(*dataPath, *outPath, *capacityMWh, *ratingMW, *priceHigh, *priceLow, *steps)
	case "carbon":
		runCarbonPolicy(*dataPath, *carbonPath, *outPath, *capacityMWh, *ratingMW, *carbonHigh, *carbonLow, *steps)
	default:
		log.Fatalf("unknown policy %q — use 'price' or 'carbon'", *policy)
	}
}

func baseIterationSettings(capacityMWh, ratingMW float64) []simulator.IterationSettings {
	return []simulator.IterationSettings{
		{
			Name:              "grid_data",
			Params:            simulator.NewParams(map[string][]float64{}),
			InitStateValues:   []float64{22000, 1500, 0},
			StateWidth:        3,
			StateHistoryDepth: 2,
			Seed:              0,
		},
		{
			Name: "residual_demand",
			Params: simulator.NewParams(map[string][]float64{
				"upstream_partition": {idxGridData},
			}),
			InitStateValues:   []float64{20500},
			StateWidth:        1,
			StateHistoryDepth: 2,
			Seed:              0,
		},
		{
			Name: "price_noise",
			Params: simulator.NewParams(map[string][]float64{
				"thetas": {2.0},
				"mus":    {0.0},
				"sigmas": {5.0},
			}),
			InitStateValues:   []float64{0.0},
			StateWidth:        1,
			StateHistoryDepth: 2,
			Seed:              42,
		},
		{
			Name: "imbalance_price",
			Params: simulator.NewParams(map[string][]float64{
				"demand_slope":     {0.002},
				"demand_intercept": {-10.0},
				"demand_partition": {idxResidualDemand},
				"noise_partition":  {idxPriceNoise},
			}),
			InitStateValues:   []float64{31.0},
			StateWidth:        1,
			StateHistoryDepth: 2,
			Seed:              0,
		},
	}
}

func batterySettings(dispatchIdx int, capacityMWh, ratingMW float64) simulator.IterationSettings {
	return simulator.IterationSettings{
		Name: "battery",
		Params: simulator.NewParams(map[string][]float64{
			"dispatch_mw":          {0.0}, // overridden by ParamsFromUpstream
			"energy_capacity_mwh":  {capacityMWh},
			"power_rating_mw":      {ratingMW},
			"charge_efficiency":    {0.92},
			"discharge_efficiency": {0.92},
			"min_soc_fraction":     {0.1},
			"max_soc_fraction":     {0.9},
		}),
		ParamsFromUpstream: map[string]simulator.UpstreamConfig{
			"dispatch_mw": {Upstream: dispatchIdx},
		},
		InitStateValues:   []float64{capacityMWh * 0.5, 0},
		StateWidth:        2,
		StateHistoryDepth: 2,
		Seed:              0,
	}
}

func runPricePolicy(dataPath, outPath string, capacityMWh, ratingMW, priceHigh, priceLow float64, steps int) {
	iterSettings := baseIterationSettings(capacityMWh, ratingMW)
	iterSettings = append(iterSettings,
		simulator.IterationSettings{
			Name: "dispatch_policy",
			Params: simulator.NewParams(map[string][]float64{
				"price_partition": {idxPrice},
				"price_high":      {priceHigh},
				"price_low":       {priceLow},
				"power_rating_mw": {ratingMW},
			}),
			InitStateValues:   []float64{0.0},
			StateWidth:        1,
			StateHistoryDepth: 2,
			Seed:              0,
		},
		batterySettings(idxDispatch, capacityMWh, ratingMW),
		simulator.IterationSettings{
			Name: "degradation",
			Params: simulator.NewParams(map[string][]float64{
				"battery_partition":   {idxBattery},
				"energy_capacity_mwh": {capacityMWh},
			}),
			InitStateValues:   []float64{0.0},
			StateWidth:        1,
			StateHistoryDepth: 2,
			Seed:              0,
		},
	)

	_ = idxDegradation // used implicitly via idxBattery

	settings := &simulator.Settings{
		Iterations:            iterSettings,
		InitTimeValue:         0.0,
		TimestepsHistoryDepth: 2,
	}

	gridIter := &grid.GridDataIteration{CsvPath: dataPath}
	iterations := []simulator.Iteration{
		gridIter,
		&grid.ResidualDemandIteration{},
		&continuous.OrnsteinUhlenbeckIteration{},
		&grid.ImbalancePriceIteration{},
		&grid.PriceThresholdDispatchIteration{},
		&grid.BatteryIteration{},
		&grid.BatteryDegradationIteration{},
	}
	run(settings, iterations, gridIter, outPath, steps)
}

func runCarbonPolicy(dataPath, carbonPath, outPath string, capacityMWh, ratingMW, carbonHigh, carbonLow float64, steps int) {
	iterSettings := baseIterationSettings(capacityMWh, ratingMW)
	iterSettings = append(iterSettings,
		simulator.IterationSettings{
			Name:              "carbon_data",
			Params:            simulator.NewParams(map[string][]float64{}),
			InitStateValues:   []float64{180.0, 183.0}, // typical moderate intensity
			StateWidth:        2,
			StateHistoryDepth: 2,
			Seed:              0,
		},
		simulator.IterationSettings{
			Name: "dispatch_policy",
			Params: simulator.NewParams(map[string][]float64{
				"carbon_partition": {idxCarbonData},
				"carbon_high":      {carbonHigh},
				"carbon_low":       {carbonLow},
				"power_rating_mw":  {ratingMW},
			}),
			InitStateValues:   []float64{0.0},
			StateWidth:        1,
			StateHistoryDepth: 2,
			Seed:              0,
		},
		batterySettings(idxCarbonDispatch, capacityMWh, ratingMW),
		simulator.IterationSettings{
			Name: "degradation",
			Params: simulator.NewParams(map[string][]float64{
				"battery_partition":   {idxCarbonBattery},
				"energy_capacity_mwh": {capacityMWh},
			}),
			InitStateValues:   []float64{0.0},
			StateWidth:        1,
			StateHistoryDepth: 2,
			Seed:              0,
		},
	)

	_ = idxCarbonDegradation // used implicitly via idxCarbonBattery

	settings := &simulator.Settings{
		Iterations:            iterSettings,
		InitTimeValue:         0.0,
		TimestepsHistoryDepth: 2,
	}

	gridIter := &grid.GridDataIteration{CsvPath: dataPath}
	iterations := []simulator.Iteration{
		gridIter,
		&grid.ResidualDemandIteration{},
		&continuous.OrnsteinUhlenbeckIteration{},
		&grid.ImbalancePriceIteration{},
		&grid.CarbonDataIteration{CsvPath: carbonPath},
		&grid.CarbonThresholdDispatchIteration{},
		&grid.BatteryIteration{},
		&grid.BatteryDegradationIteration{},
	}
	run(settings, iterations, gridIter, outPath, steps)
}

func run(
	settings *simulator.Settings,
	iterations []simulator.Iteration,
	gridIter *grid.GridDataIteration,
	outPath string,
	steps int,
) {
	for i, iter := range iterations {
		iter.Configure(i, settings)
	}

	numSteps := steps
	if numSteps == 0 {
		numSteps = gridIter.DataLen() - 1
	}

	implementations := &simulator.Implementations{
		Iterations:      iterations,
		OutputCondition: &simulator.EveryStepOutputCondition{},
		OutputFunction:  simulator.NewJsonLogOutputFunction(outPath),
		TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
			MaxNumberOfSteps: numSteps,
		},
		TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 0.5},
	}

	log.Printf("Running simulation (%d steps) → %s", numSteps, outPath)
	simulator.NewPartitionCoordinator(settings, implementations).Run()
	log.Println("Done.")
}
