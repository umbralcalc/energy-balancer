package main

import (
	"flag"
	"log"

	"github.com/umbralcalc/energy-balancer/pkg/grid"
	"github.com/umbralcalc/stochadex/pkg/continuous"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

func main() {
	dataPath := flag.String("data", "dat/demand.csv", "Path to NESO demand CSV")
	steps := flag.Int("steps", 0, "Number of steps to simulate (0 = all available data)")
	outPath := flag.String("out", "dat/simulation.log", "Output JSON log path")
	dispatchMW := flag.Float64("dispatch", 50.0, "Battery dispatch signal in MW (positive=discharge)")
	capacityMWh := flag.Float64("capacity", 200.0, "Battery energy capacity in MWh")
	ratingMW := flag.Float64("rating", 100.0, "Battery power rating in MW")
	flag.Parse()

	settings := &simulator.Settings{
		Iterations: []simulator.IterationSettings{
			{
				// index 0
				Name:              "grid_data",
				Params:            simulator.NewParams(map[string][]float64{}),
				InitStateValues:   []float64{22000, 1500, 0}, // typical overnight MW
				StateWidth:        3,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				// index 1
				Name: "residual_demand",
				Params: simulator.NewParams(map[string][]float64{
					"upstream_partition": {0},
				}),
				InitStateValues:   []float64{20500}, // 22000 - 1500 - 0
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				// index 2 — OU noise around structural price (mu=0)
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
				// index 3 — structural: slope 0.002 £/MWh/MW + intercept -10 £/MWh
				// gives ~31 £/MWh at 20.5 GW residual demand
				Name: "imbalance_price",
				Params: simulator.NewParams(map[string][]float64{
					"demand_slope":     {0.002},
					"demand_intercept": {-10.0},
					"demand_partition": {1},
					"noise_partition":  {2},
				}),
				InitStateValues:   []float64{31.0},
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				// index 4
				Name: "battery",
				Params: simulator.NewParams(map[string][]float64{
					"dispatch_mw":          {*dispatchMW},
					"energy_capacity_mwh":  {*capacityMWh},
					"power_rating_mw":      {*ratingMW},
					"charge_efficiency":    {0.92},
					"discharge_efficiency": {0.92},
					"min_soc_fraction":     {0.1},
					"max_soc_fraction":     {0.9},
				}),
				InitStateValues:   []float64{*capacityMWh * 0.5, 0},
				StateWidth:        2,
				StateHistoryDepth: 2,
				Seed:              0,
			},
		},
		InitTimeValue:         0.0,
		TimestepsHistoryDepth: 2,
	}

	iterations := []simulator.Iteration{
		&grid.GridDataIteration{CsvPath: *dataPath},
		&grid.ResidualDemandIteration{},
		&continuous.OrnsteinUhlenbeckIteration{},
		&grid.ImbalancePriceIteration{},
		&grid.BatteryIteration{},
	}
	for i, iter := range iterations {
		iter.Configure(i, settings)
	}

	numSteps := *steps
	if numSteps == 0 {
		numSteps = iterations[0].(*grid.GridDataIteration).DataLen() - 1
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

	log.Printf("Running simulation: %d steps → %s", numSteps, *outPath)
	simulator.NewPartitionCoordinator(settings, implementations).Run()
	log.Println("Done.")
}
