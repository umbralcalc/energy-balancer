package main

import (
	"flag"
	"log"

	"github.com/umbralcalc/energy-balancer/pkg/custom"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

func main() {
	dataPath := flag.String("data", "dat/demand.csv", "Path to NESO demand CSV")
	steps := flag.Int("steps", 0, "Number of steps to simulate (0 = all available data)")
	outPath := flag.String("out", "dat/simulation.log", "Output JSON log path")
	flag.Parse()

	settings := &simulator.Settings{
		Iterations: []simulator.IterationSettings{
			{
				Name:   "grid_data",
				Params: simulator.NewParams(map[string][]float64{}),
				InitStateValues:   []float64{0, 0, 0},
				StateWidth:        3,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				Name: "residual_demand",
				Params: simulator.NewParams(map[string][]float64{
					"upstream_partition": {0},
				}),
				InitStateValues:   []float64{0},
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				Name: "smoothed_residual",
				Params: simulator.NewParams(map[string][]float64{
					"alpha":              {0.1},
					"upstream_partition": {1},
				}),
				InitStateValues:   []float64{0},
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
		},
		InitTimeValue:         0.0,
		TimestepsHistoryDepth: 2,
	}

	iterations := []simulator.Iteration{
		&custom.GridDataIteration{CsvPath: *dataPath},
		&custom.ResidualDemandIteration{},
		&custom.MovingAverageIteration{},
	}
	for i, iter := range iterations {
		iter.Configure(i, settings)
	}

	// Use all available data if steps not specified
	numSteps := *steps
	if numSteps == 0 {
		numSteps = iterations[0].(*custom.GridDataIteration).DataLen() - 1
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

	log.Printf("Running simulation: %d steps, output → %s", numSteps, *outPath)

	coordinator := simulator.NewPartitionCoordinator(settings, implementations)
	coordinator.Run()

	log.Println("Simulation complete.")
}
