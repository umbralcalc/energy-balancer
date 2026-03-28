package custom

import (
	"math"
	"testing"

	"github.com/umbralcalc/stochadex/pkg/simulator"
)

func TestGridDataAndResidualDemand(t *testing.T) {
	t.Run(
		"test that grid data replays CSV values correctly",
		func(t *testing.T) {
			settings := simulator.LoadSettingsFromYaml(
				"./grid_data_settings.yaml",
			)
			iterations := []simulator.Iteration{
				&GridDataIteration{CsvPath: "testdata/demand_sample.csv"},
				&ResidualDemandIteration{},
			}
			for i, iter := range iterations {
				iter.Configure(i, settings)
			}
			store := simulator.NewStateTimeStorage()
			implementations := &simulator.Implementations{
				Iterations:      iterations,
				OutputCondition: &simulator.EveryStepOutputCondition{},
				OutputFunction:  &simulator.StateTimeStorageOutputFunction{Store: store},
				TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
					MaxNumberOfSteps: 19,
				},
				TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 1.0},
			}
			coordinator := simulator.NewPartitionCoordinator(
				settings,
				implementations,
			)
			coordinator.Run()

			// Check the last step: settlement period 20
			// ND=28394, wind=2193, solar=1589
			gridState := store.GetValues("grid_data")
			lastGrid := gridState[len(gridState)-1]
			if math.Abs(lastGrid[0]-28394.0) > 1.0 {
				t.Errorf("expected ND=28394, got %.1f", lastGrid[0])
			}
			if math.Abs(lastGrid[1]-2193.0) > 1.0 {
				t.Errorf("expected wind=2193, got %.1f", lastGrid[1])
			}
			if math.Abs(lastGrid[2]-1589.0) > 1.0 {
				t.Errorf("expected solar=1589, got %.1f", lastGrid[2])
			}

			// Residual demand reads upstream state from the previous step
			// (period 19: ND=28145, wind=2223, solar=1421)
			residState := store.GetValues("residual_demand")
			lastResid := residState[len(residState)-1]
			expectedResidual := 28145.0 - 2223.0 - 1421.0
			if math.Abs(lastResid[0]-expectedResidual) > 1.0 {
				t.Errorf("expected residual=%.1f, got %.1f", expectedResidual, lastResid[0])
			}
		},
	)
	t.Run(
		"test that grid data and residual demand run with harnesses",
		func(t *testing.T) {
			settings := simulator.LoadSettingsFromYaml(
				"./grid_data_settings.yaml",
			)
			iterations := []simulator.Iteration{
				&GridDataIteration{CsvPath: "testdata/demand_sample.csv"},
				&ResidualDemandIteration{},
			}
			store := simulator.NewStateTimeStorage()
			implementations := &simulator.Implementations{
				Iterations:      iterations,
				OutputCondition: &simulator.EveryStepOutputCondition{},
				OutputFunction:  &simulator.StateTimeStorageOutputFunction{Store: store},
				TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
					MaxNumberOfSteps: 19,
				},
				TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 1.0},
			}
			if err := simulator.RunWithHarnesses(settings, implementations); err != nil {
				t.Errorf("test harness failed: %v", err)
			}
		},
	)
}
