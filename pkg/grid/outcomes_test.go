package grid

import (
	"math"
	"testing"

	"github.com/umbralcalc/stochadex/pkg/continuous"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

func newOutcomesIterations() []simulator.Iteration {
	return []simulator.Iteration{
		&GridDataIteration{CsvPath: "testdata/demand_sample.csv"},
		&ResidualDemandIteration{},
		&continuous.OrnsteinUhlenbeckIteration{}, // price_noise
		&ImbalancePriceIteration{},
		&CarbonDataIteration{CsvPath: "testdata/carbon_sample.csv"},
		&BatteryIteration{},
		&BatteryDegradationIteration{},
		&RevenueIteration{},
		&CarbonSavingsIteration{},
	}
}

func TestRevenueIteration(t *testing.T) {
	t.Run(
		"test revenue accumulates after two steps",
		func(t *testing.T) {
			// price=50 £/MWh (constant, slope=0, intercept=50, sigmas=0)
			// battery dispatch_mw=100 MW, SoC=100 MWh, capacity=200 MWh
			// Step 1: battery discharges 100MW, actual_dispatch=100 (no limit: new_SoC=45.65>min=20)
			// Step 2: revenue reads step 1 battery state (actual_dispatch=100) and price(50)
			//         revenue = 100 * 50 * 0.5 = 2500 £
			settings := simulator.LoadSettingsFromYaml("./outcomes_settings.yaml")
			iterations := newOutcomesIterations()
			for i, iter := range iterations {
				iter.Configure(i, settings)
			}

			store := simulator.NewStateTimeStorage()
			implementations := &simulator.Implementations{
				Iterations:      iterations,
				OutputCondition: &simulator.EveryStepOutputCondition{},
				OutputFunction:  &simulator.StateTimeStorageOutputFunction{Store: store},
				TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
					MaxNumberOfSteps: 2,
				},
				TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 0.5},
			}
			simulator.NewPartitionCoordinator(settings, implementations).Run()

			revenueStates := store.GetValues("revenue")
			revenue := revenueStates[len(revenueStates)-1][0]
			expected := 100.0 * 50.0 * 0.5 // dispatch × price × dt
			if math.Abs(revenue-expected) > 0.01 {
				t.Errorf("expected revenue=%.2f £, got %.4f", expected, revenue)
			}
		},
	)

	t.Run(
		"test revenue runs with harnesses",
		func(t *testing.T) {
			settings := simulator.LoadSettingsFromYaml("./outcomes_settings.yaml")
			iterations := newOutcomesIterations()
			store := simulator.NewStateTimeStorage()
			implementations := &simulator.Implementations{
				Iterations:      iterations,
				OutputCondition: &simulator.EveryStepOutputCondition{},
				OutputFunction:  &simulator.StateTimeStorageOutputFunction{Store: store},
				TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
					MaxNumberOfSteps: 19,
				},
				TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 0.5},
			}
			if err := simulator.RunWithHarnesses(settings, implementations); err != nil {
				t.Errorf("test harness failed: %v", err)
			}
		},
	)
}

func TestCarbonSavingsIteration(t *testing.T) {
	t.Run(
		"test carbon savings accumulates after two steps",
		func(t *testing.T) {
			// battery dispatch_mw=100 MW (discharge), dt=0.5h
			// Step 1: actual_dispatch=100 MW
			// Step 2: carbon_savings reads step 1 battery state (dispatch=100)
			//         and step 1 carbon state.
			// At step 1, carbon_data reads CSV data[1] = 182 gCO₂/kWh (second row)
			// savings = max(100,0) * 0.5 * 182 / 1000 = 9.1 tCO₂
			settings := simulator.LoadSettingsFromYaml("./outcomes_settings.yaml")
			iterations := newOutcomesIterations()
			for i, iter := range iterations {
				iter.Configure(i, settings)
			}

			store := simulator.NewStateTimeStorage()
			implementations := &simulator.Implementations{
				Iterations:      iterations,
				OutputCondition: &simulator.EveryStepOutputCondition{},
				OutputFunction:  &simulator.StateTimeStorageOutputFunction{Store: store},
				TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
					MaxNumberOfSteps: 2,
				},
				TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 0.5},
			}
			simulator.NewPartitionCoordinator(settings, implementations).Run()

			carbonStates := store.GetValues("carbon_savings")
			saved := carbonStates[len(carbonStates)-1][0]
			// carbon at step 1 = CSV data[1] = 182 gCO₂/kWh
			expected := 100.0 * 0.5 * 182.0 / 1000.0
			if math.Abs(saved-expected) > 0.01 {
				t.Errorf("expected carbon savings=%.4f tCO₂, got %.4f", expected, saved)
			}
		},
	)

	t.Run(
		"test carbon savings runs with harnesses",
		func(t *testing.T) {
			settings := simulator.LoadSettingsFromYaml("./outcomes_settings.yaml")
			iterations := newOutcomesIterations()
			store := simulator.NewStateTimeStorage()
			implementations := &simulator.Implementations{
				Iterations:      iterations,
				OutputCondition: &simulator.EveryStepOutputCondition{},
				OutputFunction:  &simulator.StateTimeStorageOutputFunction{Store: store},
				TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
					MaxNumberOfSteps: 19,
				},
				TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 0.5},
			}
			if err := simulator.RunWithHarnesses(settings, implementations); err != nil {
				t.Errorf("test harness failed: %v", err)
			}
		},
	)
}
