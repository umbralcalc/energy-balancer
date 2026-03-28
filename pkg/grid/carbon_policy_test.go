package grid

import (
	"math"
	"testing"

	"github.com/umbralcalc/stochadex/pkg/continuous"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

func newCarbonPolicyIterations() []simulator.Iteration {
	return []simulator.Iteration{
		&GridDataIteration{CsvPath: "testdata/demand_sample.csv"},
		&ResidualDemandIteration{},
		&continuous.OrnsteinUhlenbeckIteration{}, // price_noise
		&ImbalancePriceIteration{},
		&CarbonDataIteration{CsvPath: "testdata/carbon_sample.csv"},
		&CarbonThresholdDispatchIteration{},
	}
}

func TestCarbonDataIteration(t *testing.T) {
	t.Run(
		"test carbon data replays actual intensity",
		func(t *testing.T) {
			// carbon_sample.csv first data row: actual=180, forecast=183
			settings := simulator.LoadSettingsFromYaml("./carbon_policy_settings.yaml")
			iterations := newCarbonPolicyIterations()
			for i, iter := range iterations {
				iter.Configure(i, settings)
			}

			store := simulator.NewStateTimeStorage()
			implementations := &simulator.Implementations{
				Iterations:      iterations,
				OutputCondition: &simulator.EveryStepOutputCondition{},
				OutputFunction:  &simulator.StateTimeStorageOutputFunction{Store: store},
				TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
					MaxNumberOfSteps: 1,
				},
				TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 0.5},
			}
			simulator.NewPartitionCoordinator(settings, implementations).Run()

			carbonStates := store.GetValues("carbon_data")
			actual := carbonStates[len(carbonStates)-1][0]
			// At step 1 CurrentStepNumber=1, so we get data[1] = second CSV row actual=182
			if math.Abs(actual-182.0) > 0.01 {
				t.Errorf("expected actual carbon intensity=182 gCO2/kWh, got %.2f", actual)
			}
		},
	)
}

func TestCarbonThresholdDispatchIteration(t *testing.T) {
	t.Run(
		"test discharge signal when carbon above threshold",
		func(t *testing.T) {
			// init carbon_data = [300.0, 290.0], carbon_high = 250.0
			// 300 > 250 → expect dispatch = +100 MW (discharge, displace gas)
			settings := simulator.LoadSettingsFromYaml("./carbon_policy_settings.yaml")
			iterations := newCarbonPolicyIterations()
			for i, iter := range iterations {
				iter.Configure(i, settings)
			}

			store := simulator.NewStateTimeStorage()
			implementations := &simulator.Implementations{
				Iterations:      iterations,
				OutputCondition: &simulator.EveryStepOutputCondition{},
				OutputFunction:  &simulator.StateTimeStorageOutputFunction{Store: store},
				TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
					MaxNumberOfSteps: 1,
				},
				TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 0.5},
			}
			simulator.NewPartitionCoordinator(settings, implementations).Run()

			dispatchStates := store.GetValues("dispatch_policy")
			dispatch := dispatchStates[len(dispatchStates)-1][0]
			if math.Abs(dispatch-100.0) > 1e-9 {
				t.Errorf("expected dispatch=+100 MW (discharge), got %.4f", dispatch)
			}
		},
	)

	t.Run(
		"test charge signal when carbon below threshold",
		func(t *testing.T) {
			// Set init carbon to 60 gCO2/kWh, carbon_low = 100
			// 60 < 100 → expect dispatch = -100 MW (charge, absorb clean energy)
			settings := simulator.LoadSettingsFromYaml("./carbon_policy_settings.yaml")
			settings.Iterations[4].InitStateValues = []float64{60.0, 55.0}
			iterations := newCarbonPolicyIterations()
			for i, iter := range iterations {
				iter.Configure(i, settings)
			}

			store := simulator.NewStateTimeStorage()
			implementations := &simulator.Implementations{
				Iterations:      iterations,
				OutputCondition: &simulator.EveryStepOutputCondition{},
				OutputFunction:  &simulator.StateTimeStorageOutputFunction{Store: store},
				TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
					MaxNumberOfSteps: 1,
				},
				TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 0.5},
			}
			simulator.NewPartitionCoordinator(settings, implementations).Run()

			dispatchStates := store.GetValues("dispatch_policy")
			dispatch := dispatchStates[len(dispatchStates)-1][0]
			if math.Abs(dispatch-(-100.0)) > 1e-9 {
				t.Errorf("expected dispatch=-100 MW (charge), got %.4f", dispatch)
			}
		},
	)

	t.Run(
		"test no dispatch when carbon between thresholds",
		func(t *testing.T) {
			// Set init carbon to 180 gCO2/kWh (between 100 and 250) → expect 0
			settings := simulator.LoadSettingsFromYaml("./carbon_policy_settings.yaml")
			settings.Iterations[4].InitStateValues = []float64{180.0, 183.0}
			iterations := newCarbonPolicyIterations()
			for i, iter := range iterations {
				iter.Configure(i, settings)
			}

			store := simulator.NewStateTimeStorage()
			implementations := &simulator.Implementations{
				Iterations:      iterations,
				OutputCondition: &simulator.EveryStepOutputCondition{},
				OutputFunction:  &simulator.StateTimeStorageOutputFunction{Store: store},
				TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
					MaxNumberOfSteps: 1,
				},
				TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 0.5},
			}
			simulator.NewPartitionCoordinator(settings, implementations).Run()

			dispatchStates := store.GetValues("dispatch_policy")
			dispatch := dispatchStates[len(dispatchStates)-1][0]
			if math.Abs(dispatch) > 1e-9 {
				t.Errorf("expected dispatch=0 MW, got %.4f", dispatch)
			}
		},
	)

	t.Run(
		"test carbon policy runs with harnesses",
		func(t *testing.T) {
			settings := simulator.LoadSettingsFromYaml("./carbon_policy_settings.yaml")
			iterations := newCarbonPolicyIterations()
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
