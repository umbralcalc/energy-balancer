package grid

import (
	"math"
	"testing"

	"github.com/umbralcalc/stochadex/pkg/continuous"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

func newDispatchPolicyIterations() []simulator.Iteration {
	return []simulator.Iteration{
		&GridDataIteration{CsvPath: "testdata/demand_sample.csv"},
		&ResidualDemandIteration{},
		&continuous.OrnsteinUhlenbeckIteration{}, // price_noise
		&ImbalancePriceIteration{},
		&PriceThresholdDispatchIteration{},
	}
}

func TestPriceThresholdDispatchIteration(t *testing.T) {
	t.Run(
		"test discharge signal when price above threshold",
		func(t *testing.T) {
			// init imbalance_price = 50.0 £/MWh, price_high = 45.0 £/MWh
			// At step 1, dispatch policy reads the init price (50.0) from history.
			// 50 > 45 → expect dispatch = +100 MW (full discharge)
			settings := simulator.LoadSettingsFromYaml("./dispatch_policy_settings.yaml")
			// Zero OU noise so price stays predictable
			settings.Iterations[2].Params.Map["sigmas"] = []float64{0.0}

			iterations := newDispatchPolicyIterations()
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
			expected := 100.0
			if math.Abs(dispatch-expected) > 1e-9 {
				t.Errorf("expected dispatch=%.1f MW (discharge), got %.4f", expected, dispatch)
			}
		},
	)

	t.Run(
		"test charge signal when price below threshold",
		func(t *testing.T) {
			// Set init price to 15.0 £/MWh, price_low = 25.0 £/MWh
			// 15 < 25 → expect dispatch = -100 MW (full charge)
			settings := simulator.LoadSettingsFromYaml("./dispatch_policy_settings.yaml")
			settings.Iterations[2].Params.Map["sigmas"] = []float64{0.0}
			settings.Iterations[3].InitStateValues = []float64{15.0}

			iterations := newDispatchPolicyIterations()
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
			expected := -100.0
			if math.Abs(dispatch-expected) > 1e-9 {
				t.Errorf("expected dispatch=%.1f MW (charge), got %.4f", expected, dispatch)
			}
		},
	)

	t.Run(
		"test no dispatch when price between thresholds",
		func(t *testing.T) {
			// Set init price to 35.0 £/MWh (between price_low=25 and price_high=45)
			// → expect dispatch = 0 MW
			settings := simulator.LoadSettingsFromYaml("./dispatch_policy_settings.yaml")
			settings.Iterations[2].Params.Map["sigmas"] = []float64{0.0}
			settings.Iterations[3].InitStateValues = []float64{35.0}

			iterations := newDispatchPolicyIterations()
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
		"test dispatch policy runs with harnesses",
		func(t *testing.T) {
			settings := simulator.LoadSettingsFromYaml("./dispatch_policy_settings.yaml")
			iterations := newDispatchPolicyIterations()
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
