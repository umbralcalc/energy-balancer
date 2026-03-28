package grid

import (
	"math"
	"testing"

	"github.com/umbralcalc/stochadex/pkg/continuous"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

func newDegradationIterations() []simulator.Iteration {
	return []simulator.Iteration{
		&GridDataIteration{CsvPath: "testdata/demand_sample.csv"},
		&ResidualDemandIteration{},
		&continuous.OrnsteinUhlenbeckIteration{}, // price_noise
		&ImbalancePriceIteration{},
		&BatteryIteration{},
		&BatteryDegradationIteration{},
	}
}

func TestBatteryDegradationIteration(t *testing.T) {
	t.Run(
		"test EFC accumulates after two steps",
		func(t *testing.T) {
			// dispatch_mw=50, dt=0.5, capacity=200
			// At step 1: battery reads dispatch=50, actual_dispatch=50 (no limit hit from SoC=100)
			// At step 2: degradation reads step 1's battery actual_dispatch=50
			// EFC = |50 * 0.5| / (2 * 200) = 25 / 400 = 0.0625
			settings := simulator.LoadSettingsFromYaml("./battery_degradation_settings.yaml")

			iterations := newDegradationIterations()
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

			degradationStates := store.GetValues("degradation")
			efc := degradationStates[len(degradationStates)-1][0]
			expected := math.Abs(50.0*0.5) / (2.0 * 200.0)
			if math.Abs(efc-expected) > 0.001 {
				t.Errorf("expected EFC=%.4f, got %.4f", expected, efc)
			}
		},
	)

	t.Run(
		"test degradation runs with harnesses",
		func(t *testing.T) {
			settings := simulator.LoadSettingsFromYaml("./battery_degradation_settings.yaml")
			iterations := newDegradationIterations()
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
