package grid

import (
	"math"
	"testing"

	"github.com/umbralcalc/stochadex/pkg/simulator"
)

const csvPath = "testdata/demand_sample.csv"

func newConditionalMeanIterations() []simulator.Iteration {
	return []simulator.Iteration{
		&GridDataIteration{CsvPath: csvPath},
		&ResidualDemandIteration{},
		&ConditionalMeanIteration{CsvPath: csvPath},
	}
}

func TestConditionalMeanIteration(t *testing.T) {
	t.Run(
		"test that conditional mean is within plausible range",
		func(t *testing.T) {
			settings := simulator.LoadSettingsFromYaml("./conditional_mean_settings.yaml")
			iterations := newConditionalMeanIterations()
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
				TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 0.5},
			}
			simulator.NewPartitionCoordinator(settings, implementations).Run()

			// With 20 rows all from January, conditional means should be
			// non-zero and in a plausible MW range (5000–40000 MW)
			condMeanStates := store.GetValues("conditional_mean")
			for i, state := range condMeanStates {
				cm := state[0]
				if math.IsNaN(cm) || cm < 5000 || cm > 40000 {
					t.Errorf("step %d: conditional mean %.1f MW outside plausible range", i, cm)
				}
			}
		},
	)

	t.Run(
		"test that conditional mean runs with harnesses",
		func(t *testing.T) {
			settings := simulator.LoadSettingsFromYaml("./conditional_mean_settings.yaml")
			iterations := newConditionalMeanIterations()
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
