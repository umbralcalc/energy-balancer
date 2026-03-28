package grid

import (
	"math"
	"testing"

	"github.com/umbralcalc/stochadex/pkg/continuous"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

func TestBatteryIteration(t *testing.T) {
	settings := simulator.LoadSettingsFromYaml("./battery_settings.yaml")
	iterations := []simulator.Iteration{
		&GridDataIteration{CsvPath: "testdata/demand_sample.csv"},
		&ResidualDemandIteration{},
		&continuous.OrnsteinUhlenbeckIteration{}, // price_noise
		&ImbalancePriceIteration{},
		&BatteryIteration{},
	}
	for i, iter := range iterations {
		iter.Configure(i, settings)
	}

	t.Run(
		"test battery SoC stays within limits",
		func(t *testing.T) {
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
			coordinator := simulator.NewPartitionCoordinator(settings, implementations)
			coordinator.Run()

			capacity := 200.0
			minSoC := 0.1 * capacity
			maxSoC := 0.9 * capacity

			batteryStates := store.GetValues("battery")
			for i, state := range batteryStates {
				soc := state[0]
				if soc < minSoC-1e-6 || soc > maxSoC+1e-6 {
					t.Errorf("step %d: SoC %.2f outside [%.1f, %.1f] MWh", i, soc, minSoC, maxSoC)
				}
			}
		},
	)

	t.Run(
		"test battery discharges correctly without hitting limits",
		func(t *testing.T) {
			// Starting SoC=100 MWh, dispatch=50 MW, dt=0.5h, discharge_eff=0.92
			// energy_delta = -50 * 0.5 / 0.92 = -27.17 MWh
			// new SoC = 100 - 27.17 = 72.83 MWh
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
			coordinator := simulator.NewPartitionCoordinator(settings, implementations)
			coordinator.Run()

			batteryStates := store.GetValues("battery")
			lastSoC := batteryStates[len(batteryStates)-1][0]
			expectedSoC := 100.0 - 50.0*0.5/0.92
			if math.Abs(lastSoC-expectedSoC) > 0.01 {
				t.Errorf("expected SoC=%.4f, got %.4f", expectedSoC, lastSoC)
			}
		},
	)

	t.Run(
		"test battery and price run with harnesses",
		func(t *testing.T) {
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
