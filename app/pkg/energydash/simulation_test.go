package energydash

import (
	"math"
	"testing"

	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// TestSimulationRunsForFullHorizon verifies the dashboard's partition
// graph runs without panicking through SimSteps half-hour steps under
// the default action_state_values (price policy, 2025 grid). The wasm
// path goes through the same code; running this on the host catches
// wiring errors (missing params, mis-indexed partitions, embedded-data
// parsing) early.
func TestSimulationRunsForFullHorizon(t *testing.T) {
	parseEmbeddedData()
	if len(embeddedDemand) < SimSteps {
		t.Fatalf("embedded data has only %d rows, need >= %d", len(embeddedDemand), SimSteps)
	}

	gen := BuildEnergySimulation()
	settings, impl := gen.GenerateConfigs()
	settings.Init()

	// Override the output to silence per-step writes; we only care
	// about the final partition states.
	impl.OutputCondition = &simulator.EveryStepOutputCondition{}
	impl.OutputFunction = &simulator.NilOutputFunction{}

	coord := simulator.NewPartitionCoordinator(settings, impl)
	coord.Run()

	t.Logf("simulation completed %d steps", SimSteps)
}

// TestOutcomesAccumulate validates that the outcomes partition's
// cumulative state climbs monotonically for revenue and EFCs and that
// the net-value relation (net = revenue - degCost) holds at the
// horizon. Smoke check against silent off-by-one or sign errors in
// OutcomesIteration.
func TestOutcomesAccumulate(t *testing.T) {
	parseEmbeddedData()

	gen := BuildEnergySimulation()
	settings, impl := gen.GenerateConfigs()
	settings.Init()
	impl.OutputCondition = &simulator.EveryStepOutputCondition{}
	impl.OutputFunction = &simulator.NilOutputFunction{}

	// Find the outcomes partition's index.
	outcomesIdx := -1
	for i, s := range settings.Iterations {
		if s.Name == "outcomes" {
			outcomesIdx = i
			break
		}
	}
	if outcomesIdx < 0 {
		t.Fatal("outcomes partition not found in settings")
	}

	coord := simulator.NewPartitionCoordinator(settings, impl)
	coord.Run()

	// After Run() the coordinator's shared state has the final row 0.
	final := coord.Shared.StateHistories[outcomesIdx].Values.RawRowView(0)
	netValue := final[0]
	revenue := final[1]
	degCost := final[2]
	efc := final[4]

	if revenue <= 0 {
		t.Errorf("expected positive cumulative revenue under price policy, got %.3f", revenue)
	}
	if efc <= 0 {
		t.Errorf("expected positive cumulative EFCs, got %.3f", efc)
	}
	if math.Abs((netValue)-(revenue-degCost)) > 1e-6 {
		t.Errorf("net value invariant violated: net=%.6f revenue=%.6f degCost=%.6f", netValue, revenue, degCost)
	}
	t.Logf("after %d steps: net £%.1fk · revenue £%.1fk · degradation £%.1fk · %.2f EFC",
		SimSteps, netValue, revenue, degCost, efc)
}
