package grid

import (
	"testing"

	"github.com/umbralcalc/stochadex/pkg/general"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

func TestScalarDifferenceIteration_RunWithHarnesses(t *testing.T) {
	gen := simulator.NewConfigGenerator()
	gen.SetSimulation(&simulator.SimulationConfig{
		OutputCondition: &simulator.NilOutputCondition{},
		OutputFunction:  &simulator.NilOutputFunction{},
		TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
			MaxNumberOfSteps: 5,
		},
		TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 1.0},
		InitTimeValue:    0,
	})
	gen.SetPartition(&simulator.PartitionConfig{
		Name:              "src_a",
		Iteration:         &general.ConstantValuesIteration{},
		Params:            simulator.NewParams(map[string][]float64{}),
		InitStateValues:   []float64{10},
		StateHistoryDepth: 2,
		Seed:              1,
	})
	gen.SetPartition(&simulator.PartitionConfig{
		Name:              "src_b",
		Iteration:         &general.ConstantValuesIteration{},
		Params:            simulator.NewParams(map[string][]float64{}),
		InitStateValues:   []float64{4},
		StateHistoryDepth: 2,
		Seed:              2,
	})
	gen.SetPartition(&simulator.PartitionConfig{
		Name:      "diff",
		Iteration: &ScalarDifferenceIteration{},
		Params:    simulator.NewParams(map[string][]float64{}),
		ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
			"a": {Upstream: "src_a"},
			"b": {Upstream: "src_b"},
		},
		InitStateValues:   []float64{0},
		StateHistoryDepth: 2,
		Seed:              3,
	})
	settings, impl := gen.GenerateConfigs()
	if err := simulator.RunWithHarnesses(settings, impl); err != nil {
		t.Fatal(err)
	}
}

func TestLaggedValuesIteration_RunWithHarnesses(t *testing.T) {
	gen := simulator.NewConfigGenerator()
	gen.SetSimulation(&simulator.SimulationConfig{
		OutputCondition: &simulator.NilOutputCondition{},
		OutputFunction:  &simulator.NilOutputFunction{},
		TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
			MaxNumberOfSteps: 5,
		},
		TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 1.0},
		InitTimeValue:    0,
	})
	gen.SetPartition(&simulator.PartitionConfig{
		Name:              "src",
		Iteration:         &general.ConstantValuesIteration{},
		Params:            simulator.NewParams(map[string][]float64{}),
		InitStateValues:   []float64{7},
		StateHistoryDepth: 2,
		Seed:              1,
	})
	gen.SetPartition(&simulator.PartitionConfig{
		Name:      "lagged",
		Iteration: &LaggedValuesIteration{},
		Params:    simulator.NewParams(map[string][]float64{}),
		ParamsAsPartitions: map[string][]string{
			"source_partition": {"src"},
		},
		InitStateValues:   []float64{0},
		StateHistoryDepth: 2,
		Seed:              2,
	})
	settings, impl := gen.GenerateConfigs()
	if err := simulator.RunWithHarnesses(settings, impl); err != nil {
		t.Fatal(err)
	}
}
