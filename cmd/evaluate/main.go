package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/umbralcalc/energy-balancer/pkg/grid"
	"github.com/umbralcalc/stochadex/pkg/continuous"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// Partition indices — identical for both policy runs.
// The only difference between price and carbon policies is partition 5
// (dispatch_policy), which is either PriceThreshold or CarbonThreshold.
const (
	idxGridData      = 0
	idxResidual      = 1
	idxPriceNoise    = 2
	idxPrice         = 3
	idxCarbonData    = 4
	idxDispatch      = 5
	idxBattery       = 6
	idxDegradation   = 7
	idxRevenue       = 8
	idxCarbonSavings = 9
)

type policyResult struct {
	revenue     float64 // cumulative revenue in £
	carbonTco2  float64 // carbon displaced in tCO₂
	efc         float64 // equivalent full cycles
	netValue    float64 // revenue minus degradation cost
}

func buildSettings(
	priceHigh, priceLow, carbonHigh, carbonLow, ratingMW, capacityMWh float64,
	isPricePolicy bool,
) *simulator.Settings {
	var dispatchParams simulator.Params
	if isPricePolicy {
		dispatchParams = simulator.NewParams(map[string][]float64{
			"price_partition": {idxPrice},
			"price_high":      {priceHigh},
			"price_low":       {priceLow},
			"power_rating_mw": {ratingMW},
		})
	} else {
		dispatchParams = simulator.NewParams(map[string][]float64{
			"carbon_partition": {idxCarbonData},
			"carbon_high":      {carbonHigh},
			"carbon_low":       {carbonLow},
			"power_rating_mw":  {ratingMW},
		})
	}

	return &simulator.Settings{
		Iterations: []simulator.IterationSettings{
			{
				Name:              "grid_data",
				Params:            simulator.NewParams(map[string][]float64{}),
				InitStateValues:   []float64{22000, 1500, 0},
				StateWidth:        3,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				Name: "residual_demand",
				Params: simulator.NewParams(map[string][]float64{
					"upstream_partition": {idxGridData},
				}),
				InitStateValues:   []float64{20500},
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				Name: "price_noise",
				Params: simulator.NewParams(map[string][]float64{
					"thetas": {2.0},
					"mus":    {0.0},
					"sigmas": {5.0},
				}),
				InitStateValues:   []float64{0.0},
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              42,
			},
			{
				Name: "imbalance_price",
				Params: simulator.NewParams(map[string][]float64{
					"demand_slope":     {0.002},
					"demand_intercept": {-10.0},
					"demand_partition": {idxResidual},
					"noise_partition":  {idxPriceNoise},
				}),
				InitStateValues:   []float64{31.0},
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				Name:              "carbon_data",
				Params:            simulator.NewParams(map[string][]float64{}),
				InitStateValues:   []float64{180.0, 183.0},
				StateWidth:        2,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				Name:              "dispatch_policy",
				Params:            dispatchParams,
				InitStateValues:   []float64{0.0},
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				Name: "battery",
				Params: simulator.NewParams(map[string][]float64{
					"dispatch_mw":          {0.0},
					"energy_capacity_mwh":  {capacityMWh},
					"power_rating_mw":      {ratingMW},
					"charge_efficiency":    {0.92},
					"discharge_efficiency": {0.92},
					"min_soc_fraction":     {0.1},
					"max_soc_fraction":     {0.9},
				}),
				ParamsFromUpstream: map[string]simulator.UpstreamConfig{
					"dispatch_mw": {Upstream: idxDispatch},
				},
				InitStateValues:   []float64{capacityMWh * 0.5, 0},
				StateWidth:        2,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				Name: "degradation",
				Params: simulator.NewParams(map[string][]float64{
					"battery_partition":   {idxBattery},
					"energy_capacity_mwh": {capacityMWh},
				}),
				InitStateValues:   []float64{0.0},
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				Name: "revenue",
				Params: simulator.NewParams(map[string][]float64{
					"battery_partition": {idxBattery},
					"price_partition":   {idxPrice},
				}),
				InitStateValues:   []float64{0.0},
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{
				Name: "carbon_savings",
				Params: simulator.NewParams(map[string][]float64{
					"battery_partition": {idxBattery},
					"carbon_partition":  {idxCarbonData},
				}),
				InitStateValues:   []float64{0.0},
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
		},
		InitTimeValue:         0.0,
		TimestepsHistoryDepth: 2,
	}
}

func buildIterations(dataPath, carbonPath string, isPricePolicy bool) ([]simulator.Iteration, *grid.GridDataIteration) {
	gridIter := &grid.GridDataIteration{CsvPath: dataPath}
	var dispatch simulator.Iteration
	if isPricePolicy {
		dispatch = &grid.PriceThresholdDispatchIteration{}
	} else {
		dispatch = &grid.CarbonThresholdDispatchIteration{}
	}
	return []simulator.Iteration{
		gridIter,
		&grid.ResidualDemandIteration{},
		&continuous.OrnsteinUhlenbeckIteration{},
		&grid.ImbalancePriceIteration{},
		&grid.CarbonDataIteration{CsvPath: carbonPath},
		dispatch,
		&grid.BatteryIteration{},
		&grid.BatteryDegradationIteration{},
		&grid.RevenueIteration{},
		&grid.CarbonSavingsIteration{},
	}, gridIter
}

func runPolicy(
	settings *simulator.Settings,
	iterations []simulator.Iteration,
	numSteps int,
) *simulator.StateTimeStorage {
	for i, iter := range iterations {
		iter.Configure(i, settings)
	}
	store := simulator.NewStateTimeStorage()
	implementations := &simulator.Implementations{
		Iterations:      iterations,
		OutputCondition: &simulator.EveryStepOutputCondition{},
		OutputFunction:  &simulator.StateTimeStorageOutputFunction{Store: store},
		TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
			MaxNumberOfSteps: numSteps,
		},
		TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 0.5},
	}
	simulator.NewPartitionCoordinator(settings, implementations).Run()
	return store
}

func extractResult(store *simulator.StateTimeStorage, capacityMWh, costPerCycle float64) policyResult {
	last := func(name string) float64 {
		vals := store.GetValues(name)
		return vals[len(vals)-1][0]
	}
	efc := last("degradation")
	revenue := last("revenue")
	return policyResult{
		revenue:    revenue,
		carbonTco2: last("carbon_savings"),
		efc:        efc,
		netValue:   revenue - efc*costPerCycle,
	}
}

func main() {
	dataPath := flag.String("data", "dat/demand.csv", "Path to NESO demand CSV")
	carbonPath := flag.String("carbon", "dat/carbon_intensity.csv", "Path to carbon intensity CSV")
	steps := flag.Int("steps", 17520, "Steps to evaluate (default: 1 year = 17520 half-hours)")
	capacityMWh := flag.Float64("capacity", 200.0, "Battery energy capacity (MWh)")
	ratingMW := flag.Float64("rating", 100.0, "Battery power rating (MW)")
	priceHigh := flag.Float64("price-high", 45.0, "Price policy: discharge threshold £/MWh")
	priceLow := flag.Float64("price-low", 25.0, "Price policy: charge threshold £/MWh")
	carbonHigh := flag.Float64("carbon-high", 250.0, "Carbon policy: discharge threshold gCO₂/kWh")
	carbonLow := flag.Float64("carbon-low", 100.0, "Carbon policy: charge threshold gCO₂/kWh")
	costPerCycle := flag.Float64("cost-per-cycle", 8000.0, "Battery degradation cost per equivalent full cycle (£)")
	flag.Parse()

	log.Printf("Evaluating policies over %d steps (%.1f years)...",
		*steps, float64(*steps)/17520.0)

	log.Println("Running price-threshold policy...")
	priceSettings := buildSettings(
		*priceHigh, *priceLow, *carbonHigh, *carbonLow, *ratingMW, *capacityMWh, true)
	priceIters, priceGrid := buildIterations(*dataPath, *carbonPath, true)
	numSteps := *steps
	if numSteps == 0 {
		priceGrid.Configure(0, priceSettings)
		numSteps = priceGrid.DataLen() - 1
	}
	priceStore := runPolicy(priceSettings, priceIters, numSteps)
	priceResult := extractResult(priceStore, *capacityMWh, *costPerCycle)

	log.Println("Running carbon-threshold policy...")
	carbonSettings := buildSettings(
		*priceHigh, *priceLow, *carbonHigh, *carbonLow, *ratingMW, *capacityMWh, false)
	carbonIters, _ := buildIterations(*dataPath, *carbonPath, false)
	carbonStore := runPolicy(carbonSettings, carbonIters, numSteps)
	carbonResult := extractResult(carbonStore, *capacityMWh, *costPerCycle)

	fmt.Println()
	fmt.Printf("%-28s  %14s  %16s  %8s  %14s\n",
		"Policy", "Revenue (£)", "Carbon (tCO₂)", "EFC", "Net Value (£)")
	fmt.Printf("%-28s  %14s  %16s  %8s  %14s\n",
		"----------------------------", "--------------", "----------------", "--------", "--------------")
	printRow := func(name string, r policyResult) {
		fmt.Printf("%-28s  %14.2f  %16.2f  %8.2f  %14.2f\n",
			name, r.revenue, r.carbonTco2, r.efc, r.netValue)
	}
	printRow(fmt.Sprintf("price (>£%.0f discharge, <£%.0f charge)", *priceHigh, *priceLow), priceResult)
	printRow(fmt.Sprintf("carbon (>%vg discharge, <%vg charge)", int(*carbonHigh), int(*carbonLow)), carbonResult)

	fmt.Println()
	if priceResult.netValue > carbonResult.netValue {
		fmt.Printf("Price-threshold policy wins on net value by £%.2f\n",
			priceResult.netValue-carbonResult.netValue)
	} else {
		fmt.Printf("Carbon-threshold policy wins on net value by £%.2f\n",
			carbonResult.netValue-priceResult.netValue)
	}
	if carbonResult.carbonTco2 > priceResult.carbonTco2 {
		fmt.Printf("Carbon-threshold policy saves %.2f more tCO₂\n",
			carbonResult.carbonTco2-priceResult.carbonTco2)
	} else {
		fmt.Printf("Price-threshold policy saves %.2f more tCO₂\n",
			priceResult.carbonTco2-carbonResult.carbonTco2)
	}
}
