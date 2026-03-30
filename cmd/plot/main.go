package main

import (
	"flag"
	"log"
	"os"

	"github.com/go-echarts/go-echarts/v2/components"
	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	"github.com/umbralcalc/energy-balancer/pkg/grid"
	"github.com/umbralcalc/stochadex/pkg/analysis"
	"github.com/umbralcalc/stochadex/pkg/continuous"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// Partition indices — must match the order in buildSettings.
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

// weekSteps is the number of half-hourly steps in one week.
const weekSteps = 48 * 7

func buildSettings(
	priceHigh, priceLow, carbonHigh, carbonLow,
	ratingMW, capacityMWh, windScale, solarScale float64,
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
					"wind_scale":         {windScale},
					"solar_scale":        {solarScale},
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

func buildIterations(dataPath, carbonPath string, isPricePolicy bool) []simulator.Iteration {
	var dispatch simulator.Iteration
	if isPricePolicy {
		dispatch = &grid.PriceThresholdDispatchIteration{}
	} else {
		dispatch = &grid.CarbonThresholdDispatchIteration{}
	}
	return []simulator.Iteration{
		&grid.GridDataIteration{CsvPath: dataPath},
		&grid.ResidualDemandIteration{},
		&continuous.OrnsteinUhlenbeckIteration{},
		&grid.ImbalancePriceIteration{},
		&grid.CarbonDataIteration{CsvPath: carbonPath},
		dispatch,
		&grid.BatteryIteration{},
		&grid.BatteryDegradationIteration{},
		&grid.RevenueIteration{},
		&grid.CarbonSavingsIteration{},
	}
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

// runEntry pairs a human-readable label with its simulation output.
type runEntry struct {
	label string
	store *simulator.StateTimeStorage
}

// buildCombinedDF extracts one value index from a named partition across
// multiple stores and concatenates them into a DataFrame suitable for
// grouped line/scatter plots.
func buildCombinedDF(
	runs []runEntry,
	partition string,
	colIdx int,
	colName string,
	maxSteps int,
) dataframe.DataFrame {
	var times, vals []float64
	var labels []string
	for _, r := range runs {
		allVals := r.store.GetValues(partition)
		allTimes := r.store.GetTimes()
		n := len(allVals)
		if maxSteps > 0 && n > maxSteps {
			n = maxSteps
		}
		for i := 0; i < n; i++ {
			times = append(times, allTimes[i])
			vals = append(vals, allVals[i][colIdx])
			labels = append(labels, r.label)
		}
	}
	return dataframe.New(
		series.New(times, series.Float, "time_h"),
		series.New(vals, series.Float, colName),
		series.New(labels, series.String, "run"),
	)
}

func main() {
	dataPath := flag.String("data", "dat/demand.csv", "Path to NESO demand CSV")
	carbonPath := flag.String("carbon", "dat/carbon_intensity.csv", "Path to carbon intensity CSV")
	steps := flag.Int("steps", 17520, "Steps to evaluate (1 year = 17520 half-hours)")
	capacityMWh := flag.Float64("capacity", 200.0, "Battery energy capacity (MWh)")
	ratingMW := flag.Float64("rating", 100.0, "Battery power rating (MW)")
	priceHigh := flag.Float64("price-high", 45.0, "Price policy: discharge threshold £/MWh")
	priceLow := flag.Float64("price-low", 25.0, "Price policy: charge threshold £/MWh")
	carbonHigh := flag.Float64("carbon-high", 250.0, "Carbon policy: discharge threshold gCO₂/kWh")
	carbonLow := flag.Float64("carbon-low", 100.0, "Carbon policy: charge threshold gCO₂/kWh")
	windScale2030 := flag.Float64("wind-scale-2030", 2.1, "2030 scenario: wind capacity scale factor")
	solarScale2030 := flag.Float64("solar-scale-2030", 2.0, "2030 scenario: solar capacity scale factor")
	outPath := flag.String("out", "dat/plots/evaluation.html", "Output HTML file")
	flag.Parse()

	type scenario struct {
		label      string
		windScale  float64
		solarScale float64
	}
	scenarios := []scenario{
		{"2025 (current grid)", 1.0, 1.0},
		{"2030 (Holistic Transition)", *windScale2030, *solarScale2030},
	}

	// Determine step count from data length if steps == 0.
	numSteps := *steps
	if numSteps == 0 {
		probe := buildSettings(
			*priceHigh, *priceLow, *carbonHigh, *carbonLow,
			*ratingMW, *capacityMWh, 1.0, 1.0, true,
		)
		g := &grid.GridDataIteration{CsvPath: *dataPath}
		g.Configure(0, probe)
		numSteps = g.DataLen() - 1
	}

	log.Printf("Running 4 policy evaluations over %d steps...", numSteps)

	// Run all four scenario/policy combinations.
	var allRuns []runEntry
	var priceRuns []runEntry // for residual demand comparison (2025 vs 2030)
	for _, sc := range scenarios {
		log.Printf("  Scenario: %s", sc.label)

		for _, isPrice := range []bool{true, false} {
			policyName := "carbon threshold"
			if isPrice {
				policyName = "price threshold"
			}
			log.Printf("    Policy: %s", policyName)
			s := buildSettings(
				*priceHigh, *priceLow, *carbonHigh, *carbonLow,
				*ratingMW, *capacityMWh, sc.windScale, sc.solarScale, isPrice,
			)
			iters := buildIterations(*dataPath, *carbonPath, isPrice)
			store := runPolicy(s, iters, numSteps)
			label := policyName + " — " + sc.label
			entry := runEntry{label: label, store: store}
			allRuns = append(allRuns, entry)
			if isPrice {
				priceRuns = append(priceRuns, runEntry{label: sc.label, store: store})
			}
		}
	}

	// --- Plot 1: Battery SoC over the first week ---
	log.Println("Building plot: battery SoC (first week)...")
	socDF := buildCombinedDF(allRuns, "battery", 0, "soc_mwh", weekSteps)
	socPlot := analysis.NewLinePlotFromDataFrame(&socDF, "time_h", "soc_mwh", "run")

	// --- Plot 2: Cumulative revenue over the full evaluation window ---
	log.Println("Building plot: cumulative revenue...")
	revDF := buildCombinedDF(allRuns, "revenue", 0, "revenue_gbp", 0)
	revPlot := analysis.NewLinePlotFromDataFrame(&revDF, "time_h", "revenue_gbp", "run")

	// --- Plot 3: Residual demand — 2025 vs 2030 ---
	// Use only the price-threshold runs so the demand difference is purely
	// from the wind/solar scaling, not the dispatch policy.
	log.Println("Building plot: residual demand (2025 vs 2030)...")
	rdDF := buildCombinedDF(priceRuns, "residual_demand", 0, "residual_demand_mw", 0)
	rdPlot := analysis.NewLinePlotFromDataFrame(&rdDF, "time_h", "residual_demand_mw", "run")

	// --- Plot 4: Imbalance price vs residual demand — 2025 price-threshold run ---
	log.Println("Building plot: imbalance price vs residual demand (2025)...")
	price2025Store := priceRuns[0].store
	priceScatter := analysis.NewScatterPlotFromPartition(
		price2025Store,
		analysis.DataRef{
			PartitionName: "residual_demand",
			ValueIndices:  []int{0},
		},
		[]analysis.DataRef{
			{
				PartitionName: "imbalance_price",
				ValueIndices:  []int{0},
			},
		},
	)

	// Render all four charts into a single HTML page.
	if err := os.MkdirAll("dat/plots", 0o755); err != nil {
		log.Fatalf("creating output dir: %v", err)
	}
	page := components.NewPage()
	page.AddCharts(socPlot, revPlot, rdPlot, priceScatter)
	f, err := os.Create(*outPath)
	if err != nil {
		log.Fatalf("creating output file: %v", err)
	}
	defer f.Close()
	if err := page.Render(f); err != nil {
		log.Fatalf("rendering page: %v", err)
	}
	log.Printf("saved %s", *outPath)
}
