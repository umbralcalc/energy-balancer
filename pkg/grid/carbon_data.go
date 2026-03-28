package grid

import (
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// CarbonDataIteration replays historical carbon intensity data from the
// Carbon Intensity API CSV (columns: from, to, forecast_gco2_kwh,
// actual_gco2_kwh, index).
//
// State: [actual_gco2_kwh, forecast_gco2_kwh]
type CarbonDataIteration struct {
	CsvPath string
	data    [][]float64
}

func (c *CarbonDataIteration) Configure(
	partitionIndex int,
	settings *simulator.Settings,
) {
	records, err := readCSV(c.CsvPath)
	if err != nil {
		panic("carbon_data: " + err.Error())
	}

	header := records[0]
	actualIdx := colIndex(header, "actual_gco2_kwh")
	forecastIdx := colIndex(header, "forecast_gco2_kwh")

	c.data = make([][]float64, len(records)-1)
	for i, row := range records[1:] {
		actual := parseFloat(row[actualIdx])
		forecast := parseFloat(row[forecastIdx])
		c.data[i] = []float64{actual, forecast}
	}
}

func (c *CarbonDataIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	step := max(timestepsHistory.CurrentStepNumber, 0)
	if step >= len(c.data) {
		return c.data[len(c.data)-1]
	}
	return c.data[step]
}

// DataLen returns the number of data rows loaded from the CSV.
func (c *CarbonDataIteration) DataLen() int {
	return len(c.data)
}
