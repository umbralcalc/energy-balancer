package custom

import (
	"encoding/csv"
	"os"
	"strconv"

	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// GridDataIteration replays historical grid state from the NESO demand CSV.
// Set CsvPath before calling Configure.
// State vector: [national_demand_mw, embedded_wind_mw, embedded_solar_mw]
type GridDataIteration struct {
	CsvPath string
	data    [][]float64
}

func (g *GridDataIteration) Configure(
	partitionIndex int,
	settings *simulator.Settings,
) {
	records, err := readCSV(g.CsvPath)
	if err != nil {
		panic("grid_data: " + err.Error())
	}

	// Find column indices
	header := records[0]
	ndIdx := colIndex(header, "ND")
	windIdx := colIndex(header, "EMBEDDED_WIND_GENERATION")
	solarIdx := colIndex(header, "EMBEDDED_SOLAR_GENERATION")

	g.data = make([][]float64, len(records)-1)
	for i, row := range records[1:] {
		nd := parseFloat(row[ndIdx])
		wind := parseFloat(row[windIdx])
		solar := parseFloat(row[solarIdx])
		g.data[i] = []float64{nd, wind, solar}
	}
}

func (g *GridDataIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	step := max(timestepsHistory.CurrentStepNumber, 0)
	if step >= len(g.data) {
		return g.data[len(g.data)-1]
	}
	return g.data[step]
}

// DataLen returns the number of data rows loaded from the CSV.
func (g *GridDataIteration) DataLen() int {
	return len(g.data)
}

func readCSV(path string) ([][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return csv.NewReader(f).ReadAll()
}

func colIndex(header []string, name string) int {
	for i, h := range header {
		if h == name {
			return i
		}
	}
	panic("grid_data: column not found: " + name)
}

func parseFloat(s string) float64 {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}
	return v
}


