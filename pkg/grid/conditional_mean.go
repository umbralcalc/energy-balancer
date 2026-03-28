package grid

import (
	"fmt"
	"strings"
	"time"

	"github.com/umbralcalc/stochadex/pkg/simulator"
)

var shortMonths = map[string]time.Month{
	"JAN": time.January, "FEB": time.February, "MAR": time.March,
	"APR": time.April, "MAY": time.May, "JUN": time.June,
	"JUL": time.July, "AUG": time.August, "SEP": time.September,
	"OCT": time.October, "NOV": time.November, "DEC": time.December,
}

// ConditionalMeanIteration computes the mean residual demand conditioned on
// settlement period (1–48) and calendar month (1–12), then replays that
// conditional mean step-by-step in the same order as the demand CSV.
//
// This provides the deterministic seasonal "skeleton" for the OU residual
// demand model: the OU long-run mean (mus) varies by time of day and season.
//
// Set CsvPath before calling Configure.
// State: [conditional_mean_residual_demand_mw]
type ConditionalMeanIteration struct {
	CsvPath string
	// means[month-1][period-1] = mean residual demand
	means    [12][48]float64
	counts   [12][48]int
	// sequence of (month, period) indices for each row in the CSV
	sequence [][2]int
}

func (c *ConditionalMeanIteration) Configure(
	partitionIndex int,
	settings *simulator.Settings,
) {
	records, err := readCSV(c.CsvPath)
	if err != nil {
		panic("conditional_mean: " + err.Error())
	}

	header := records[0]
	dateIdx := colIndex(header, "SETTLEMENT_DATE")
	periodIdx := colIndex(header, "SETTLEMENT_PERIOD")
	ndIdx := colIndex(header, "ND")
	windIdx := colIndex(header, "EMBEDDED_WIND_GENERATION")
	solarIdx := colIndex(header, "EMBEDDED_SOLAR_GENERATION")

	// Reset all accumulated state before recomputing
	c.means = [12][48]float64{}
	c.counts = [12][48]int{}

	var sums [12][48]float64
	c.sequence = make([][2]int, len(records)-1)

	for i, row := range records[1:] {
		month, err := parseSettlementMonth(row[dateIdx])
		if err != nil {
			panic(fmt.Sprintf("conditional_mean: row %d: %v", i+1, err))
		}
		period := int(parseFloat(row[periodIdx]))
		if period < 1 || period > 48 {
			continue
		}
		m := int(month) - 1
		p := period - 1
		c.sequence[i] = [2]int{m, p}

		residual := parseFloat(row[ndIdx]) -
			parseFloat(row[windIdx]) -
			parseFloat(row[solarIdx])
		sums[m][p] += residual
		c.counts[m][p]++
	}

	// Second pass: compute means
	for m := range 12 {
		for p := range 48 {
			if c.counts[m][p] > 0 {
				c.means[m][p] = sums[m][p] / float64(c.counts[m][p])
			}
		}
	}
}

func (c *ConditionalMeanIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	step := max(timestepsHistory.CurrentStepNumber, 0)
	if step >= len(c.sequence) {
		step = len(c.sequence) - 1
	}
	idx := c.sequence[step]
	return []float64{c.means[idx[0]][idx[1]]}
}

// parseSettlementMonth extracts the calendar month from either:
//   - "DD-MON-YYYY" (e.g. "01-APR-2020")
//   - "YYYY-MM-DD"  (e.g. "2024-01-01")
func parseSettlementMonth(s string) (time.Month, error) {
	parts := strings.Split(s, "-")
	if len(parts) != 3 {
		return 0, fmt.Errorf("unrecognised date format %q", s)
	}
	// YYYY-MM-DD: first part is 4-char year
	if len(parts[0]) == 4 {
		t, err := time.Parse("2006-01-02", s)
		if err != nil {
			return 0, err
		}
		return t.Month(), nil
	}
	// DD-MON-YYYY
	mon, ok := shortMonths[strings.ToUpper(parts[1])]
	if !ok {
		return 0, fmt.Errorf("unrecognised month abbreviation %q", parts[1])
	}
	return mon, nil
}
