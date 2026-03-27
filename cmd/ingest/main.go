package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const (
	baseURL      = "https://api.carbonintensity.org.uk"
	maxRangeDays = 30
	timeLayout   = "2006-01-02T15:04Z"
)

// API response types for /intensity endpoint
type intensityResponse struct {
	Data []intensityRecord `json:"data"`
}

type intensityRecord struct {
	From      string         `json:"from"`
	To        string         `json:"to"`
	Intensity intensityValue `json:"intensity"`
}

type intensityValue struct {
	Forecast int    `json:"forecast"`
	Actual   int    `json:"actual"`
	Index    string `json:"index"`
}

// API response types for /generation endpoint
type generationResponse struct {
	Data []generationRecord `json:"data"`
}

type generationRecord struct {
	From          string    `json:"from"`
	To            string    `json:"to"`
	GenerationMix []fuelMix `json:"generationmix"`
}

type fuelMix struct {
	Fuel string  `json:"fuel"`
	Perc float64 `json:"perc"`
}

var fuelTypes = []string{
	"biomass", "coal", "imports", "gas", "nuclear",
	"other", "hydro", "solar", "wind",
}

func main() {
	fromStr := flag.String("from", "2020-01-01", "Start date (YYYY-MM-DD)")
	toStr := flag.String("to", "2025-12-31", "End date (YYYY-MM-DD)")
	outDir := flag.String("out", "dat", "Output directory for CSV files")
	flag.Parse()

	from, err := time.Parse("2006-01-02", *fromStr)
	if err != nil {
		log.Fatalf("invalid -from date: %v", err)
	}
	to, err := time.Parse("2006-01-02", *toStr)
	if err != nil {
		log.Fatalf("invalid -to date: %v", err)
	}
	if !to.After(from) {
		log.Fatal("-to must be after -from")
	}

	if err := os.MkdirAll(*outDir, 0755); err != nil {
		log.Fatalf("creating output dir: %v", err)
	}

	log.Printf("Fetching carbon intensity data from %s to %s", *fromStr, *toStr)
	if err := fetchIntensity(from, to, *outDir); err != nil {
		log.Fatalf("fetching intensity: %v", err)
	}

	log.Printf("Fetching generation mix data from %s to %s", *fromStr, *toStr)
	if err := fetchGeneration(from, to, *outDir); err != nil {
		log.Fatalf("fetching generation: %v", err)
	}

	log.Println("Done.")
}

// fetchIntensity pulls half-hourly carbon intensity data in 30-day chunks
// and writes it to a single CSV file.
func fetchIntensity(from, to time.Time, outDir string) error {
	path := filepath.Join(outDir, "carbon_intensity.csv")
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	w.Write([]string{"from", "to", "forecast_gco2_kwh", "actual_gco2_kwh", "index"})

	total := 0
	for chunkStart := from; chunkStart.Before(to); {
		chunkEnd := chunkStart.AddDate(0, 0, maxRangeDays)
		if chunkEnd.After(to) {
			chunkEnd = to
		}

		url := fmt.Sprintf("%s/intensity/%s/%s",
			baseURL,
			chunkStart.Format(timeLayout),
			chunkEnd.Format(timeLayout),
		)

		var resp intensityResponse
		if err := apiGet(url, &resp); err != nil {
			return fmt.Errorf("chunk %s–%s: %w",
				chunkStart.Format("2006-01-02"), chunkEnd.Format("2006-01-02"), err)
		}

		for _, r := range resp.Data {
			w.Write([]string{
				r.From, r.To,
				fmt.Sprintf("%d", r.Intensity.Forecast),
				fmt.Sprintf("%d", r.Intensity.Actual),
				r.Intensity.Index,
			})
		}
		total += len(resp.Data)
		log.Printf("  intensity: fetched %s to %s (%d records)",
			chunkStart.Format("2006-01-02"), chunkEnd.Format("2006-01-02"), len(resp.Data))

		chunkStart = chunkEnd
	}

	log.Printf("  intensity: %d total records → %s", total, path)
	return nil
}

// fetchGeneration pulls half-hourly generation mix data in 30-day chunks
// and writes it to a single CSV with one column per fuel type.
func fetchGeneration(from, to time.Time, outDir string) error {
	path := filepath.Join(outDir, "generation_mix.csv")
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	header := []string{"from", "to"}
	for _, fuel := range fuelTypes {
		header = append(header, fuel+"_pct")
	}
	w.Write(header)

	total := 0
	for chunkStart := from; chunkStart.Before(to); {
		chunkEnd := chunkStart.AddDate(0, 0, maxRangeDays)
		if chunkEnd.After(to) {
			chunkEnd = to
		}

		url := fmt.Sprintf("%s/generation/%s/%s",
			baseURL,
			chunkStart.Format(timeLayout),
			chunkEnd.Format(timeLayout),
		)

		var resp generationResponse
		if err := apiGet(url, &resp); err != nil {
			return fmt.Errorf("chunk %s–%s: %w",
				chunkStart.Format("2006-01-02"), chunkEnd.Format("2006-01-02"), err)
		}

		for _, r := range resp.Data {
			fuelMap := make(map[string]float64)
			for _, fm := range r.GenerationMix {
				fuelMap[fm.Fuel] = fm.Perc
			}
			row := []string{r.From, r.To}
			for _, fuel := range fuelTypes {
				row = append(row, fmt.Sprintf("%.1f", fuelMap[fuel]))
			}
			w.Write(row)
		}
		total += len(resp.Data)
		log.Printf("  generation: fetched %s to %s (%d records)",
			chunkStart.Format("2006-01-02"), chunkEnd.Format("2006-01-02"), len(resp.Data))

		chunkStart = chunkEnd
	}

	log.Printf("  generation: %d total records → %s", total, path)
	return nil
}

func apiGet(url string, target interface{}) error {
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("HTTP GET: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	return json.NewDecoder(resp.Body).Decode(target)
}
