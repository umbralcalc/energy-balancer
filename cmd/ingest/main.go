package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

const (
	carbonBaseURL = "https://api.carbonintensity.org.uk"
	pvLiveBaseURL = "https://api0.solar.sheffield.ac.uk/pvlive/api/v4"
	nesoBaseURL   = "https://api.neso.energy/api/3/action"
	maxRangeDays  = 30
	timeLayout    = "2006-01-02T15:04Z"
)

// Carbon Intensity API types
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

// PV_Live API types
type pvLiveResponse struct {
	Data [][]interface{} `json:"data"`
	Meta []string        `json:"meta"`
}

// NESO CKAN API types
type nesoResponse struct {
	Result nesoResult `json:"result"`
}

type nesoResult struct {
	Records []map[string]interface{} `json:"records"`
	Total   int                      `json:"total"`
}

// NESO demand resource IDs by year
var nesoDemandResources = map[int]string{
	2020: "33ba6857-2a55-479f-9308-e5c4c53d4381",
	2021: "18c69c42-f20d-46f0-84e9-e279045befc6",
	2022: "bb44a1b5-75b1-4db2-8491-257f23385006",
	2023: "bf5ab335-9b40-4ea4-b93a-ab4af7bce003",
	2024: "f6d02c0f-957b-48cb-82ee-09003f2ba759",
	2025: "b2bde559-3455-4021-b179-dfe60c0337b0",
}

var demandFields = []string{
	"SETTLEMENT_DATE", "SETTLEMENT_PERIOD",
	"ND", "TSD", "ENGLAND_WALES_DEMAND",
	"EMBEDDED_WIND_GENERATION", "EMBEDDED_WIND_CAPACITY",
	"EMBEDDED_SOLAR_GENERATION", "EMBEDDED_SOLAR_CAPACITY",
	"NON_BM_STOR", "PUMP_STORAGE_PUMPING",
	"IFA_FLOW", "IFA2_FLOW", "BRITNED_FLOW", "MOYLE_FLOW",
	"EAST_WEST_FLOW", "NEMO_FLOW", "NSL_FLOW", "ELECLINK_FLOW",
}

func main() {
	fromStr := flag.String("from", "2020-01-01", "Start date (YYYY-MM-DD)")
	toStr := flag.String("to", "2025-12-31", "End date (YYYY-MM-DD)")
	outDir := flag.String("out", "dat", "Output directory for CSV files")
	source := flag.String("source", "all", "Data source to fetch: all, carbon, generation, solar, demand")
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

	all := *source == "all"

	if all || *source == "carbon" {
		log.Printf("Fetching carbon intensity data from %s to %s", *fromStr, *toStr)
		if err := fetchIntensity(from, to, *outDir); err != nil {
			log.Fatalf("fetching intensity: %v", err)
		}
	}

	if all || *source == "generation" {
		log.Printf("Fetching generation mix data from %s to %s", *fromStr, *toStr)
		if err := fetchGeneration(from, to, *outDir); err != nil {
			log.Fatalf("fetching generation: %v", err)
		}
	}

	if all || *source == "solar" {
		log.Printf("Fetching solar PV generation from %s to %s", *fromStr, *toStr)
		if err := fetchSolarPV(from, to, *outDir); err != nil {
			log.Fatalf("fetching solar PV: %v", err)
		}
	}

	if all || *source == "demand" {
		log.Printf("Fetching demand data from %s to %s", *fromStr, *toStr)
		if err := fetchDemand(from, to, *outDir); err != nil {
			log.Fatalf("fetching demand: %v", err)
		}
	}

	log.Println("Done.")
}

// fetchIntensity pulls half-hourly carbon intensity data in 30-day chunks.
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

		u := fmt.Sprintf("%s/intensity/%s/%s",
			carbonBaseURL,
			chunkStart.Format(timeLayout),
			chunkEnd.Format(timeLayout),
		)

		var resp intensityResponse
		if err := apiGet(u, &resp); err != nil {
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

// fetchGeneration pulls half-hourly generation mix data in 30-day chunks.
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

		u := fmt.Sprintf("%s/generation/%s/%s",
			carbonBaseURL,
			chunkStart.Format(timeLayout),
			chunkEnd.Format(timeLayout),
		)

		var resp generationResponse
		if err := apiGet(u, &resp); err != nil {
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

// fetchSolarPV pulls national half-hourly solar PV generation from Sheffield Solar PV_Live
// in 1-year chunks.
func fetchSolarPV(from, to time.Time, outDir string) error {
	path := filepath.Join(outDir, "solar_pv.csv")
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	w.Write([]string{"datetime_gmt", "generation_mw"})

	total := 0
	pvTimeLayout := "2006-01-02T15:04:05"
	for chunkStart := from; chunkStart.Before(to); {
		chunkEnd := chunkStart.AddDate(1, 0, 0)
		if chunkEnd.After(to) {
			chunkEnd = to
		}

		u := fmt.Sprintf("%s/pes/0?start=%s&end=%s",
			pvLiveBaseURL,
			chunkStart.Format(pvTimeLayout),
			chunkEnd.Format(pvTimeLayout),
		)

		var resp pvLiveResponse
		if err := apiGet(u, &resp); err != nil {
			return fmt.Errorf("chunk %s–%s: %w",
				chunkStart.Format("2006-01-02"), chunkEnd.Format("2006-01-02"), err)
		}

		// PV_Live returns newest first; reverse for chronological order
		for i := len(resp.Data) - 1; i >= 0; i-- {
			row := resp.Data[i]
			if len(row) < 3 {
				continue
			}
			datetime, _ := row[1].(string)
			genMW := "0"
			if row[2] != nil {
				genMW = fmt.Sprintf("%.2f", row[2].(float64))
			}
			w.Write([]string{datetime, genMW})
		}
		total += len(resp.Data)
		log.Printf("  solar: fetched %s to %s (%d records)",
			chunkStart.Format("2006-01-02"), chunkEnd.Format("2006-01-02"), len(resp.Data))

		chunkStart = chunkEnd
	}

	log.Printf("  solar: %d total records → %s", total, path)
	return nil
}

// fetchDemand pulls half-hourly demand data from NESO historic demand archives,
// one year at a time via the CKAN datastore API.
func fetchDemand(from, to time.Time, outDir string) error {
	path := filepath.Join(outDir, "demand.csv")
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	w.Write(demandFields)

	total := 0
	for year := from.Year(); year <= to.Year(); year++ {
		resourceID, ok := nesoDemandResources[year]
		if !ok {
			log.Printf("  demand: no resource ID for year %d, skipping", year)
			continue
		}

		offset := 0
		batchSize := 10000
		yearTotal := 0
		for {
			u := fmt.Sprintf("%s/datastore_search?resource_id=%s&limit=%d&offset=%d&sort=%s",
				nesoBaseURL, resourceID, batchSize, offset,
				url.QueryEscape("SETTLEMENT_DATE asc, SETTLEMENT_PERIOD asc"),
			)

			var resp nesoResponse
			if err := apiGet(u, &resp); err != nil {
				return fmt.Errorf("year %d offset %d: %w", year, offset, err)
			}

			for _, rec := range resp.Result.Records {
				row := make([]string, len(demandFields))
				for i, field := range demandFields {
					row[i] = formatNesoField(rec[field])
				}
				w.Write(row)
			}

			yearTotal += len(resp.Result.Records)
			if len(resp.Result.Records) < batchSize {
				break
			}
			offset += batchSize
		}

		total += yearTotal
		log.Printf("  demand: fetched %d (%d records)", year, yearTotal)
	}

	log.Printf("  demand: %d total records → %s", total, path)
	return nil
}

func formatNesoField(v interface{}) string {
	if v == nil {
		return ""
	}
	switch val := v.(type) {
	case float64:
		return strconv.FormatFloat(val, 'f', -1, 64)
	case string:
		return val
	default:
		return fmt.Sprintf("%v", val)
	}
}

func apiGet(u string, target interface{}) error {
	resp, err := http.Get(u)
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
