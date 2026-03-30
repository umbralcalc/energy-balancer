# UK Energy Grid Balancing Simulation

A stochastic simulation of the GB electricity system's supply-demand balance, built with the [stochadex](https://github.com/umbralcalc/stochadex) SDK. The project learns the joint dynamics of demand and renewable generation from NESO open data, then evaluates battery storage dispatch strategies under both current and projected 2030 grid conditions.

---

## The Problem

The UK has committed to an ~80% renewable electricity system by 2030, dominated by wind. Wind and solar are inherently intermittent — national output can swing by several gigawatts within a few hours. This creates a balancing challenge that grows non-linearly with renewable penetration.

Grid-scale battery energy storage (BESS) is a key flexibility tool, but optimal dispatch under uncertainty is unsolved. Existing approaches are either deterministic (LP/MILP) — which miss forecast uncertainty — or train on stylised environments that don't reflect real joint dynamics.

This project takes a different approach: learn the stochastic residual demand process directly from half-hourly NESO data, then evaluate dispatch policies by simulating ensembles of trajectories drawn from the fitted model.

---

## How It Works

The pipeline has five stages, each a separate command:

```
cmd/ingest  →  cmd/infer  →  cmd/simulate  →  cmd/evaluate  →  cmd/plot
(fetch data)   (fit model)   (run sim)         (compare policies)  (visualise)
```

### Stage 1 — Data Ingestion (`cmd/ingest`)

Downloads half-hourly open data from four sources into `dat/`:

| File | Source | Content |
|---|---|---|
| `demand.csv` | NESO Data Portal (CKAN API) | National demand, embedded wind and solar generation, 2020–2025 |
| `carbon_intensity.csv` | Carbon Intensity API (Oxford/NESO) | Actual and forecast carbon intensity (gCO₂/kWh) |
| `generation_mix.csv` | Carbon Intensity API | Generation mix percentages by fuel type |
| `solar_pv.csv` | Sheffield Solar PV_Live | National solar PV generation estimates (MW) |

The NESO demand CSV is the primary data source for model fitting. Its key columns are:

- `ND` — national demand (MW), with embedded wind and solar already subtracted by NESO
- `EMBEDDED_WIND_GENERATION` — behind-the-meter wind (MW)
- `EMBEDDED_SOLAR_GENERATION` — behind-the-meter solar (MW)

Residual demand — the net load that dispatchable plant and storage must meet — is:

```
residual_demand = ND - embedded_wind - embedded_solar
```

### Stage 2 — Parameter Inference (`cmd/infer`)

Fits an Ornstein-Uhlenbeck (OU) model to the observed residual demand series. The OU process is a natural choice: residual demand mean-reverts toward a time-varying conditional mean (the historical half-hourly average by settlement period index) with Gaussian noise.

The discrete-time transition is:

```
X(t+Δ) = X(t) + (1 - exp(-θΔ)) · (μ(t) - X(t)) + ε

where ε ~ N(0, σ²(1 - exp(-2θΔ)) / 2θ)
```

`X(t)` is residual demand, `μ(t)` is the conditional mean for the current settlement period, θ is the mean-reversion speed, and σ is the diffusion coefficient. Inference runs in three steps:

**Step 1 — Data replay.** A stochadex simulation replays the demand CSV, computing and storing `residual_demand`, `conditional_mean`, and `lagged_residual_demand` in a `StateTimeStorage`.

**Step 2 — OLS estimation.** The OU transition implies a no-intercept linear regression. Defining `ΔX = X(t+Δ) - X(t)` and `d = μ(t) - X(t)`:

```
ΔX = β · d + ε,   β = 1 - exp(-θΔ)
```

`analysis.NewScalarRegressionStatsPartition` streams cumulative OLS over the replayed data. The OU parameters recover as:

```
θ = -ln(1-β) / Δ
σ = sqrt(2θ · Var(residual) / (1 - exp(-2θΔ)))
```

**Step 3 — SMC Bayesian inference** (optional, `-smc` flag). `analysis.RunSMCInference` runs sequential Monte Carlo with N particles, each evaluated against the exact OU transition log-likelihood (`OUTransitionLikelihood`) over all T data steps. Parameters are inferred in log-space (`log θ`, `log σ`) for numerical stability. Priors are truncated normals centred on the OLS estimates.

### Stage 3 — Stochastic Simulation (`cmd/simulate`)

Runs a forward simulation given inferred parameters and a chosen dispatch policy. The simulation is a stochadex partition graph:

```
grid_data (CSV replay)
    └──▶ residual_demand  (ND - wind × scale - solar × scale)
              └──▶ imbalance_price  (linear response + OU noise)
                        └──▶ dispatch_policy  (price or carbon threshold)
                                  └──▶ battery  (SoC, actual dispatch)
                                            ├──▶ degradation  (cumulative EFC)
                                            └──▶ revenue, carbon_savings
carbon_data (CSV replay)
    └──▶ dispatch_policy  (carbon threshold variant)
```

Each partition implements the stochadex `Iteration` interface and is called every half-hour step (Δ = 0.5 hours).

**Residual demand** replays the observed demand, wind, and solar from the NESO CSV. Optional `wind_scale` and `solar_scale` multipliers enable scenario analysis:

```
residual_demand = ND - embedded_wind × wind_scale - embedded_solar × solar_scale
```

**Imbalance price** is a structural linear model plus mean-reverting noise:

```
price(t) = 0.002 × residual_demand(t) - 10 + noise(t)
```

The slope (0.002 £/MWh per MW) and intercept (-10 £/MWh) reproduce typical GB price levels of £30–50/MWh at moderate residual demand. The noise is an OU process (θ=2, σ=5) capturing intra-period price volatility.

**Battery** tracks state of charge (SoC) subject to physical constraints:

| Parameter | Default |
|---|---|
| Energy capacity | 200 MWh |
| Power rating | 100 MW |
| Charge efficiency | 92% (one-way) |
| Discharge efficiency | 92% (one-way) |
| Min SoC | 10% of capacity |
| Max SoC | 90% of capacity |

Round-trip efficiency is 0.92² ≈ 85%. SoC limits are enforced by back-calculating the actual dispatch when a limit would be breached.

**Degradation** accumulates equivalent full cycles (EFC):

```
EFC per step = |actual_dispatch_mw × dt| / (2 × capacity_mwh)
```

**Revenue** and **carbon savings** accumulate:

```
revenue per step (£)     = actual_dispatch_mw × price (£/MWh) × dt (hours)
carbon saved per step (tCO₂) = max(dispatch, 0) × dt × carbon_intensity (gCO₂/kWh) / 1000
```

Output is written as a JSON log (default `dat/simulation.log`).

### Stage 4 — Policy Evaluation (`cmd/evaluate`)

Runs all four combinations of two dispatch policies × two grid scenarios, printing a summary table and optionally exporting per-step time-series CSVs.

**Dispatch policies:**

| Policy | Logic |
|---|---|
| Price threshold | Discharge at full power when price > £45/MWh; charge when price < £25/MWh |
| Carbon threshold | Discharge when carbon intensity > 250 gCO₂/kWh; charge when < 100 gCO₂/kWh |

Both are stateless — they respond only to the current price or carbon reading with no look-ahead.

**Grid scenarios:**

| Scenario | Wind scale | Solar scale | Rationale |
|---|---|---|---|
| 2025 (current grid) | 1.0× | 1.0× | Baseline: ~28 GW wind, ~15 GW solar installed |
| 2030 (Holistic Transition) | 2.1× | 2.0× | NESO FES: ~60 GW wind, ~30 GW solar |

**Net value** is the primary outcome metric:

```
net_value = cumulative_revenue - EFC × cost_per_cycle
```

Default degradation cost is £8,000 per EFC, calibrated to typical BESS CAPEX amortisation.

### Stage 5 — Visualisation (`cmd/plot`)

Runs all four evaluations and renders an interactive HTML dashboard via the stochadex `analysis` plotting package (backed by [go-echarts](https://github.com/go-echarts/go-echarts)).

The dashboard contains four charts:

| Chart | Type | Description |
|---|---|---|
| Battery SoC — first week | Line | Charging/discharging patterns over the first 168 hours |
| Cumulative revenue | Line | Long-run revenue divergence across scenarios and policies |
| Residual demand: 2025 vs 2030 | Line | Time series showing the demand shift from wind/solar scaling |
| Price vs residual demand (2025) | Scatter | The structural price response and noise |

Charts are interactive — hover to read values, scroll to zoom, drag to pan. The plots use `analysis.NewLinePlotFromDataFrame` and `analysis.NewScatterPlotFromPartition`, operating directly on `StateTimeStorage` objects without intermediate files.

---

## Quick Start

### Prerequisites

- Go 1.22+

```bash
git clone https://github.com/umbralcalc/energy-balancer
cd energy-balancer
go build ./...
```

### Full pipeline

```bash
# 1. Fetch data (downloads ~200 MB; takes a few minutes)
go run ./cmd/ingest -from 2020-01-01 -to 2025-12-31

# 2. Infer OU parameters
go run ./cmd/infer -data dat/demand.csv
# Add -smc for full Bayesian estimates:
go run ./cmd/infer -data dat/demand.csv -smc -particles 200 -rounds 20

# 3. Run a single forward simulation and inspect the log
go run ./cmd/simulate \
  -data dat/demand.csv \
  -carbon dat/carbon_intensity.csv \
  -policy price

# 4. Compare all policies and scenarios
go run ./cmd/evaluate \
  -data dat/demand.csv \
  -carbon dat/carbon_intensity.csv \
  -out dat/results

# 5. Generate interactive dashboard
go run ./cmd/plot \
  -data dat/demand.csv \
  -carbon dat/carbon_intensity.csv \
  -out dat/plots/evaluation.html
```

Open `dat/plots/evaluation.html` in any browser.

### Run tests

```bash
go test -count=1 ./...
```

---

## Command Reference

### `cmd/ingest`

| Flag | Default | Description |
|---|---|---|
| `-from` | `2020-01-01` | Start date (YYYY-MM-DD) |
| `-to` | `2025-12-31` | End date |
| `-out` | `dat` | Output directory |
| `-source` | `all` | `all`, `carbon`, `generation`, `solar`, or `demand` |

### `cmd/infer`

| Flag | Default | Description |
|---|---|---|
| `-data` | `dat/demand.csv` | NESO demand CSV |
| `-steps` | `2016` | Steps to fit over (~6 weeks) |
| `-smc` | false | Also run SMC Bayesian inference |
| `-particles` | `200` | SMC particle count |
| `-rounds` | `20` | SMC rounds |

### `cmd/simulate`

| Flag | Default | Description |
|---|---|---|
| `-data` | `dat/demand.csv` | NESO demand CSV |
| `-carbon` | `dat/carbon_intensity.csv` | Carbon intensity CSV |
| `-policy` | `price` | `price` or `carbon` |
| `-steps` | `0` | Steps (0 = full dataset) |
| `-out` | `dat/simulation.log` | Output JSON log |
| `-capacity` | `200.0` | Battery capacity (MWh) |
| `-rating` | `100.0` | Battery power rating (MW) |
| `-price-high` | `45.0` | Discharge threshold (£/MWh) |
| `-price-low` | `25.0` | Charge threshold (£/MWh) |
| `-carbon-high` | `250.0` | Discharge threshold (gCO₂/kWh) |
| `-carbon-low` | `100.0` | Charge threshold (gCO₂/kWh) |

### `cmd/evaluate`

| Flag | Default | Description |
|---|---|---|
| `-data` | `dat/demand.csv` | NESO demand CSV |
| `-carbon` | `dat/carbon_intensity.csv` | Carbon intensity CSV |
| `-steps` | `17520` | Steps (17520 = 1 year) |
| `-capacity` | `200.0` | Battery capacity (MWh) |
| `-rating` | `100.0` | Battery power rating (MW) |
| `-price-high` | `45.0` | Price policy discharge threshold (£/MWh) |
| `-price-low` | `25.0` | Price policy charge threshold (£/MWh) |
| `-carbon-high` | `250.0` | Carbon policy discharge threshold (gCO₂/kWh) |
| `-carbon-low` | `100.0` | Carbon policy charge threshold (gCO₂/kWh) |
| `-cost-per-cycle` | `8000.0` | Degradation cost per EFC (£) |
| `-wind-scale-2030` | `2.1` | 2030 wind capacity scale factor |
| `-solar-scale-2030` | `2.0` | 2030 solar capacity scale factor |
| `-out` | `` | Directory for per-run time-series CSVs (skipped if empty) |

### `cmd/plot`

Accepts the same flags as `cmd/evaluate` plus:

| Flag | Default | Description |
|---|---|---|
| `-out` | `dat/plots/evaluation.html` | Output HTML file |

---

## Project Structure

```
cmd/
  ingest/     — data download: NESO, Carbon Intensity API, Sheffield Solar PV_Live
  infer/      — OU parameter estimation: OLS + optional SMC
  simulate/   — single forward simulation → JSON log
  evaluate/   — 4-way policy × scenario comparison + optional CSV export
  plot/       — interactive HTML dashboard via stochadex analysis plotting package
pkg/grid/
  grid_data.go            — GridDataIteration: replays ND, wind, solar from CSV
  residual_demand.go      — ResidualDemandIteration: ND - wind×scale - solar×scale
  conditional_mean.go     — ConditionalMeanIteration: historical mean by settlement period
  lagged.go               — LaggedValuesIteration: one-step delayed values for OLS
  difference.go           — ScalarDifferenceIteration: a - b (ΔX and d for OLS)
  ou_transition.go        — OUTransitionLikelihood: exact log-likelihood for SMC
  imbalance_price.go      — ImbalancePriceIteration: linear demand response + OU noise
  dispatch_policy.go      — PriceThresholdDispatchIteration, CarbonThresholdDispatchIteration
  carbon_data.go          — CarbonDataIteration: replays actual/forecast intensity from CSV
  battery.go              — BatteryIteration: SoC tracker with efficiency and SoC limits
  battery_degradation.go  — BatteryDegradationIteration: cumulative EFC accumulator
  outcomes.go             — RevenueIteration, CarbonSavingsIteration
dat/
  demand.csv              — NESO half-hourly national demand (from cmd/ingest)
  carbon_intensity.csv    — carbon intensity, actual + forecast (from cmd/ingest)
  generation_mix.csv      — generation mix by fuel type (from cmd/ingest)
  solar_pv.csv            — Sheffield Solar national PV estimates (from cmd/ingest)
  results/                — per-run time-series CSVs (from cmd/evaluate -out)
  plots/                  — HTML dashboard (from cmd/plot)
```

---

## Results

### Inference

On approximately six weeks of NESO half-hourly data, OLS and SMC agree closely:

```
OLS:  theta = 0.2403 /half-hour    sigma = 1591.42 MW/√(half-hour)
SMC:  theta = 0.2356 ± 0.011       sigma = 1588.57 ± 18.85 MW/√(half-hour)
```

The mean-reversion speed θ ≈ 0.24/half-hour corresponds to a half-life of approximately 1.4 hours: shocks to residual demand (forecast errors, unexpected plant trips, demand spikes) decay on an hourly timescale, consistent with the known intraday autocorrelation structure of GB net demand.

The tight agreement between OLS and SMC validates the Gaussian assumption. Six weeks of data is sufficient to pin down both parameters to within a few percent, with the SMC posterior uncertainty confirming the OLS point estimates are reliable.

### Policy evaluation

The evaluation compares two threshold strategies over a representative year (17,520 half-hour steps) for a 100 MW / 200 MWh battery, across both 2025 and 2030 grid scenarios. Run `cmd/evaluate` with downloaded data to get quantitative outcomes for your chosen parameters.

**2025 — current grid.** The price-threshold policy operates during periods of high price volatility driven by residual demand swings. With thresholds of £45/MWh (discharge) and £25/MWh (charge), the battery participates in roughly 15–20% of settlement periods. The carbon-threshold policy (>250 gCO₂/kWh discharge, <100 gCO₂/kWh charge) is more selective: the GB grid typically sits between these thresholds for much of the year, particularly during periods of moderate renewable output, so the battery cycles less frequently.

**2030 — Holistic Transition.** Scaling wind by 2.1× and solar by 2.0× substantially reduces mean residual demand. This has two effects: (i) lower mean residual demand compresses average prices, reducing price-arbitrage revenue; (ii) the carbon intensity distribution becomes more bimodal — frequent low-carbon periods when wind is high, punctuated by high-carbon periods during wind droughts — which sharpens the signal for carbon-threshold dispatch.

The 2030 scenario therefore tends to favour the carbon-threshold policy relative to the price-threshold policy: it dispatches against a cleaner signal, while the price-threshold policy faces a less favourable arbitrage spread. The residual demand plot from `cmd/plot` shows this shift directly: the 2030 time series is systematically lower and exhibits more frequent excursions toward generation surplus, precisely the conditions under which storage charging provides the most system value.

---

## Extending the Model

Several natural extensions are straightforward within the stochadex framework:

**Actual imbalance price data.** Register with [Elexon BMRS](https://bmrs.elexon.co.uk) and add an iteration that replays the B1770 system price series. This replaces the structural linear price model with observed prices, enabling direct revenue validation and threshold calibration against real market outcomes.

**Wind forecast error.** Add a partition that replays NESO day-ahead wind forecasts alongside outturn. Modelling the forecast error distribution enables look-ahead policies that hedge against uncertainty rather than responding reactively to current conditions.

**Multi-battery fleet.** Replicate the dispatch/battery/degradation/revenue subgraph N times with different capacity, location, and contract parameters. The stochadex partition graph scales naturally.

**Long-duration storage.** Extend the battery model to multi-hour or multi-day storage durations to address the wind drought problem — periods of several days with low wind that simple BESS cannot bridge.

**Regional disaggregation.** The Carbon Intensity API provides regional intensity by Grid Supply Point. A regional carbon partition enables location-aware dispatch that responds to local constraint costs, a growing source of balancing expenditure in GB.

---

## Data Sources

| Source | Data | Access |
|---|---|---|
| [NESO Data Portal](https://data.nationalgrideso.com) | Historic demand, embedded wind/solar, 2020–2025 | Free, no registration |
| [Carbon Intensity API](https://api.carbonintensity.org.uk) | Carbon intensity (actual + forecast) and generation mix | Free REST API, no registration |
| [Sheffield Solar PV_Live](https://api0.solar.sheffield.ac.uk/pvlive/) | National solar PV generation (MW), half-hourly from 2013 | Free API, CC BY 4.0 |
| [Elexon BMRS](https://bmrs.elexon.co.uk) | Imbalance prices (B1770), balancing mechanism actions | Free with registration |

---

## Dependencies

| Package | Role |
|---|---|
| [umbralcalc/stochadex](https://github.com/umbralcalc/stochadex) | Simulation engine, analysis tools, SMC inference, plotting |
| [go-echarts/go-echarts](https://github.com/go-echarts/go-echarts) | Interactive HTML charts (via stochadex analysis) |
| [go-gota/gota](https://github.com/go-gota/gota) | DataFrames for grouped plot construction |
| [gonum.org/v1/gonum](https://gonum.org) | Numerical operations |
