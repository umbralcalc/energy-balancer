# UK Energy Grid Balancing Simulation: Project Plan

## Applying the Stochadex to Storage Dispatch and Demand Response Optimisation

---

## Overview

Build a stochastic simulation of the GB electricity system's supply-demand balance under increasing renewable penetration, learned from freely available generation, demand, and pricing data, with a decision science layer to evaluate and optimise battery storage dispatch and demand-side response strategies.

The core question: **given the current grid state and stochastic forecasts for wind, solar and demand, what storage dispatch and demand response schedule minimises expected balancing cost and carbon intensity over the next 24–48 hours?**

---

## Why This Problem

- The UK has committed to a net-zero electricity system, with renewables expected to supply around 80% of total electricity by 2030, dominated by wind (~70% of renewables).
- Wind and solar output are inherently stochastic — generation can swing by gigawatts within hours depending on weather. This intermittency creates balancing challenges that grow non-linearly with renewable penetration.
- Grid-scale battery energy storage (BESS) capacity is growing rapidly in GB, with projects ranging from 98MW/196MWh (Yorkshire, linked to Dogger Bank offshore wind) to numerous smaller installations. BESS provides fast-response flexibility, but optimal dispatch under uncertainty is an unsolved operational problem.
- The National Energy System Operator (NESO) spends billions annually on balancing services. Better dispatch strategies could reduce costs, cut carbon emissions from gas peaking plants, and reduce renewable curtailment.
- Existing dispatch optimisation tools are predominantly deterministic (linear/mixed-integer programming) or use simplified stochastic models. They typically don't learn the joint stochastic dynamics of wind, solar, demand, and price from historical data — they assume them.

---

## The Gap This Fills

| Approach | Examples | Limitation |
|----------|----------|------------|
| Deterministic dispatch (LP/MILP) | PyPSA, standard unit commitment models | Don't propagate forecast uncertainty; can overestimate battery value by ~15% with relaxed formulations |
| Stochastic programming | Two-stage scenario-based models (SDED-S) | Generate scenarios synthetically rather than learning joint dynamics from data; often computationally expensive |
| RL-based dispatch | Various DQN/PPO battery controllers | Train on simplified environments, struggle with non-stationarity and multi-service stacking |
| Market price forecasting | ARIMA, LSTM price predictors | Predict prices but don't simulate the physical system that generates them; can't evaluate policy counterfactuals |

**The stochadex differentiator:** a generalised stochastic simulation that learns the joint dynamics of wind generation, solar generation, demand, and imbalance prices from years of half-hourly NESO/Elexon data, then uses the decision science layer to evaluate storage dispatch policies under realistic forecast uncertainty. Same proven pattern as AMR, flood risk, rugby, fishing — ingest freely available data, build a simulation that learns from it, optimise actions.

---

## Phase 1: Data Ingestion

### 1.1 Generation mix and carbon intensity

**Source: NESO Data Portal (formerly National Grid ESO)**

- Historic generation mix and carbon intensity at half-hourly resolution
- Generation by fuel type: gas, coal, nuclear, wind (national + embedded), solar, hydro, biomass, interconnectors, other (including BESS)
- Day-ahead wind and solar forecasts at half-hourly resolution (within-day to 14 days ahead)
- Weekly wind generator availability at MW level
- Balancing costs and system prices

**Portal:** `data.nationalgrideso.com`

**Source: Carbon Intensity API (NESO + University of Oxford)**

- National and regional carbon intensity (gCO₂/kWh) at half-hourly resolution
- Generation mix percentages by fuel type per region and nationally
- Forecast carbon intensity 96+ hours ahead
- Free API, no registration required

**API base:** `api.carbonintensity.org.uk`

```
# National intensity for a date range
GET /intensity/{from}/{to}

# Regional generation mix for current half hour
GET /regional

# Intensity by postcode
GET /regional/postcode/{postcode}
```

### 1.2 Solar PV generation

**Source: Sheffield Solar PV_Live**

- Half-hourly PV generation estimates for the entire GB network (embedded, non-BM solar)
- National and regional (by Grid Supply Point or DNO area) breakdowns
- Historical data from January 2013 onwards
- Licensed under CC BY 4.0
- Python API library available (`pvlive-api` on PyPI)

**API:** `api0.solar.sheffield.ac.uk/pvlive/`

### 1.3 Demand data

**Source: NESO Data Portal — Historic Demand Data**

- Half-hourly national demand outturn and forecasts
- Demand profiles and profile dates for forecast construction
- Note: NESO "demand" = true demand minus embedded wind and solar (an important subtlety for modelling)

### 1.4 Wholesale market and balancing data

**Source: Elexon BMRS (Balancing Mechanism Reporting Service)**

- System imbalance prices (reports B1770) and volumes (B1780) at half-hourly resolution
- Generation by fuel type (actual)
- Balancing Mechanism actions (bids, offers, acceptances) at BMU level
- Final Physical Notifications (FPN), Maximum Export/Import Limits
- Free API access with registration (scripting key)

**API:** `bmrs.elexon.co.uk/api-documentation`

Also available via the Elexon Insights Solution and IRIS near-real-time push service.

### 1.5 Weather data

**Source: Met Office DataPoint API**

- Site-specific and gridded weather forecasts and observations
- Wind speed, solar radiation, temperature — the physical drivers of generation and demand
- Free API tier available (registration required)

### 1.6 Initial data scope

- **Time window:** 3–5 years of half-hourly data (2020–2025) for model fitting, covering the period of rapid renewable growth
- **Resolution:** Half-hourly (matching the settlement period of the GB electricity market)
- **Variables:** Wind generation (national + embedded), solar PV (from PV_Live), total demand, gas generation, interconnector flows, system imbalance price, carbon intensity

---

## Phase 2: Model Structure

### 2.1 State variables

The stochadex simulation tracks the grid as a coupled stochastic system:

1. **Wind generation process** — stochastic, driven by weather with strong diurnal and seasonal patterns, significant forecast error that grows with lead time
2. **Solar PV generation process** — stochastic, driven by irradiance with deterministic seasonal envelope and stochastic cloud cover perturbation
3. **Demand process** — stochastic with strong daily/weekly/seasonal structure, temperature dependence, and trend components (EV uptake, heat pump adoption)
4. **Residual demand** — true demand minus wind minus solar. This is what dispatchable generation and storage must meet. Its stochastic behaviour is the core modelling challenge.
5. **Battery state of charge (SoC)** — deterministic given dispatch decisions, but the optimal dispatch is a function of the stochastic forecasts above
6. **System imbalance price** — stochastic, strongly correlated with residual demand and wind forecast error. The economic signal that storage dispatch responds to.

### 2.2 Simulation diagram

```
┌─────────────────────────────────────────────────────────┐
│                  WEATHER STATE                           │
│  Wind speed, irradiance, temperature (stochastic)        │
│  Learned from Met Office + NESO forecast error analysis  │
└────┬──────────────┬──────────────┬──────────────────────┘
     │              │              │
     ▼              ▼              ▼
┌──────────┐ ┌──────────┐ ┌────────────────────────────┐
│   WIND   │ │  SOLAR   │ │        DEMAND               │
│   GEN    │ │  PV GEN  │ │  (temp-dependent +           │
│ (MW, HH) │ │ (MW, HH) │ │   stochastic residual)      │
└────┬─────┘ └────┬─────┘ └────────────┬───────────────┘
     │            │                     │
     └────────────┼─────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│              RESIDUAL DEMAND                              │
│  = Demand − Wind − Solar                                 │
│  This is what dispatchable plant + storage must meet     │
│  Its distribution is the core stochastic object          │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│            DISPATCHABLE RESPONSE                          │
│  Gas CCGT/OCGT (merit order), interconnectors, hydro     │
│  Modelled as price-responsive capacity with ramp limits  │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│         BATTERY STORAGE (POLICY LEVER)                    │
│  Charge when: residual demand low / price low / RE high  │
│  Discharge when: residual demand high / price high       │
│  State: SoC, degradation, cycle count                    │
│  Constraints: power rating, energy capacity, ramp rate   │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│         DEMAND RESPONSE (POLICY LEVER)                    │
│  Shift flexible load (EV charging, heat pumps, industrial)│
│  DFS-style incentive events during peak/tight periods    │
│  Constraints: comfort, process requirements, rebound     │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│         OUTCOMES                                          │
│  System balance (MW surplus/deficit per HH)              │
│  Imbalance price (£/MWh)                                │
│  Carbon intensity (gCO₂/kWh)                            │
│  Balancing cost (£)                                     │
│  Renewable curtailment (MWh)                            │
└─────────────────────────────────────────────────────────┘
```

### 2.3 Key modelling choices

- **National-level aggregation** initially: model the GB system as a single node. This is appropriate for storage dispatch and demand response decisions that respond to national price signals. Regional disaggregation (by DNO area) is an extension.
- **Half-hourly time step** matching the settlement period — the natural cadence of the market and all data sources.
- **Stochastic generation model** learned from observed joint distributions of wind, solar, and demand (not from weather models directly). The key is capturing the correlation structure — e.g., low wind often coincides with high pressure (cold, clear), meaning high demand and high solar in summer but high demand and low solar in winter.
- **Battery degradation** modelled as a simple cycle-counting function with depth-of-discharge weighting, sufficient for policy comparison without needing electrochemical detail.
- **Ensemble approach:** run hundreds of stochastic trajectories per policy to build distributions of cost, carbon, and curtailment outcomes.

---

## Phase 3: Learning from Data

### 3.1 Simulation-based inference

The stochadex's established pattern:

1. **Smooth and aggregate** the NESO/Elexon half-hourly data to characterise the joint stochastic behaviour of wind, solar, demand, and price — conditional on time of day, day of week, season, and weather regime.
2. **Fit deviation coefficients** using SBI, matching simulated residual demand and price trajectories to observed historical data.
3. **Key parameters to learn:**
   - Wind generation forecast error distribution as a function of lead time and weather regime
   - Solar PV forecast error distribution (cloud cover uncertainty)
   - Demand forecast error (temperature sensitivity, special events)
   - Cross-correlations between wind, solar, and demand errors
   - Imbalance price response function: how price responds to residual demand deviations
   - Gas plant merit order response: how CCGT/OCGT dispatch responds to residual demand

### 3.2 Renewable scenario generation

For evaluating policies under future grid mixes (e.g., 2030 with ~80% renewables), perturb the learned model by scaling installed wind and solar capacity while preserving the stochastic structure. This is analogous to using UKCP18 change factors in the flood project — same dynamics, different forcing.

### 3.3 Validation strategy

- **Temporal holdout:** Train on 2020–2023, validate on 2024–2025. Key test: does the model reproduce the distribution of imbalance prices and wind forecast errors, not just means?
- **Event reproduction:** Test on known stress events — e.g., the low-wind periods of September 2021 ("wind drought"), cold snaps with high demand, or periods of high curtailment.
- **Price distribution:** Validate that simulated price distributions match the heavy-tailed empirical distribution of imbalance prices (this is where deterministic models fail).

---

## Phase 4: Decision Science Layer

### 4.1 Policy actions to evaluate

The decision science layer evaluates storage dispatch and demand response strategies:

| Policy type | Description | Decision variables |
|-------------|-------------|-------------------|
| **Price-threshold dispatch** | Charge below price P₁, discharge above P₂ | Threshold values P₁, P₂ |
| **Forecast-based dispatch** | Charge when wind forecast high / demand forecast low; discharge on inverse | Forecast lead time, trigger levels |
| **Carbon-minimising dispatch** | Charge during low-carbon periods, discharge to displace gas | Carbon intensity thresholds |
| **Stacked services** | Combine energy arbitrage with frequency response and reserve provision | Allocation fractions across services |
| **Demand response scheduling** | Shift EV charging and heat pump operation to low-carbon/low-price periods | Flexibility window, shift magnitude |
| **Combined storage + DR** | Coordinated battery and demand-side actions | Joint optimisation of both levers |

### 4.2 The forecast uncertainty problem

The central insight is that optimal dispatch depends on forecast accuracy, which degrades with lead time. A deterministic dispatch that charges at 2am based on a "wind will be high tomorrow" forecast is brittle — if the wind doesn't materialise, the battery is full when it shouldn't be. Stochastic ensemble dispatch naturally handles this by evaluating policies across the distribution of outcomes, not just the point forecast.

### 4.3 Objective function

For each dispatch policy, simulate multiple trajectories and evaluate:

- **Primary outcome:** Expected net cost (balancing cost reduction minus battery degradation cost) over 1 year
- **Secondary outcomes:** Carbon intensity reduction (gCO₂/kWh avoided), renewable curtailment avoided (MWh), number of price spikes mitigated
- **Robustness metric:** Performance in worst-case scenarios (e.g., 95th percentile cost during "wind drought" weeks)
- **Future-proofing:** Performance under 2030 grid mix with higher wind/solar penetration

### 4.4 Output

For a given battery installation and grid scenario, produce actionable recommendations:

> *"For a 100MW/200MWh battery operating in the 2025 GB market, a forecast-based dispatch strategy with 6-hour look-ahead outperforms simple price-threshold dispatch by 23% on expected annual revenue, primarily by better anticipating wind ramps. Under a 2030 grid mix with 60GW wind capacity, the same strategy reduces system carbon intensity by 4.2 gCO₂/kWh on average, with the largest impact during autumn evening peaks. Adding coordinated EV charging demand response (shifting 2GW of flexible load by ±4 hours) provides a further 15% reduction in balancing cost."*

---

## Phase 5: Extensions

1. **Regional disaggregation:** Model by DNO area using regional carbon intensity and PV_Live regional data, capturing constraint costs from transmission bottlenecks (a major and growing cost in GB)
2. **Multi-storage coordination:** Model a portfolio of batteries at different locations with different durations, optimising fleet dispatch jointly
3. **Long-duration storage:** Extend to hydrogen electrolysis and storage, addressing multi-day and seasonal storage for "wind drought" events that last a week or more
4. **Vehicle-to-grid (V2G):** Model EV batteries as distributed storage with driving constraints, evaluating V2G as a grid flexibility resource
5. **Market simulation:** Add wholesale market bidding dynamics — simulate strategic behaviour of storage operators competing in the day-ahead and balancing markets
6. **Real-time operational tool:** Connect to live NESO and Elexon APIs for rolling-horizon dispatch optimisation with continuously updated stochastic forecasts

---

## Implementation Status

Phases 1–4 are implemented and working end-to-end. The four `cmd/` commands form the pipeline:

| Command | Phase | Status | Description |
|---------|-------|--------|-------------|
| `cmd/ingest` | 1 | ✅ Done | Downloads NESO historic demand CSV to `dat/demand.csv` |
| `cmd/simulate` | 2 | ✅ Done | Runs full stochastic grid simulation (OU residual demand, battery SoC, imbalance price, outcomes) |
| `cmd/infer` | 3 | ✅ Done | Infers OU parameters from observed data: OLS (fast) + SMC Bayesian (optional) |
| `cmd/evaluate` | 4 | ✅ Done | Evaluates battery dispatch policies against simulated outcomes |

**stochadex version:** `v0.0.0-20260330061034-1555b7d4e430`

### Inference pipeline (`cmd/infer`)

Three steps in sequence:

1. **Data replay** — runs a stochadex simulation over `dat/demand.csv` storing `residual_demand`, `conditional_mean`, and `lagged_residual_demand` via `analysis.NewStateTimeStorageFromPartitions`.
2. **OLS estimation** — uses `analysis.NewScalarRegressionStatsPartition` to stream cumulative regression of ΔX = X_next − X_prev on d = μ − X_prev (no intercept). Recovers OU parameters: θ = −ln(1−β)/Δ, σ² from residual variance formula.
3. **SMC Bayesian inference** (optional, `-smc` flag) — `analysis.RunSMCInference` with N particles, each evaluated against the exact OU transition log-likelihood over all T data steps. Priors are TruncatedNormal in log(θ), log(σ) space, centred on OLS estimates.

Example results on 6 weeks of NESO half-hourly demand:
```
OLS:  theta = 0.2403/half-hour,  sigma = 1591.42 MW/√(half-hour)
SMC:  theta = 0.2356 ± 0.011,    sigma = 1588.57 ± 18.85 MW/√(half-hour)
```

---

## Concrete First Steps

### Week 1–2: Data acquisition and exploration

- [x] Pull historic demand data from NESO Data Portal (2020–2025, half-hourly) — `cmd/ingest`
- [ ] Pull generation by fuel type from NESO (historic generation mix)
- [ ] Pull PV generation from Sheffield Solar PV_Live API (national, half-hourly, 2020–2025)
- [ ] Pull carbon intensity time series from the Carbon Intensity API
- [ ] Register for Elexon and pull imbalance prices (B1770) and volumes (B1780)
- [ ] Exploratory analysis: characterise the joint distribution of wind, solar, demand, and price; identify correlation structure, seasonal patterns, and tail behaviour

### Week 3–4: Minimal stochadex simulation

- [x] Implement a stochastic residual demand model: demand − wind − solar, with learned marginal distributions and correlation structure — `cmd/simulate` with OU process
- [x] Implement a simple price response model: imbalance price as a stochastic function of residual demand — `pkg/grid/imbalance_price.go`
- [x] Add a single-battery state tracker with SoC, power limits, and efficiency losses — `pkg/grid/battery.go`
- [x] Verify the simulation reproduces qualitatively sensible half-hourly dynamics over a week

### Week 5–6: Simulation-based inference

- [x] Smooth and aggregate observed demand data into conditional mean by time-of-day and day-of-week — `pkg/grid/conditional_mean.go`
- [x] Set up OLS + SMC Bayesian inference to learn OU model parameters from observed data — `cmd/infer`
- [ ] Validate: does the simulated price distribution match the empirical heavy-tailed distribution? Does the wind forecast error structure look realistic?

### Week 7–8: Decision science layer

- [x] Implement candidate dispatch policies (price-threshold battery dispatch) — `pkg/grid/battery.go`
- [x] Run policy evaluation: simulate ensembles and compute outcome statistics — `cmd/evaluate`
- [ ] Scale wind/solar capacity to 2030 levels and re-evaluate
- [ ] Produce initial findings and visualisations
- [ ] Write up as a blog post in the "Engineering Smart Actions in Practice" series

---

## Key Data Sources Summary

| Source | URL | Data type | Access |
|--------|-----|-----------|--------|
| NESO Data Portal | data.nationalgrideso.com | Generation mix, demand, wind/solar forecasts, balancing costs, constraint data | Free download, API |
| Carbon Intensity API | api.carbonintensity.org.uk | National/regional carbon intensity + generation mix, HH, 96hr forecast | Free REST API, no registration |
| Sheffield Solar PV_Live | solar.sheffield.ac.uk/pvlive/ | GB solar PV generation estimates (national + regional by GSP), HH from 2013 | Free API, CC BY 4.0 |
| Elexon BMRS | bmrs.elexon.co.uk | Imbalance prices, volumes, generation by fuel type, BM actions, market data | Free API with registration |
| Met Office DataPoint | metoffice.gov.uk/services/data | Weather forecasts and observations (wind, solar radiation, temperature) | Free tier with registration |
| NESO Future Energy Scenarios | nationalgrideso.com/future-energy | Scenario data for 2030/2040/2050 grid mixes, demand projections | Free download |
| Energy Dashboard | energydashboard.co.uk | Aggregated live and historic generation data from Elexon + NESO + PV_Live | Free web access |

---

## References and Related Work

- PyPSA-GB — open-source model of GB power system for simulating future energy scenarios, including storage and demand-side management (University of Edinburgh, 2024)
- SDED-S framework — stochastic dynamic economic dispatch with battery storage, demonstrating cost savings increase sharply beyond 30% renewable penetration (arXiv, 2025)
- UK BESS market analysis — modelling optimal battery dispatch across day-ahead, intraday, frequency response and imbalance settlement in the UK market structure (ScienceDirect, 2024)
- NESO Demand Flexibility Service (DFS) — real operational data on consumer demand response during peak winter periods, available from NESO data portal
- NESO Future Energy Scenarios (FES) — official scenarios for GB energy system evolution to 2050, providing the basis for scaling renewable capacity in future scenarios