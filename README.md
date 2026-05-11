# F1 Race Outcome Predictor

Predicts finishing positions for Formula 1 races using a **LightGBM learning-to-rank model** trained on 8 seasons of race data (2018–2026). The model ranks all drivers within each race rather than predicting finish positions directly — the right framing for a sport where relative order is what matters.

Race data stays current automatically: every training run fetches the latest completed races via FastF1 and the dashboard pre-selects the next upcoming GP on boot.

---

## Results

Evaluated on the most recent 20% of races (chronological hold-out — no data leakage):

| Metric | Score |
|--------|-------|
| Mean Spearman rank correlation | **0.670** |
| Median Spearman rank correlation | **0.723** |
| NDCG@3 — podium accuracy | **0.910** |
| NDCG@10 — points-finish accuracy | **0.892** |
| NDCG@20 — full-field accuracy | **0.954** |

**Spearman** measures how well the predicted order matches the actual order within each race — 1.0 is perfect, 0.0 is random. **NDCG@k** rewards getting the top-k positions right and penalises mistakes at the front more than at the back. A podium NDCG of 0.91 means the model almost always identifies all three podium finishers, even if the exact order shifts.

Hyperparameters were found via 30-trial Optuna search (NDCG@10 objective, time-series validation).

---

## How it works

### 1. Data pipeline

Race results come from two sources that are merged automatically:

- **Ergast database** — historical F1 data (CSV files) from 2018 onward, covering results, qualifying, driver info, team info, and circuit metadata.
- **FastF1** — live race results and weather for any races after the Ergast export cutoff. On every `python train.py` run, the fetcher queries the FastF1 event schedule, identifies completed races not yet cached, and downloads them. Saves incrementally so interruptions don't lose progress.

### 2. Feature engineering

All rolling features use `shift(1)` — a race never sees its own result in its lookback window, preventing data leakage.

| Feature | What it captures |
|---------|-----------------|
| `start` | Qualifying grid position |
| `finish_ewm3` | Driver's exponentially-weighted recent form (span=3, recent races count more) |
| `driver_reliability_ewm10` | 1 − EWM driver DNF rate (span=10) — how reliably this driver finishes |
| `team_reliability_ewm10` | 1 − EWM team DNF rate (span=10) — mechanical reliability |
| `driver_circuit_avg` | Driver's expanding historical average finish at this specific circuit |
| `age_at_race` | Driver age in years on race day |
| `rainfall` | Binary — did it rain? |
| `avg_track_temp` | Median track temperature across the race (°C) |
| `min_humidity` | Minimum humidity during the race (%) |
| `driver_active` / `team_active` | Whether this entry is on the current grid |
| `GP name`, `team`, `driver`, `circuit name` | Categorical identities (native LightGBM categoricals) |

EWM (exponentially weighted) windows weight recent races more than older ones — a driver who retired last race matters more than one who retired two seasons ago.

### 3. Model

**LGBMRanker** with `lambdarank` objective. This is a learning-to-rank model: it learns to order drivers *within* each race rather than predict a raw finish number. Rows are grouped by race date; the model is penalised for getting the top positions wrong more than the back of the field.

Training uses an 80/20 chronological split — the most recent races form the test set, matching real-world usage where you always predict forward in time.

### 4. Inference

For an upcoming race, the predictor:
1. Loads the **feature snapshot** saved at training time — this contains each driver's latest rolling reliability and form values, and their historical average at every circuit
2. Builds a feature row for each driver using the snapshot + your grid input + weather conditions
3. Runs the LGBMRanker and ranks drivers by predicted score
4. Converts raw scores to **confidence percentages** via softmax (sums to 100% across the field)

The snapshot approach means inference is instant and leakage-free — it only uses information that existed before the race being predicted.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the Ergast dataset

Download from Kaggle (one-time setup):

```bash
python -c "
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
api.dataset_download_files(
    'rohanrao/formula-1-world-championship-1950-2020',
    path='./f1_datasets', unzip=True
)
"
```

### 3. Train the model

```bash
python train.py
```

This will:
- Fetch any completed races not yet in the local cache (FastF1)
- Build features and train the LGBMRanker on all available data
- Evaluate on the held-out test set and print metrics
- Save the model and feature snapshot to `models/`
- Log the run to MLflow

On subsequent runs, only new races are fetched — already-cached data is skipped.

### 4. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

Opens at `http://localhost:8502`. The sidebar automatically detects and pre-selects the next upcoming GP and its date. Edit the starting grid, set weather conditions, and hit **Predict**.

### 5. Launch the REST API

```bash
uvicorn api.main:app --reload
```

Docs at `http://localhost:8000/docs`.

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Dashboard

The Streamlit UI lets you:
- Select any race from the 2026 calendar (next upcoming race is pre-selected on boot)
- Edit the starting grid — driver and team dropdowns for all 22 entries
- Set weather conditions (rain toggle, track temp slider, humidity slider)
- View the **predicted podium** with team-colour highlights
- View the **full predicted ranking** with confidence % per driver

Confidence is computed as the softmax of the LGBMRanker scores within the race — it reflects how much more the model favours one driver over another, not just whether it picks them first.

---

## REST API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "race_name": "Canadian Grand Prix",
    "race_date": "2026-06-14",
    "rainfall": 0,
    "avg_track_temp": 28,
    "min_humidity": 55,
    "grid": [
      {"driver": "Antonelli",  "team": "Mercedes", "start": 1},
      {"driver": "Russell",    "team": "Mercedes", "start": 2},
      {"driver": "Norris",     "team": "McLaren",  "start": 3},
      {"driver": "Piastri",    "team": "McLaren",  "start": 4},
      {"driver": "Leclerc",    "team": "Ferrari",  "start": 5}
    ]
  }'
```

---

## Project structure

```
f1_datasets/              # Ergast CSVs + auto-fetched recent race cache
models/                   # Saved model + feature snapshot (git-ignored)
src/
├── config.py             # Paths, constants, active driver/team lists
├── data/
│   ├── loader.py         # Loads Ergast CSVs and static season2025.csv
│   ├── cleaner.py        # Merges, standardises, and patches all data sources
│   └── fetcher.py        # Auto-discovers and fetches new races via FastF1
├── features/
│   └── engineer.py       # EWM rolling features, circuit affinity, feature snapshot
├── models/
│   ├── train.py          # LGBMRanker, time-based split, BEST_PARAMS
│   └── evaluate.py       # Spearman and NDCG@k per race
└── predict/
    ├── predictor.py      # Loads model bundle, runs inference
    └── prep.py           # Builds feature DataFrame for a new race from snapshot
api/
└── main.py               # FastAPI REST endpoint with /health and /predict
app/
└── dashboard.py          # Streamlit dashboard
train.py                  # CLI entry point — fetch → clean → train → evaluate → save
tests/
├── test_features.py      # Feature engineering correctness (no leakage, correct rolling)
└── test_predict.py       # Prediction pipeline correctness (20 unique positions, etc.)
```

---

## Experiment tracking

Training runs are logged to MLflow automatically. Launch the UI with:

```bash
mlflow ui
# http://localhost:5000
```

Each run logs: hyperparameters, training race count, test race count, feature count, and all five evaluation metrics.

---

## Docker

```bash
docker build -t f1-predictor .
docker run -p 8000:8000 f1-predictor
```

---

## Data sources

- **Ergast F1 database** — historical race data via [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
- **FastF1** — live race results, weather, and event schedules (2025–present)

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `Predictor.ipynb` | Full research notebook: EDA, feature selection, model development |
| `2025 season.ipynb` | Builds the static season2025.csv via FastF1 |
| `track data.ipynb` | Circuit characteristics exploration |
| `compare.py` | Baseline vs improved feature set comparison (flat rolling → EWM) |
| `compare_v2.py` | Optuna tuning vs DNF classifier vs ensemble — all three ablations |
