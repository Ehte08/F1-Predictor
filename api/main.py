"""
F1 Race Outcome Predictor — REST API

Run with:
    uvicorn api.main:app --reload

Docs at: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict.predictor import F1Predictor
from src.predict.prep import build_race_features

app = FastAPI(
    title="F1 Race Outcome Predictor",
    description="Predicts finishing positions for an upcoming F1 race using LGBMRanker.",
    version="1.0.0",
)

_predictor: F1Predictor | None = None


def get_predictor() -> F1Predictor:
    global _predictor
    if _predictor is None:
        _predictor = F1Predictor()
    return _predictor


# ── Request / Response schemas ────────────────────────────────────────────────

class GridEntry(BaseModel):
    driver: str = Field(..., example="Norris")
    team: str = Field(..., example="McLaren")
    start: int = Field(..., ge=1, le=20, description="Starting grid position")


class PredictRequest(BaseModel):
    race_name: str = Field(..., example="Qatar Grand Prix")
    race_date: str = Field(..., example="2025-11-30", description="ISO date YYYY-MM-DD")
    grid: list[GridEntry] = Field(..., min_length=20, max_length=20)
    rainfall: int = Field(0, ge=0, le=1, description="1 if rain expected, 0 if dry")
    avg_track_temp: float = Field(35.0, description="Expected track temperature °C")
    min_humidity: float = Field(40.0, description="Expected minimum humidity %")


class DriverResult(BaseModel):
    pred_finish: int
    driver: str
    team: str
    start: int
    pred_score: float
    p_win: float
    p_podium: float
    p_points: float
    p_dnf: float


class PredictResponse(BaseModel):
    race: str
    date: str
    model_trained_through: str
    predictions: list[DriverResult]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health():
    predictor = get_predictor()
    return {
        "status": "ok",
        "model_trained_through": predictor.training_cutoff,
        "n_features": len(predictor.feature_names),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Predictions"])
def predict(req: PredictRequest):
    if len(req.grid) != 20:
        raise HTTPException(status_code=422, detail="Grid must contain exactly 20 entries.")

    predictor = get_predictor()

    race_df = build_race_features(
        race_name=req.race_name,
        race_date=req.race_date,
        grid=[e.model_dump() for e in req.grid],
        rainfall=req.rainfall,
        avg_track_temp=req.avg_track_temp,
        min_humidity=req.min_humidity,
    )

    # Align to model feature set
    for col in predictor.feature_names:
        if col not in race_df.columns:
            race_df[col] = 0.5

    cols = predictor.feature_names + ["driver", "team", "start"]
    result = predictor.predict_with_sim(race_df[cols], n_sims=5000)

    return PredictResponse(
        race=req.race_name,
        date=req.race_date,
        model_trained_through=predictor.training_cutoff,
        predictions=[
            DriverResult(
                pred_finish=int(row["pred_finish"]),
                driver=str(row["driver"]),
                team=str(row["team"]),
                start=int(row["start"]),
                pred_score=float(row["pred_score"]),
                p_win=float(row["p_win"]),
                p_podium=float(row["p_podium"]),
                p_points=float(row["p_points"]),
                p_dnf=float(row["p_dnf"]),
            )
            for _, row in result.iterrows()
        ],
    )
