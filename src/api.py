from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import model_file_for_location
from src.predictor import VisitorPredictor

app = FastAPI(title="Visitor Forecast API", version="2.0.0")
_predictor_cache: dict[str, VisitorPredictor] = {}


class PredictRequest(BaseModel):
    location_id: str
    target_date: str | None = None
    meal_buffer_pct: float | None = None



def _get_predictor(location_id: str) -> VisitorPredictor:
    if location_id in _predictor_cache:
        return _predictor_cache[location_id]
    model_path = model_file_for_location(location_id)
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model not found for location: {location_id}")
    p = VisitorPredictor(str(model_path))
    _predictor_cache[location_id] = p
    return p


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict-next")
def predict_next(req: PredictRequest):
    predictor = _get_predictor(req.location_id)
    pred = predictor.predict_next(target_date=req.target_date, meal_buffer_pct=req.meal_buffer_pct)
    return {
        "location_id": req.location_id,
        "service_date": pred.service_date.strftime("%Y-%m-%d"),
        "predicted_visitors": round(pred.predicted_visitors, 2),
        "predicted_quantile": round(pred.predicted_quantile, 2),
        "residual_buffer": round(pred.residual_buffer, 2),
        "model_segment": pred.model_segment,
        "suggested_meals": pred.suggested_meals,
        "meal_buffer_pct": pred.meal_buffer_pct,
    }
