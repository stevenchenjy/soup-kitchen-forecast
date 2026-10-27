from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import ForecastTargetDateError, model_file_for_location, parse_service_date
from src.meal_policy import meal_policy_config_fingerprint
from src.predictor import VisitorPredictor, WeatherForecastUnavailableError

app = FastAPI(title="Visitor Forecast API", version="2.0.0")
_predictor_cache: dict[str, tuple[tuple, VisitorPredictor]] = {}


class PredictRequest(BaseModel):
    location_id: str
    target_date: str | None = None
    meal_buffer_pct: float | None = None



def _get_predictor(location_id: str) -> VisitorPredictor:
    model_path = Path(model_file_for_location(location_id)).resolve()
    try:
        stat = model_path.stat()
    except FileNotFoundError:
        _predictor_cache.pop(location_id, None)
        raise HTTPException(status_code=404, detail=f"Model not found for location: {location_id}")
    try:
        policy_fingerprint = meal_policy_config_fingerprint()
    except ValueError as exc:
        _predictor_cache.pop(location_id, None)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    signature = (
        str(model_path), stat.st_ino, stat.st_size, stat.st_mtime_ns,
        stat.st_ctime_ns, policy_fingerprint,
    )
    cached = _predictor_cache.get(location_id)
    if cached is not None and cached[0] == signature:
        return cached[1]
    try:
        p = VisitorPredictor(str(model_path))
    except ValueError as exc:
        _predictor_cache.pop(location_id, None)
        raise HTTPException(
            status_code=503, detail="Model or meal preparation policy is invalid."
        ) from exc
    # Nightly publication replaces the file atomically. Do not associate a
    # package loaded during replacement with the identity of another version.
    try:
        after = model_path.stat()
        after_policy_fingerprint = meal_policy_config_fingerprint()
    except FileNotFoundError:
        after = None
        after_policy_fingerprint = None
    except ValueError as exc:
        _predictor_cache.pop(location_id, None)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if after is None or signature != (
        str(model_path), after.st_ino, after.st_size, after.st_mtime_ns,
        after.st_ctime_ns, after_policy_fingerprint,
    ) or getattr(p, "meal_policy_config_fingerprint", None) not in (
        None, policy_fingerprint,
    ):
        _predictor_cache.pop(location_id, None)
        raise HTTPException(
            status_code=503,
            detail="Model or preparation policy changed while loading; retry the request.",
        )
    _predictor_cache[location_id] = (signature, p)
    return p


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict-next")
def predict_next(req: PredictRequest):
    try:
        normalized_date = parse_service_date(req.target_date) if req.target_date else None
        predictor = _get_predictor(req.location_id)
        pred = predictor.predict_next(target_date=normalized_date, meal_buffer_pct=req.meal_buffer_pct)
    except ForecastTargetDateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WeatherForecastUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "location_id": req.location_id,
        "service_date": pred.service_date.strftime("%Y-%m-%d"),
        "predicted_visitors": round(pred.predicted_visitors, 2),
        "predicted_quantile": round(pred.predicted_quantile, 2),
        "residual_buffer": round(pred.residual_buffer, 2),
        "model_segment": pred.model_segment,
        "suggested_meals": pred.suggested_meals,
        "meal_buffer_pct": pred.meal_buffer_pct,
        "recommendation_policy_id": getattr(pred, "recommendation_policy_id", None),
        "base_recommendation_policy_id": getattr(pred, "base_recommendation_policy_id", None),
        "raw_recommended_meals": getattr(pred, "raw_recommended_meals", None),
        "meal_reduction": getattr(pred, "meal_reduction", 0),
    }
