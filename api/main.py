import sys
import os
from datetime import date
from fastapi import HTTPException, FastAPI
from collections import defaultdict
from fastapi.responses import JSONResponse
import subprocess
from pydantic import BaseModel
import logging
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from ForecastDatabase import ForecastDatabase

# 🔧 Logging config
logging.basicConfig(
    filename="api/logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Forecast API")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "forecast_results.db")



class DateRange(BaseModel):
    start_date: date
    end_date: date



@app.get("/predict")
def get_predictions_only(date: date):
    logger.info(f"[GET /predict] Requête reçue pour la date : {date}")
    try:
        db = ForecastDatabase(DB_PATH)
        df = db.get_predictions_by_date(date.isoformat())

        if df.empty:
            logger.warning(f"[GET /predict] Aucune prédiction pour la date : {date}")
            date_range = db.get_prediction_date_range()
            db.close()
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "Aucune prédiction trouvée pour cette date.",
                    "disponible_de": date_range["start"],
                    "disponible_jusque": date_range["end"]
                }
            )

        logger.info(f"[GET /predict] {len(df)} lignes récupérées pour {date}")
        db.close()

        result = defaultdict(list)
        for _, row in df.iterrows():
            result[row["timestamp"]].append({
                "model": row["model_name"],
                "version": row["version"],
                "prediction": round(row["prediction"], 2)
            })

        structured = [{"timestamp": ts, "predictions": preds} for ts, preds in sorted(result.items())]
        return structured

    except HTTPException as e:
        logger.warning(f"[GET /predict] {e.detail}")
        raise

    except Exception as e:
        logger.error(f"[GET /predict] Erreur inattendue : {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Erreur interne serveur")


@app.post("/combined")
def get_combined_predictions(range: DateRange):
    logger.info(f"[POST /combined] Requête reçue pour période {range.start_date} → {range.end_date}")
    try:
        db = ForecastDatabase(DB_PATH)
        df = db.get_predictions_in_range(range.start_date.isoformat(), range.end_date.isoformat())
        db.close()

        if df.empty:
            logger.warning(f"[POST /combined] Aucune donnée pour la période {range.start_date} → {range.end_date}")
            raise HTTPException(status_code=404, detail="Aucune donnée disponible pour cette période.")

        logger.info(f"[POST /combined] {len(df)} lignes récupérées.")
        return [
            {
                "timestamp": row["timestamp"],
                "model": row["model_name"],
                "version": row["version"],
                "prediction": round(row["prediction"], 2),
                "target": round(row["target"], 2) if row["target"] is not None else None
            }
            for _, row in df.iterrows()
        ]

    except HTTPException as e:
        logger.warning(f"[POST /combined] {e.detail}")
        raise

    except Exception as e:
        logger.error(f"[POST /combined] Erreur inattendue : {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Erreur interne serveur")





@app.get("/version")
def get_version():
    try:
        commit_id = os.getenv("CI_COMMIT_ID")
        if commit_id:
            logger.info(f"[GET /version] Version CI/CD détectée : {commit_id}")
            return JSONResponse(content={"version": commit_id})

        local_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
        logger.info(f"[GET /version] Version Git locale : {local_commit}")
        return JSONResponse(content={"version": "0.0.0"})  # ou local_commit si souhaité

    except Exception as e:
        logger.warning(f"[GET /version] Impossible d'obtenir le commit : {e}")
        return JSONResponse(content={"version": "0.0.0"})
