# generate_prediction.py
import sys
import os
from datetime import datetime
from collections import defaultdict
import pandas as pd

# Assure que le module est trouvé
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..' ,"src")))

from ForecastDatabase import ForecastDatabase

DB_PATH = os.path.join("data", "forecast_results.db")

def generate_prediction(date_str: str):
    print(f"📅 Génération des prédictions pour le {date_str}")
    
    # Connexion DB
    db = ForecastDatabase(DB_PATH)
    df = db.get_predictions_by_date(date_str)
    db.close()

    if df.empty:
        print(f"❌ Aucune prédiction trouvée pour la date : {date_str}")
        return

    # Regroupement
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[row["timestamp"]].append({
            "model": row["model_name"],
            "version": row["version"],
            "prediction": round(row["prediction"], 2)
        })

    # Affichage
    for timestamp in sorted(grouped.keys()):
        print(f"\n🕒 {timestamp}")
        for pred in grouped[timestamp]:
            print(f"   • {pred['model']} (v{pred['version']}): {pred['prediction']}°C")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_prediction.py YYYY-MM-DD")
        sys.exit(1)

    generate_prediction(sys.argv[1])
