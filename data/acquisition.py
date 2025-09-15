import sys
import os
import yaml
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.frequencies import to_offset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from WeatherDataAcquisition import WeatherDataAcquisition
from ForecastDatabase import ForecastDatabase



def fetch_weather_data_only(lat, lon, start_date, end_date):
    """
    Récupère, nettoie et agrège les données météo, sans les stocker en base.
    """
    print("🌍 Étape 0 : Initialisation de WeatherDataAcquisition")
    acquisition = WeatherDataAcquisition(latitude=lat, longitude=lon)

    extra_vars = [
        "relative_humidity_2m",
        "wind_speed_10m",
        "cloud_cover",
    ]

    try:
        df_raw = acquisition.fetch_weather_data(start_date, end_date, extra_variables=extra_vars)
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lors de la récupération des données météo : {e}")

    if df_raw is None or df_raw.empty:
        raise ValueError("❌ Aucune donnée météo récupérée (None ou vide).")

    df_clean = acquisition.handle_missing_values(df_raw)
    df_agg = acquisition.aggregate_to_3h_intervals(df_clean)

    if df_agg.empty:
        raise ValueError("❌ Résultat vide après l’agrégation des données.")

    return df_agg



def fetch_and_store_weather_data(lat, lon, start_date, end_date, db_path="data/forecast_results.db"):
    """
    Récupère, nettoie, agrège et stocke les données météo dans SQLite.
    """
    df = fetch_weather_data_only(lat, lon, start_date, end_date)
    print(f"📊 Aperçu : {len(df)-8} lignes récupérées")
    if "time" not in df.columns:
        raise ValueError("❌ La colonne 'time' est manquante dans le DataFrame météo.")

    try:
        db = ForecastDatabase(db_path)
        db.insert_weather_data(df)
        db.close()
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lors de l’insertion en base : {e}")

    return df



def load_weather_data_from_db(start_date, end_date, db_path="data/forecast_results.db"):
    """
    Récupère les données météo depuis la base entre deux dates.
    Vérifie que toutes les données attendues sont présentes (toutes les 3h).
    """
    try:
        db = ForecastDatabase(db_path)
        df = db.query_weather_data_by_period(start=start_date, end=end_date)
        db.close()
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lors de la lecture de la base : {e}")

    if df.empty:
        raise ValueError("⚠️ Aucune donnée météo trouvée en base pour cette période.")

    print(f"✅ {len(df)} lignes récupérées depuis la base.")

    # Mise en forme temporelle
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df["time"] = df.index  # Nécessaire pour les opérations downstream

    # Vérifie que toutes les données sont présentes
    expected_index = pd.date_range(start=start_date, end=pd.to_datetime(end_date) - pd.Timedelta(hours=3), freq="3h")
    missing = expected_index.difference(df.index)

    if not missing.empty:
        actual_min = df.index.min().strftime("%Y-%m-%d %H:%M")
        actual_max = df.index.max().strftime("%Y-%m-%d %H:%M")
        raise ValueError(
            f"❌ Données incomplètes entre {start_date} et {end_date}.\n"
            f"👉 La base contient actuellement des données de {actual_min} à {actual_max}."
        )

    return df



if __name__ == "__main__":
     # 🔧 Lecture du fichier de config
    with open("configs/acquisition_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    lat = config["latitude"]
    lon = config["longitude"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    db_path = config["db_path"]

    print("🚀 Lancement de l'acquisition météo...")
    df = fetch_and_store_weather_data(
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
        db_path=db_path
    )
    print(f"✅ {len(df)} lignes récupérées.")

    load_weather_data_from_db(
        start_date=start_date,
        end_date=end_date,
        db_path=db_path
    )


