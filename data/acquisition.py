import sys
import os
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.frequencies import to_offset

# Ajoute le dossier `src/` au PYTHONPATH
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

    # 1. Récupération brute
    df_raw = acquisition.fetch_weather_data(start_date, end_date, extra_variables=extra_vars)

    if df_raw is None:
        raise ValueError("❌ Échec lors de la récupération des données météo (None).")
    if df_raw.empty:
        raise ValueError("❌ Échec lors de la récupération des données météo (vide).")

    # 2. Interpolation des valeurs manquantes
    df_clean = acquisition.handle_missing_values(df_raw)

    # 3. Agrégation toutes les 3 heures
    df_agg = acquisition.aggregate_to_3h_intervals(df_clean)

    return df_agg


def fetch_and_store_weather_data(lat, lon, start_date, end_date, db_path="data/forecast_results.db"):
    """
    Récupère, nettoie, agrège et stocke les données météo dans SQLite.
    """
    df = fetch_weather_data_only(lat, lon, start_date, end_date)

    print(df)

    db = ForecastDatabase(db_path)
    db.insert_weather_data(df)
    db.close()

    return df





def inspect_weather_data_table(db_path="data/forecast_results.db", table_name="weather_data", limit=10):
    """
    Affiche les premières lignes de la table météo et le nombre total de lignes.
    """
    import sqlite3
    import pandas as pd

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Nombre total de lignes
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        print(f"📊 Nombre total de lignes dans '{table_name}' : {total_rows}")

        # Affichage des premières lignes
        df = pd.read_sql_query(f"""
            SELECT * FROM {table_name}
            ORDER BY timestamp ASC
            LIMIT {limit}
        """, conn)

        print(f"\n📋 Aperçu des {limit} premières lignes :")
        print(df)




if __name__ == "__main__":

    # 🌍 Paramètres codés en dur
    lat = 41.9260
    lon = 8.7369
    start_date = "2021-01-01"
    end_date = "2023-12-31"
    db_path = "data/forecast_results.db"

    print("🚀 Lancement de l'acquisition météo...")
    df = fetch_and_store_weather_data(
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
        db_path=db_path
    )
    print(f"✅ {len(df)} lignes récuperé.")

    inspect_weather_data_table()

