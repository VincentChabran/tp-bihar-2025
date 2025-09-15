import requests
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import os
import numpy as np

class WeatherDataAcquisition:
    def __init__(self, latitude, longitude, db_path="../data/weather.db"):
        self.latitude = latitude
        self.longitude = longitude
        self.db_path = db_path
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        

    def fetch_weather_data(self, start_date, end_date, extra_variables=None):
        """
        Récupère la température + variables exogènes supplémentaires (optionnelles) pour une période donnée.
        
        Params :
        - start_date (str): Date de début (YYYY-MM-DD)
        - end_date (str): Date de fin (YYYY-MM-DD)
        - extra_variables (list[str]): Variables exogènes supplémentaires à récupérer
        """
        base_vars = ["temperature_2m"]
        if extra_variables:
            base_vars += extra_variables

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(base_vars),
            "timezone": "Europe/Paris"
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame({'time': pd.to_datetime(data['hourly']['time'])})
            for var in base_vars:
                df[var] = data['hourly'].get(var)

            return df

        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur lors de la récupération des données météo : {e}")
            return None

    

    def aggregate_to_3h_intervals(self, df):
        """
        Agrège les données horaires en intervalles de 3h pour toutes les colonnes numériques
        """
        df = df.copy()
        df['hour_group'] = (df['time'].dt.hour // 3) * 3
        df['date'] = df['time'].dt.date
        
        # Identifier les colonnes numériques (exclure 'time', 'date', 'hour_group')
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['hour_group']  # 'date' n'est pas numérique donc pas dans numeric_cols
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Créer les nouveaux timestamps pour chaque groupe de 3h
        aggregated_data = []
        
        for (date, hour_group), group in df.groupby(['date', 'hour_group']):
            new_timestamp = pd.Timestamp.combine(date, pd.Timestamp('00:00:00').time()) + pd.Timedelta(hours=hour_group)
            
            # Créer un dictionnaire avec le timestamp
            row_data = {'time': new_timestamp}
            
            # Calculer la moyenne pour chaque colonne numérique
            for col in numeric_cols:
                row_data[col] = group[col].mean()
            
            aggregated_data.append(row_data)
        
        return pd.DataFrame(aggregated_data).sort_values('time').reset_index(drop=True)
    

    def handle_missing_values(self, df):
        """
        Traite les valeurs manquantes par interpolation linéaire pour toutes les colonnes numériques
        """
        df = df.copy()
        
        # Identifier les colonnes numériques (exclure 'time' si présente)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Appliquer l'interpolation linéaire à chaque colonne numérique
        for col in numeric_cols:
            df[col] = df[col].interpolate(method='linear')

        # if df[numeric_cols].isna().any().any():
        #     raise ValueError("⚠️ Certaines valeurs manquantes n’ont pas pu être interpolées.")
  
        return df
    

