import sqlite3
import json
from datetime import datetime
import pandas as pd


class ForecastDatabase:
    def __init__(self, db_path="../data/forecast.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()


    def _create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weather_data (
                id INTEGER PRIMARY KEY,
                timestamp TEXT UNIQUE,
                temperature_2m REAL,
                relative_humidity_2m REAL,
                wind_speed_10m REAL,
                cloud_cover REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY,
                model_name TEXT,
                train_start TEXT,
                train_end TEXT,
                version TEXT,
                params TEXT,
                date_trained TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                model_id INTEGER,
                timestamp TEXT,
                target REAL,
                prediction REAL,
                features TEXT,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        ''')

        self.conn.commit()


    def insert_model(self, model_name, train_start, train_end, params_dict, version="1.0"):
        cursor = self.conn.cursor()
        params_json = json.dumps(params_dict)
        date_trained = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO models (model_name, train_start, train_end, version, params, date_trained)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (model_name, train_start, train_end, version, params_json, date_trained))
        self.conn.commit()
        return cursor.lastrowid


    def insert_predictions(self, model_id, timestamps, predictions, targets=None, features_list=None):
        cursor = self.conn.cursor()
        for i in range(len(timestamps)):
            ts = timestamps[i]
            pred = predictions[i]
            target = targets[i] if targets is not None else None
            features = json.dumps(features_list[i]) if features_list else None

            cursor.execute('''
                INSERT INTO predictions (model_id, timestamp, target, prediction, features)
                VALUES (?, ?, ?, ?, ?)
            ''', (model_id, ts, target, pred, features))

        self.conn.commit()

    def insert_weather_data(self, df, table_name="weather_data"):
        """
        Insère des données météo dans la base, en évitant les doublons sur `timestamp`.
        """
        df = df.copy()

        if "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"]).astype(str)
        elif isinstance(df.index, pd.DatetimeIndex):
            df["timestamp"] = df.index.astype(str)
        else:
            raise ValueError("❌ Impossible de déterminer le champ 'timestamp' pour l'insertion.")

        with self.conn:
            cursor = self.conn.cursor()
            for _, row in df.iterrows():
                cursor.execute(f"""
                    INSERT OR IGNORE INTO {table_name} 
                    (timestamp, temperature_2m, relative_humidity_2m, wind_speed_10m, cloud_cover)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    row["timestamp"],
                    row.get("temperature_2m"),
                    row.get("relative_humidity_2m"),
                    row.get("wind_speed_10m"),
                    row.get("cloud_cover"),
                ))


   


    def close(self):
        self.conn.close()

