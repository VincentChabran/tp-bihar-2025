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
                time TEXT UNIQUE,
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
                date_trained TEXT,
                UNIQUE(model_name, version)
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

        # Index unique pour les prédictions
        cursor.execute('''
            CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_prediction
            ON predictions (model_id, timestamp)
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
        duplicates = 0
        for i in range(len(timestamps)):
            ts = timestamps[i]
            pred = predictions[i]
            target = targets[i] if targets is not None else None
            features = json.dumps(features_list[i]) if features_list else None

            cursor.execute('''
                INSERT OR IGNORE INTO predictions (model_id, timestamp, target, prediction, features)
                VALUES (?, ?, ?, ?, ?)
            ''', (model_id, ts, target, pred, features))

            if cursor.rowcount == 0:
                duplicates += 1
                print(f"⚠️ Prédiction déjà existante ignorée pour modèle {model_id} à {ts}")

        self.conn.commit()
        print(f"✅ {len(timestamps) - duplicates} nouvelles prédictions insérées.")
        if duplicates > 0:
            print(f"🟡 {duplicates} doublons ignorés.")



    def insert_weather_data(self, df, table_name="weather_data"):
        """
        Insère des données météo dans la base, en évitant les doublons sur `time`.
        """
        df = df.copy()

        if "time" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df["time"] = df.index.astype(str)
            else:
                raise ValueError("❌ Impossible de déterminer la colonne 'time' pour l'insertion.")
        else:
            df["time"] = pd.to_datetime(df["time"]).astype(str)

        with self.conn:
            cursor = self.conn.cursor()
            for _, row in df.iterrows():
                cursor.execute(f"""
                    INSERT OR IGNORE INTO {table_name} 
                    (time, temperature_2m, relative_humidity_2m, wind_speed_10m, cloud_cover)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    row["time"],
                    row.get("temperature_2m"),
                    row.get("relative_humidity_2m"),
                    row.get("wind_speed_10m"),
                    row.get("cloud_cover"),
                ))



    def query_weather_data_by_period(self, start, end, table="weather_data"):
        """
        Récupère les données météo depuis la base entre deux dates.
        """
        cursor = self.conn.cursor()
        cursor.execute(f'''
            SELECT * FROM {table}
            WHERE time BETWEEN ? AND ?
            ORDER BY time ASC
        ''', (start, end))
        
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        df = pd.DataFrame(rows, columns=columns)

        return df
    
    def get_predictions_by_date(self, date_str):
        """
        Récupère toutes les prédictions pour une date donnée (au format 'YYYY-MM-DD').
        """
        print(f"📆 Requête SQL pour la date : {date_str}")

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT p.timestamp, p.prediction, p.target, m.model_name, m.version
            FROM predictions p
            JOIN models m ON p.model_id = m.id
            WHERE DATE(p.timestamp) = ?
            ORDER BY p.timestamp
        ''', (date_str,))

        rows = cursor.fetchall()
        if rows:
            print("🔍 Première ligne :", rows[0])

        columns = [desc[0] for desc in cursor.description]

        return pd.DataFrame(rows, columns=columns)


    def get_predictions_by_date_json(self, date_str):
        df = self.get_predictions_by_date(date_str)
        return df.to_dict(orient="records")


    def get_prediction_date_range(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT MIN(DATE(timestamp)), MAX(DATE(timestamp)) FROM predictions
        ''')
        row = cursor.fetchone()

        print(f"📅 Plage de dates des prédictions : {row}")
        return {"start": row[0], "end": row[1]}


    def get_predictions_in_range(self, start_date, end_date):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT p.timestamp, p.prediction, p.target,
                m.model_name, m.version
            FROM predictions p
            JOIN models m ON p.model_id = m.id
            WHERE DATE(p.timestamp) BETWEEN ? AND ?
            ORDER BY p.timestamp ASC
        ''', (start_date, end_date))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)


    def close(self):
        self.conn.close()

