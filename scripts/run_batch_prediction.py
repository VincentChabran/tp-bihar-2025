import sys
import os


# Ajout du dossier src au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))



from datetime import datetime, timedelta
from train_pipeline import main 
from acquisition import fetch_and_store_weather_data

import yaml



if __name__ == "__main__":

   with open("configs/acquisition_config.yaml", "r") as f:
      config = yaml.safe_load(f)

   required_keys = ["latitude", "longitude", "start_date", "end_date", "db_path"]
   for key in required_keys:
      if key not in config:
         raise KeyError(f"❌ Clé manquante dans le fichier de config : {key}")

   lat = config["latitude"]
   lon = config["longitude"]
   start_date = config["start_date"]
   end_date = config["end_date"]
   db_path = config["db_path"]
   
   fetch_and_store_weather_data(lat, lon, start_date, end_date, db_path)
   
   # 3. Train model
   main("configs/arima_config.yaml")  
