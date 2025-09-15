# monitoring/compare_predictions.py
import sys
import argparse
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ForecastDatabase import ForecastDatabase



def main(date_str):
   # Connexion à la base
   db_path = os.path.join(os.path.dirname(__file__), "..", "data", "forecast_results.db")
   db = ForecastDatabase(db_path)

   # Chargement des prédictions pour une date donnée
   df = db.get_predictions_by_date(date_str)
   db.close()

   if df.empty:
      print(f"❌ Aucune donnée trouvée pour la date {date_str}")
      return

   # Création du graphique
   plt.figure(figsize=(12, 6))
   models = df["model_name"].unique()

   for model in models:
      df_model = df[df["model_name"] == model]
      plt.plot(df_model["timestamp"], df_model["prediction"], label=f"Prédiction {model} ({df_model['version'].iloc[0]})")

   if df["target"].notnull().any():
      plt.plot(df["timestamp"], df["target"], label="Observé", color="black", linewidth=2, linestyle="--")

   plt.title(f"Comparaison des prédictions - {date_str}")
   plt.xlabel("Horodatage")
   plt.ylabel("Température")
   plt.xticks(rotation=45)
   plt.legend()
   plt.tight_layout()

   # Sauvegarde dans monitoring/output/
   output_dir = os.path.join(os.path.dirname(__file__), "output")
   os.makedirs(output_dir, exist_ok=True)
   output_path = os.path.join(output_dir, f"comparaison_{date_str}.png")
   plt.savefig(output_path)
   print(f"✅ Graphique sauvegardé : {output_path}")





if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--date", type=str, required=True, help="Date cible au format YYYY-MM-DD")
   args = parser.parse_args()

   main(args.date)
