import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
import matplotlib.pyplot as plt



warnings.filterwarnings('ignore')

class StationarityHelper:
   
   @staticmethod
   def test_stationarity(series: pd.Series):
      """
      Effectue les tests ADF et KPSS sur une série temporelle.
      Affiche les résultats.
      """
      print("=== TEST DE STATIONNARITÉ : ADF ===")
      adf_stat, adf_pval, _, _, adf_crit, *_ = adfuller(series.dropna())
      print(f"ADF Statistic : {adf_stat:.4f}")
      print(f"p-value       : {adf_pval:.4f}")
      for key, val in adf_crit.items():
         print(f"Critique {key}% : {val:.3f}")
      if adf_pval < 0.05:
         print("→ On rejette H₀ : la série est **stationnaire**")
      else:
         print("→ On ne rejette pas H₀ : la série **n’est pas stationnaire**")
      
      print("\n=== TEST DE STATIONNARITÉ : KPSS ===")
      kpss_stat, kpss_pval, _, kpss_crit = kpss(series.dropna(), regression='c', nlags='auto')
      print(f"KPSS Statistic : {kpss_stat:.4f}")
      print(f"p-value        : {kpss_pval:.4f}")
      for key, val in kpss_crit.items():
         print(f"Critique {key}% : {val:.3f}")
      if kpss_pval < 0.05:
         print("→ On rejette H₀ : la série **n’est pas stationnaire**")
      else:
         print("→ On ne rejette pas H₀ : la série est **stationnaire**")

      print("\n=== INTERPRÉTATION COMBINÉE ===")
      if adf_pval < 0.05 and kpss_pval > 0.05:
         print("✅ Les deux tests confirment que la série est stationnaire.")
      elif adf_pval > 0.05 and kpss_pval < 0.05:
         print("❌ Les deux tests indiquent une série non stationnaire.")
      else:
         print("⚠️ Résultats contradictoires entre ADF et KPSS. Envisager une différenciation.")


   

   @staticmethod
   def plot_stationarity(series, window=24):
      """
      Affiche la moyenne mobile et l'écart-type mobile pour évaluer visuellement la stationnarité.
      
      - series : pd.Series indexée par le temps
      - window : taille de la fenêtre mobile (en nombre de points)
      """
      rolling_mean = series.rolling(window=window).mean()
      rolling_std = series.rolling(window=window).std()

      plt.figure(figsize=(14, 5))
      plt.plot(series, label="Série originale", alpha=0.5)
      plt.plot(rolling_mean, label=f"Moyenne mobile ({window})", color="blue")
      plt.plot(rolling_std, label=f"Écart-type mobile ({window})", color="orange")
      plt.title("Stationnarité - Moyenne et Écart-Type Mobiles")
      plt.xlabel("Temps")
      plt.ylabel("Valeur")
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.show()


   

   @staticmethod
   def make_stationary(df: pd.DataFrame, column: str = "temperature_2m") -> pd.DataFrame:
      """
      Applique une différenciation première à une série temporelle pour la stationnariser,
      met à jour le DataFrame et affiche les résultats des tests ADF et KPSS.

      Paramètres :
      - df : DataFrame contenant la série
      - column : nom de la colonne cible (par défaut "temperature_2m")

      Retour :
      - df_stationary : DataFrame transformé
      """
      df_diff = df.copy()
      df_diff[column] = df_diff[column].diff()
      df_diff = df_diff.dropna()

      series = df_diff[column]

      # Test ADF
      adf_stat, adf_pval, _, _, adf_crit, *_ = adfuller(series)
      print("\n=== TEST DE STATIONNARITÉ : ADF (série différenciée) ===")
      print(f"ADF Statistic : {adf_stat:.4f}")
      print(f"p-value       : {adf_pval:.4f}")
      for k, v in adf_crit.items():
         print(f"Critique {k}% : {v:.3f}")
      print("→", "Stationnaire" if adf_pval <= 0.05 else "Non stationnaire")

      # Test KPSS
      kpss_stat, kpss_pval, _, kpss_crit = kpss(series, regression='c', nlags='auto')
      print("\n=== TEST DE STATIONNARITÉ : KPSS (série différenciée) ===")
      print(f"KPSS Statistic : {kpss_stat:.4f}")
      print(f"p-value        : {kpss_pval:.4f}")
      for k, v in kpss_crit.items():
         print(f"Critique {k}% : {v:.3f}")
      print("→", "Stationnaire" if kpss_pval > 0.05 else "Non stationnaire")

      print("\n=== INTERPRÉTATION COMBINÉE ===")
      if adf_pval <= 0.05 and kpss_pval > 0.05:
         print("✅ Série transformée stationnaire.")
      else:
         print("⚠️ Résultat ambigu. Envisager transformation supplémentaire ou paramétrage alternatif.")

      return df_diff

