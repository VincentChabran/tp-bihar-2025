import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
import warnings


class ArimaTrainer:
   def __init__(self, train_ts, val_ts, test_ts):
      self.train_ts = train_ts
      self.val_ts = val_ts
      self.test_ts = test_ts
      self.model = None
      self.results = None
      self.val_forecast = None
      self.test_forecast = None
      self.best_order = None

   def train(self, order=(2, 1, 2), validate=True):
      """Entraîne le modèle ARIMA avec l'ordre spécifié"""
      print(f"📦 Entraînement ARIMA{order}")
      
      try:
         self.model = ARIMA(self.train_ts, order=order)
         self.results = self.model.fit()
         
         if validate and self.val_ts is not None:
               # Évaluation sur validation
               self.val_forecast = self.results.forecast(steps=len(self.val_ts))
               self._print_metrics(self.val_ts, self.val_forecast, "VALIDATION")
         
      except Exception as e:
         print(f"❌ Erreur lors de l'entraînement: {e}")
         return False
      
      return True

   def evaluate_on_test(self):
      """Évalue le modèle entraîné sur le set de test"""
      if self.results is None:
         print("⚠️ Modèle non entraîné.")
         return None
      
      self.test_forecast = self.results.forecast(steps=len(self.test_ts))
      metrics = self._print_metrics(self.test_ts, self.test_forecast, "TEST")
      return metrics


   def _print_metrics(self, y_true, y_pred, dataset_name):
      """Calcule et affiche les métriques"""
      mae = mean_absolute_error(y_true, y_pred)
      rmse = np.sqrt(mean_squared_error(y_true, y_pred))
      r2 = r2_score(y_true, y_pred)
      aic = self.results.aic if self.results else None

      # Emojis basés sur des seuils adaptatifs
      mae_icon = "😊" if mae < np.std(y_true) * 0.5 else "😐" if mae < np.std(y_true) else "😞"
      rmse_icon = "😊" if rmse < np.std(y_true) * 0.6 else "😐" if rmse < np.std(y_true) else "😞"
      r2_icon = "😊" if r2 > 0.8 else "😐" if r2 > 0.5 else "😞"

      print(f"\n📊 Métriques {dataset_name}:")
      print(f"{mae_icon} MAE  : {mae:.3f}")
      print(f"{rmse_icon} RMSE : {rmse:.3f}")
      print(f"{r2_icon} R²    : {r2:.3f}")
      if aic:
         aic_icon = "📉" if aic < 500 else "📈"
         print(f"{aic_icon} AIC   : {aic:.2f}")

      return {'mae': mae, 'rmse': rmse, 'r2': r2, 'aic': aic}


   def search_best_arima(self, p_range=range(0, 4), d_range=[0, 1], q_range=range(0, 4), metric='rmse', verbose=True):
      """Recherche des meilleurs hyperparamètres ARIMA"""
      if self.val_ts is None:
         print("⚠️ Pas de set de validation fourni.")
         return None, None

      best_score = np.inf
      best_order = None
      results_log = []

      grid = list(product(p_range, d_range, q_range))
      print(f"🔍 Recherche sur {len(grid)} combinaisons...")

      with warnings.catch_warnings():
         warnings.simplefilter("ignore")
         
         for p, d, q in tqdm(grid, desc="Grid Search ARIMA", disable=not verbose):
            try:
               model = ARIMA(self.train_ts, order=(p, d, q))
               model_fit = model.fit()
               forecast = model_fit.forecast(steps=len(self.val_ts))
               
               # Calcul de la métrique choisie
               if metric == 'rmse':
                  score = np.sqrt(mean_squared_error(self.val_ts, forecast))
               elif metric == 'mae':
                  score = mean_absolute_error(self.val_ts, forecast)
               elif metric == 'aic':
                  score = model_fit.aic
               else:
                  score = np.sqrt(mean_squared_error(self.val_ts, forecast))

               results_log.append({
                  'order': (p, d, q),
                  'score': score,
                  'aic': model_fit.aic
               })

               if score < best_score:
                  best_score = score
                  best_order = (p, d, q)

            except Exception as e:
               if verbose:
                  print(f"⚠️ Erreur pour ARIMA{(p,d,q)}: {str(e)[:50]}...")
               continue

      if best_order:
         self.best_order = best_order
         print(f"\n✅ Meilleur ARIMA: {best_order}")
         print(f"📈 {metric.upper()}: {best_score:.3f}")
         
         # Réentraîner avec les meilleurs paramètres
         self.train(order=best_order, validate=True)
      else:
         print("❌ Aucun modèle valide trouvé.")

      return best_order, best_score

   def plot_predictions(self, show_validation=False):
      """Visualise les prédictions"""
      if self.test_forecast is None and self.val_forecast is None:
         print("⚠️ Aucune prédiction disponible.")
         return

      fig, axes = plt.subplots(1, 2 if show_validation else 1, figsize=(15, 5))
      if not show_validation:
         axes = [axes]

      # Plot test
      if self.test_forecast is not None:
         ax = axes[0] if show_validation else axes[0]
         ax.plot(self.test_ts.index, self.test_ts.values, 
                  label="Observations", linewidth=2, color='blue')
         ax.plot(self.test_ts.index, self.test_forecast.values, 
                  label="Prédictions", alpha=0.8, color='red', linestyle='--')
         ax.set_title("Test Set - Prédictions vs Observations")
         ax.set_xlabel("Date")
         ax.set_ylabel("Valeur")
         ax.legend()
         ax.grid(True, alpha=0.3)

      # Plot validation si demandé
      if show_validation and self.val_forecast is not None:
         ax = axes[1]
         ax.plot(self.val_ts.index, self.val_ts.values, 
                  label="Observations", linewidth=2, color='blue')
         ax.plot(self.val_ts.index, self.val_forecast.values, 
                  label="Prédictions", alpha=0.8, color='orange', linestyle='--')
         ax.set_title("Validation Set - Prédictions vs Observations")
         ax.set_xlabel("Date")
         ax.set_ylabel("Valeur")
         ax.legend()
         ax.grid(True, alpha=0.3)

      plt.tight_layout()
      plt.show()


   def get_summary(self):
      """Retourne un résumé du modèle"""
      if self.results is None:
         return "Modèle non entraîné"
      
      return self.results.summary()


   def predict_future(self, steps=10):
      """Prédit les valeurs futures"""
      if self.results is None:
         print("⚠️ Modèle non entraîné.")
         return None
      
      # Prédiction sur toute la série (train + val + test)
      full_series = pd.concat([self.train_ts, self.val_ts, self.test_ts])
      model_full = ARIMA(full_series, order=self.results.model.order)
      results_full = model_full.fit()
      
      future_forecast = results_full.forecast(steps=steps)
      return future_forecast
   

