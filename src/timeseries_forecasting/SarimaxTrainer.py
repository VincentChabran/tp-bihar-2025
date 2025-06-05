import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
import warnings
import time

class SarimaxTrainer:
   def __init__(self, train_ts, val_ts, test_ts, train_exog, val_exog, test_exog):
      self.train_ts = train_ts
      self.val_ts = val_ts
      self.test_ts = test_ts
      self.train_exog = train_exog
      self.val_exog = val_exog
      self.test_exog = test_exog

      self.model = None
      self.results = None
      self.val_forecast = None
      self.test_forecast = None
      self.best_order = None
      self.best_seasonal_order = None
      self.search_history = []


   def train(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), validate=True):
      print(f"📦 Entraînement SARIMAX{order}x{seasonal_order}")
      start = time.time()

      try:
         self.model = SARIMAX(
               self.train_ts,
               exog=self.train_exog,
               order=order,
               seasonal_order=seasonal_order,
               enforce_stationarity=False,
               enforce_invertibility=False
         )
         self.results = self.model.fit(disp=False)

         if validate and self.val_ts is not None:
               self.val_forecast = self.results.forecast(
                  steps=len(self.val_ts),
                  exog=self.val_exog
               )
               self._print_metrics(self.val_ts, self.val_forecast, "VALIDATION")

         print(f"⏱️ Temps d'entraînement : {time.time() - start:.2f}s")

      except Exception as e:
         print(f"❌ Erreur lors de l'entraînement : {e}")
         return False

      return True


   def evaluate_on_test(self):
      if self.results is None:
         print("⚠️ Modèle non entraîné.")
         return None

      self.test_forecast = self.results.forecast(
         steps=len(self.test_ts),
         exog=self.test_exog
      )
      return self._print_metrics(self.test_ts, self.test_forecast, "TEST")


   def _print_metrics(self, y_true, y_pred, dataset_name):
      mae = mean_absolute_error(y_true, y_pred)
      rmse = np.sqrt(mean_squared_error(y_true, y_pred))
      r2 = r2_score(y_true, y_pred)
      mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
      aic = self.results.aic if self.results else None

      print(f"\n📊 Métriques {dataset_name}:")
      print(f"MAE  : {mae:.3f}")
      print(f"RMSE : {rmse:.3f}")
      print(f"R²   : {r2:.3f}")
      print(f"MAPE : {mape:.2f}%")
      if aic:
         print(f"AIC  : {aic:.2f}")

      return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'aic': aic}


   def search_best_sarimax(self, p_range=range(0, 3), d_range=[0, 1], q_range=range(0, 3),
                           sp_range=range(0, 2), sd_range=[0, 1], sq_range=range(0, 2),
                           seasonal_period=24, metric='rmse', max_iterations=None):
      if self.val_ts is None:
         print("⚠️ Pas de set de validation fourni.")
         return None, None

      best_score = np.inf
      grid = list(product(p_range, d_range, q_range, sp_range, sd_range, sq_range))
      if max_iterations:
         grid = grid[:max_iterations]

      print(f"🔍 Recherche sur {len(grid)} combinaisons...")

      with warnings.catch_warnings():
         warnings.simplefilter("ignore")

         for p, d, q, sp, sd, sq in tqdm(grid, desc="Grid Search SARIMAX"):
            try:
               model = SARIMAX(
                  self.train_ts,
                  exog=self.train_exog,
                  order=(p, d, q),
                  seasonal_order=(sp, sd, sq, seasonal_period),
                  enforce_stationarity=False,
                  enforce_invertibility=False
               )
               results = model.fit(disp=False)
               forecast = results.forecast(steps=len(self.val_ts), exog=self.val_exog)

               if metric == 'rmse':
                  score = np.sqrt(mean_squared_error(self.val_ts, forecast))
               elif metric == 'mae':
                  score = mean_absolute_error(self.val_ts, forecast)
               elif metric == 'aic':
                  score = results.aic
               else:
                  score = np.sqrt(mean_squared_error(self.val_ts, forecast))

               self.search_history.append({
                  'order': (p, d, q),
                  'seasonal_order': (sp, sd, sq, seasonal_period),
                  'score': score,
                  'aic': results.aic
               })

               if score < best_score:
                  best_score = score
                  self.best_order = (p, d, q)
                  self.best_seasonal_order = (sp, sd, sq, seasonal_period)

            except Exception:
               continue

      if self.best_order:
         print(f"\n✅ Meilleur SARIMAX: {self.best_order}x{self.best_seasonal_order}")
         print(f"📈 {metric.upper()}: {best_score:.3f}")
         self.train(order=self.best_order, seasonal_order=self.best_seasonal_order)

      return self.best_order, self.best_seasonal_order

   def predict_future(self, steps=24, future_exog=None):
      if self.results is None:
         print("⚠️ Modèle non entraîné.")
         return None

      full_y = pd.concat([self.train_ts, self.val_ts, self.test_ts])
      full_X = pd.concat([self.train_exog, self.val_exog, self.test_exog])

      model = SARIMAX(
         full_y,
         exog=full_X,
         order=self.results.model.order,
         seasonal_order=self.results.model.seasonal_order,
         enforce_stationarity=False,
         enforce_invertibility=False
      )
      results = model.fit(disp=False)

      forecast = results.get_forecast(steps=steps, exog=future_exog)
      return {
         'forecast': forecast.predicted_mean,
         'conf_int': forecast.conf_int()
      }

   def plot_predictions(self):
      if self.test_forecast is None:
         print("⚠️ Aucune prédiction disponible.")
         return

      plt.figure(figsize=(15, 5))
      plt.plot(self.test_ts.index, self.test_ts.values, label="Observations", color="blue")
      plt.plot(self.test_ts.index, self.test_forecast.values, label="Prédictions", color="red", linestyle="--")
      plt.title("Test Set - Prédictions SARIMAX")
      plt.xlabel("Date")
      plt.ylabel("Valeur")
      plt.legend()
      plt.grid(True, alpha=0.3)
      plt.tight_layout()
      plt.show()

   def get_model_summary(self):
      return self.results.summary() if self.results else "Modèle non entraîné"
