import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
import warnings
import time
from scipy import stats
from statsmodels.tsa.stattools import acf


class SarimaTrainer:
   def __init__(self, train_ts, val_ts, test_ts):
      self.train_ts = train_ts
      self.val_ts = val_ts
      self.test_ts = test_ts
      self.model = None
      self.results = None
      self.val_forecast = None
      self.test_forecast = None
      self.best_order = None
      self.best_seasonal_order = None
      self.search_history = []  # Historique des recherches


   def train(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24), validate=True):
      """Entraîne le modèle SARIMA avec les paramètres spécifiés"""
      print(f"📦 Entraînement SARIMA{order}x{seasonal_order}")
      start_time = time.time()
      
      try:
         self.model = SARIMAX(self.train_ts,
                              order=order,
                              seasonal_order=seasonal_order,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
         self.results = self.model.fit(disp=False)

         if validate and self.val_ts is not None:
               self.val_forecast = self.results.forecast(steps=len(self.val_ts))
               self._print_metrics(self.val_ts, self.val_forecast, "VALIDATION")
         
         training_time = time.time() - start_time
         print(f"⏱️ Temps d'entraînement: {training_time:.2f}s")

      except Exception as e:
         print(f"❌ Erreur lors de l'entraînement: {e}")
         return False

      return True


   def evaluate_on_test(self):
      """Évalue le modèle sur le set de test"""
      if self.results is None:
         print("⚠️ Modèle non entraîné.")
         return None

      self.test_forecast = self.results.forecast(steps=len(self.test_ts))
      return self._print_metrics(self.test_ts, self.test_forecast, "TEST")


   def _print_metrics(self, y_true, y_pred, dataset_name):
      """Calcule et affiche les métriques de performance"""
      mae = mean_absolute_error(y_true, y_pred)
      rmse = np.sqrt(mean_squared_error(y_true, y_pred))
      r2 = r2_score(y_true, y_pred)
      aic = self.results.aic if self.results else None
      
      # Calcul MAPE (Mean Absolute Percentage Error)
      mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

      # Seuils adaptatifs basés sur les données
      std_y = np.std(y_true)
      mae_icon = "😊" if mae < std_y * 0.5 else "😐" if mae < std_y else "😞"
      rmse_icon = "😊" if rmse < std_y * 0.6 else "😐" if rmse < std_y else "😞"
      r2_icon = "😊" if r2 > 0.8 else "😐" if r2 > 0.5 else "😞"
      mape_icon = "😊" if mape < 10 else "😐" if mape < 20 else "😞"

      print(f"\n📊 Métriques {dataset_name}:")
      print(f"{mae_icon} MAE  : {mae:.3f}")
      print(f"{rmse_icon} RMSE : {rmse:.3f}")
      print(f"{r2_icon} R²    : {r2:.3f}")
      print(f"{mape_icon} MAPE : {mape:.2f}%")
      if aic:
         aic_icon = "📉" if aic < 500 else "📈"
         print(f"{aic_icon} AIC  : {aic:.2f}")

      return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'aic': aic}


   def search_best_sarima(self, p_range=range(0, 3), d_range=[0, 1], q_range=range(0, 3),
                        sp_range=range(0, 2), sd_range=[0, 1], sq_range=range(0, 2),
                        seasonal_period=24, metric='rmse', verbose=True, max_iterations=None):
      """Recherche les meilleurs hyperparamètres SARIMA"""
      if self.val_ts is None:
         print("⚠️ Pas de set de validation fourni.")
         return None, None

      best_score = np.inf
      best_order = None
      best_seasonal_order = None
      failed_combinations = 0
      
      grid = list(product(p_range, d_range, q_range, sp_range, sd_range, sq_range))
      
      # Limiter le nombre d'itérations si spécifié
      if max_iterations and len(grid) > max_iterations:
         grid = grid[:max_iterations]
         print(f"⚠️ Limitation à {max_iterations} combinaisons sur {len(list(product(p_range, d_range, q_range, sp_range, sd_range, sq_range)))}")

      print(f"🔍 Recherche sur {len(grid)} combinaisons...")
      start_time = time.time()
      
      with warnings.catch_warnings():
         warnings.simplefilter("ignore")

         for i, (p, d, q, sp, sd, sq) in enumerate(tqdm(grid, desc="Grid Search SARIMA", disable=not verbose)):
               try:
                  model = SARIMAX(self.train_ts,
                                 order=(p, d, q),
                                 seasonal_order=(sp, sd, sq, seasonal_period),
                                 enforce_stationarity=False,
                                 enforce_invertibility=False)
                  results = model.fit(disp=False)
                  forecast = results.forecast(steps=len(self.val_ts))

                  # Calcul de la métrique
                  if metric == 'rmse':
                     score = np.sqrt(mean_squared_error(self.val_ts, forecast))
                  elif metric == 'mae':
                     score = mean_absolute_error(self.val_ts, forecast)
                  elif metric == 'aic':
                     score = results.aic
                  elif metric == 'mape':
                     score = np.mean(np.abs((self.val_ts - forecast) / self.val_ts)) * 100
                  else:
                     score = np.sqrt(mean_squared_error(self.val_ts, forecast))

                  # Stocker dans l'historique
                  self.search_history.append({
                     'order': (p, d, q),
                     'seasonal_order': (sp, sd, sq, seasonal_period),
                     'score': score,
                     'aic': results.aic,
                     'metric': metric
                  })

                  if score < best_score:
                     best_score = score
                     best_order = (p, d, q)
                     best_seasonal_order = (sp, sd, sq, seasonal_period)

               except Exception as e:
                  failed_combinations += 1
                  if verbose and failed_combinations % 10 == 0:
                     print(f"⚠️ {failed_combinations} combinaisons échouées...")
                  continue

      search_time = time.time() - start_time
      success_rate = ((len(grid) - failed_combinations) / len(grid)) * 100

      if best_order:
         self.best_order = best_order
         self.best_seasonal_order = best_seasonal_order
         print(f"\n✅ Meilleur SARIMA: {best_order}x{best_seasonal_order}")
         print(f"📈 {metric.upper()}: {best_score:.3f}")
         print(f"⏱️ Temps de recherche: {search_time:.1f}s")
         print(f"📊 Taux de succès: {success_rate:.1f}%")
         
         # Réentraîner avec les meilleurs paramètres
         self.train(order=best_order, seasonal_order=best_seasonal_order)
      else:
         print("❌ Aucun modèle valide trouvé.")

      return best_order, best_seasonal_order


   def get_search_summary(self, top_n=5):
      """Affiche un résumé des meilleures combinaisons trouvées"""
      if not self.search_history:
         print("Aucune recherche effectuée.")
         return
      
      # Trier par score
      sorted_results = sorted(self.search_history, key=lambda x: x['score'])
      
      print(f"\n🏆 Top {min(top_n, len(sorted_results))} des combinaisons:")
      print("Rang | SARIMA | Score | AIC")
      print("-" * 40)
      
      for i, result in enumerate(sorted_results[:top_n], 1):
         order = result['order']
         s_order = result['seasonal_order'][:3]  # Sans la période
         score = result['score']
         aic = result['aic']
         print(f"{i:2d}   | {order}x{s_order} | {score:.3f} | {aic:.1f}")


   def plot_predictions(self, show_validation=False, figsize=(15, 6)):
      """Visualise les prédictions avec options avancées"""
      if self.test_forecast is None and self.val_forecast is None:
         print("⚠️ Aucune prédiction disponible.")
         return

      n_plots = 2 if show_validation and self.val_forecast is not None else 1
      fig, axes = plt.subplots(1, n_plots, figsize=figsize)
      if n_plots == 1:
         axes = [axes]

      # Plot test
      if self.test_forecast is not None:
         ax = axes[0]
         ax.plot(self.test_ts.index, self.test_ts.values, 
                  label="Observations", linewidth=2, color='blue', alpha=0.8)
         ax.plot(self.test_ts.index, self.test_forecast.values, 
                  label="Prédictions", linewidth=2, color='red', linestyle='--', alpha=0.9)
         
         # Calcul des résidus pour l'affichage
         residuals = self.test_ts.values - self.test_forecast.values
         rmse = np.sqrt(np.mean(residuals**2))
         
         ax.set_title(f"Test Set - SARIMA {self.best_order}x{self.best_seasonal_order[:3]}\nRMSE: {rmse:.3f}")
         ax.set_xlabel("Date")
         ax.set_ylabel("Valeur")
         ax.legend()
         ax.grid(True, alpha=0.3)

      # Plot validation
      if show_validation and self.val_forecast is not None:
         ax = axes[1]
         ax.plot(self.val_ts.index, self.val_ts.values, 
                  label="Observations", linewidth=2, color='blue', alpha=0.8)
         ax.plot(self.val_ts.index, self.val_forecast.values, 
                  label="Prédictions", linewidth=2, color='orange', linestyle='--', alpha=0.9)
         
         val_residuals = self.val_ts.values - self.val_forecast.values
         val_rmse = np.sqrt(np.mean(val_residuals**2))
         
         ax.set_title(f"Validation Set\nRMSE: {val_rmse:.3f}")
         ax.set_xlabel("Date")
         ax.set_ylabel("Valeur")
         ax.legend()
         ax.grid(True, alpha=0.3)

      plt.tight_layout()
      plt.show()



   def predict_future(self, steps=24):
      """Prédictions futures avec intervalles de confiance"""
      if self.results is None:
         print("⚠️ Modèle non entraîné.")
         return None
      
      # Réentraîner sur toutes les données disponibles
      full_series = pd.concat([self.train_ts, self.val_ts, self.test_ts])
      model_full = SARIMAX(full_series, 
                           order=self.results.model.order,
                           seasonal_order=self.results.model.seasonal_order,
                           enforce_stationarity=False,
                           enforce_invertibility=False)
      results_full = model_full.fit(disp=False)
      
      # Prédictions avec intervalles de confiance
      forecast = results_full.get_forecast(steps=steps)
      forecast_mean = forecast.predicted_mean
      forecast_ci = forecast.conf_int()
      
      return {
         'forecast': forecast_mean,
         'conf_int': forecast_ci,
         'lower': forecast_ci.iloc[:, 0],
         'upper': forecast_ci.iloc[:, 1]
      }


   def get_model_summary(self):
      """Résumé détaillé du modèle"""
      if self.results is None:
         return "Modèle non entraîné"
      
      return self.results.summary()