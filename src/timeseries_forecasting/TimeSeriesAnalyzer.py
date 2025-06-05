import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 6)

# === Classe utilitaire d'analyse exploratoire ===
class TimeSeriesAnalyzer:

   @staticmethod
   def exploratory_analysis(df):
      """Analyse exploratoire de la température"""
      print("=== ANALYSE EXPLORATOIRE ===")
      print(f"Période des données: {df['time'].min()} à {df['time'].max()}")
      print(f"Nombre de points: {len(df)}")
      print(f"Valeurs manquantes: {df['temperature_2m'].isna().sum()}")
      print(f"Température moyenne: {df['temperature_2m'].mean():.2f}°C")
      print(f"Écart-type: {df['temperature_2m'].std():.2f}°C")
      print(f"Min/Max: {df['temperature_2m'].min():.2f}°C / {df['temperature_2m'].max():.2f}°C")

      # Graphiques
      fig, axes = plt.subplots(2, 2, figsize=(15, 10))

      # Série temporelle
      axes[0, 0].plot(df['time'], df['temperature_2m'])
      axes[0, 0].set_title('Série temporelle')
      axes[0, 0].set_xlabel('Date')
      axes[0, 0].set_ylabel('Température (°C)')

      # Histogramme
      axes[0, 1].hist(df['temperature_2m'], bins=50, alpha=0.7)
      axes[0, 1].set_title('Distribution des températures')

      # Moyenne mensuelle
      monthly = df.copy()
      monthly['month'] = monthly['time'].dt.month
      monthly_stats = monthly.groupby('month')['temperature_2m'].agg(['mean', 'std'])
      axes[1, 0].plot(monthly_stats.index, monthly_stats['mean'], marker='o')
      axes[1, 0].fill_between(monthly_stats.index,
                              monthly_stats['mean'] - monthly_stats['std'],
                              monthly_stats['mean'] + monthly_stats['std'],
                              alpha=0.3)
      axes[1, 0].set_title('Variation saisonnière')

      # Moyenne par heure
      hourly = df.copy()
      hourly['hour'] = hourly['time'].dt.hour
      hourly_stats = hourly.groupby('hour')['temperature_2m'].agg(['mean', 'std'])
      axes[1, 1].plot(hourly_stats.index, hourly_stats['mean'], marker='o')
      axes[1, 1].fill_between(hourly_stats.index,
                              hourly_stats['mean'] - hourly_stats['std'],
                              hourly_stats['mean'] + hourly_stats['std'],
                              alpha=0.3)
      axes[1, 1].set_title('Variation journalière')

      plt.tight_layout()
      plt.show()

   @staticmethod
   def decompose_series(df, period=8*365):
      """Décomposition en tendance / saisonnalité / résidus"""
      ts_data = df.set_index('time')['temperature_2m']

      if len(ts_data) < period * 2:
         print(f"⚠️ Série trop courte pour une période={period}. Réduction à {period//2}")
         period = period // 2

      decomposition = seasonal_decompose(ts_data, model='additive', period=period)

      fig, axes = plt.subplots(4, 1, figsize=(15, 12))
      decomposition.observed.plot(ax=axes[0], title='Série originale')
      decomposition.trend.plot(ax=axes[1], title='Tendance')
      decomposition.seasonal.plot(ax=axes[2], title='Saisonnalité')
      decomposition.resid.plot(ax=axes[3], title='Résidus')

      plt.tight_layout()
      plt.show()

      return decomposition








   
   # def prepare_data_for_ml(self, n_lags=24, test_size=0.2):
   #    """Préparation des données pour les modèles ML"""
   #    df = self.data.copy()
      
   #    # Création des variables retardées (lags)
   #    for i in range(1, n_lags + 1):
   #       df[f'temp_lag_{i}'] = df['temperature_2m'].shift(i)
      
   #    # Variables temporelles
   #    df['hour'] = df['time'].dt.hour
   #    df['day_of_week'] = df['time'].dt.dayofweek
   #    df['month'] = df['time'].dt.month
   #    df['day_of_year'] = df['time'].dt.dayofyear
      
   #    # Variables cycliques pour capturer la périodicité
   #    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
   #    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
   #    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
   #    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
      
   #    # Moyennes mobiles
   #    df['temp_ma_24'] = df['temperature_2m'].rolling(window=24, min_periods=1).mean()
   #    df['temp_ma_168'] = df['temperature_2m'].rolling(window=168, min_periods=1).mean()  # 7 jours
      
   #    # Suppression des lignes avec des NaN
   #    df = df.dropna()
      
   #    # Séparation des features et target
   #    feature_cols = [col for col in df.columns if col not in ['time', 'temperature_2m']]
   #    X = df[feature_cols]
   #    y = df['temperature_2m']
      
   #    # Division train/test
   #    split_idx = int(len(df) * (1 - test_size))
      
   #    self.X_train = X.iloc[:split_idx]
   #    self.X_test = X.iloc[split_idx:]
   #    self.y_train = y.iloc[:split_idx]
   #    self.y_test = y.iloc[split_idx:]
   #    self.test_dates = df['time'].iloc[split_idx:]
      
   #    print(f"Données d'entraînement: {len(self.X_train)} échantillons")
   #    print(f"Données de test: {len(self.X_test)} échantillons")
      
   #    return self.X_train, self.X_test, self.y_train, self.y_test
   

   # def train_ml_models(self):
   #    """Entraînement des modèles ML"""
   #    print("=== ENTRAÎNEMENT DES MODÈLES ML ===")
      
   #    # Normalisation des données
   #    scaler = StandardScaler()
   #    X_train_scaled = scaler.fit_transform(self.X_train)
   #    X_test_scaled = scaler.transform(self.X_test)
      
   #    models = {
   #       'Linear Regression': LinearRegression(),
   #       'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
   #       'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
   #    }
      
   #    for name, model in models.items():
   #       print(f"Entraînement: {name}")
         
   #       if name == 'Linear Regression':
   #             model.fit(X_train_scaled, self.y_train)
   #             predictions = model.predict(X_test_scaled)
   #       else:
   #             model.fit(self.X_train, self.y_train)
   #             predictions = model.predict(self.X_test)
         
   #       self.models[name] = model
   #       self.predictions[name] = predictions
         
   #       # Calcul des métriques
   #       mae = mean_absolute_error(self.y_test, predictions)
   #       rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
   #       r2 = r2_score(self.y_test, predictions)
         
   #       self.metrics[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
         
   #       print(f"  MAE: {mae:.3f}")
   #       print(f"  RMSE: {rmse:.3f}")
   #       print(f"  R²: {r2:.3f}")
   




   # def train_arima(self):
   #    """Entraînement du modèle ARIMA"""
   #    print("=== ENTRAÎnEMENT ARIMA ===")

   #    ts_data = self.data.set_index('time')['temperature_2m']
   #    split_point = int(len(ts_data) * 0.8)
   #    train_ts = ts_data.iloc[:split_point]
   #    test_ts = ts_data.iloc[split_point:]

   #    try:
   #       print("ARIMA(2,1,2)")
   #       model = ARIMA(train_ts, order=(2,1,2))
   #       fit = model.fit()
   #       forecast = fit.forecast(steps=len(test_ts))

   #       self.models['ARIMA'] = fit
   #       self.predictions['ARIMA'] = forecast

   #       mae = mean_absolute_error(test_ts, forecast)
   #       rmse = np.sqrt(mean_squared_error(test_ts, forecast))
   #       r2 = r2_score(test_ts, forecast)

   #       self.metrics['ARIMA'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
   #       print(f"  MAE: {mae:.3f}")
   #       print(f"  RMSE: {rmse:.3f}")
   #       print(f"  R²: {r2:.3f}")

   #    except Exception as e:
   #       print(f"Erreur ARIMA: {e}")


   # def reconstruct_arima_predictions(self):
   #    """
   #    Reconstruit les prédictions ARIMA (différenciées) en températures réelles.
   #    Nécessite que train_arima() ait été exécuté.
   #    """
   #    if 'ARIMA' not in self.predictions:
   #       print("❌ Aucun résultat ARIMA trouvé. Veuillez exécuter train_arima() d'abord.")
   #       return None

   #    # Série de test (différenciée)
   #    ts_data = self.data.set_index('time')['temperature_2m']
   #    split_point = int(len(ts_data) * 0.8)
   #    test_index = ts_data.iloc[split_point:].index

   #    # Dernière valeur connue avant la prédiction
   #    last_train_value = ts_data.iloc[split_point - 1]

   #    # Reconstruction par cumul
   #    pred_diff = self.predictions['ARIMA']
   #    pred_reconstructed = pred_diff.cumsum() + last_train_value

   #    # Sauvegarde dans les prédictions
   #    self.predictions['ARIMA_reconstructed'] = pd.Series(pred_reconstructed.values, index=test_index)

   #    print("✅ Prédictions ARIMA reconstruites en températures réelles.")
   #    return self.predictions['ARIMA_reconstructed']



   # def train_sarima(self):
   #    """Entraînement du modèle SARIMA"""
   #    print("=== ENTRAÎnEMENT SARIMA ===")

   #    ts_data = self.data.set_index('time')['temperature_2m']
   #    split_point = int(len(ts_data) * 0.8)
   #    train_ts = ts_data.iloc[:split_point]
   #    test_ts = ts_data.iloc[split_point:]

   #    try:
   #       print("SARIMA(1,1,1)(1,1,1,24)")
   #       model = SARIMAX(train_ts, order=(1,1,1), seasonal_order=(1,1,1,24))
   #       fit = model.fit()
   #       forecast = fit.forecast(steps=len(test_ts))

   #       self.models['SARIMA'] = fit
   #       self.predictions['SARIMA'] = forecast

   #       mae = mean_absolute_error(test_ts, forecast)
   #       rmse = np.sqrt(mean_squared_error(test_ts, forecast))
   #       r2 = r2_score(test_ts, forecast)

   #       self.metrics['SARIMA'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
   #       print(f"  MAE: {mae:.3f}")
   #       print(f"  RMSE: {rmse:.3f}")
   #       print(f"  R²: {r2:.3f}")

   #    except Exception as e:
   #       print(f"Erreur SARIMA: {e}")
   



   # def compare_models(self):
   #    """Comparaison des performances des modèles"""
   #    print("\n=== COMPARAISON DES MODÈLES ===")
      
   #    metrics_df = pd.DataFrame(self.metrics).T
   #    print(metrics_df.round(3))
      
   #    # Graphique de comparaison
   #    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
      
   #    # MAE
   #    mae_values = [self.metrics[model]['MAE'] for model in self.metrics.keys()]
   #    axes[0,0].bar(self.metrics.keys(), mae_values)
   #    axes[0,0].set_title('Mean Absolute Error (MAE)')
   #    axes[0,0].set_ylabel('MAE')
   #    axes[0,0].tick_params(axis='x', rotation=45)
      
   #    # RMSE
   #    rmse_values = [self.metrics[model]['RMSE'] for model in self.metrics.keys()]
   #    axes[0,1].bar(self.metrics.keys(), rmse_values)
   #    axes[0,1].set_title('Root Mean Square Error (RMSE)')
   #    axes[0,1].set_ylabel('RMSE')
   #    axes[0,1].tick_params(axis='x', rotation=45)
      
   #    # R²
   #    r2_values = [self.metrics[model]['R2'] for model in self.metrics.keys()]
   #    axes[1,0].bar(self.metrics.keys(), r2_values)
   #    axes[1,0].set_title('R-squared (R²)')
   #    axes[1,0].set_ylabel('R²')
   #    axes[1,0].tick_params(axis='x', rotation=45)
      
   #    # Prédictions vs réalité (exemple avec le meilleur modèle)
   #    best_model = min(self.metrics.keys(), key=lambda x: self.metrics[x]['RMSE'])
      
   #    if hasattr(self, 'test_dates'):
   #       axes[1,1].plot(self.test_dates.iloc[:100], self.y_test.iloc[:100], 
   #                      label='Réel', alpha=0.7)
   #       axes[1,1].plot(self.test_dates.iloc[:100], self.predictions[best_model][:100], 
   #                      label=f'Prédiction ({best_model})', alpha=0.7)
   #    axes[1,1].set_title(f'Prédictions vs Réalité - {best_model}')
   #    axes[1,1].set_xlabel('Date')
   #    axes[1,1].set_ylabel('Température (°C)')
   #    axes[1,1].legend()
   #    axes[1,1].tick_params(axis='x', rotation=45)
      
   #    plt.tight_layout()
   #    plt.show()
      
   #    return best_model

