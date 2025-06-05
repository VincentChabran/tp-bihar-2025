import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class MLRegressorTrainer:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, model_type='rf', random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.model_type = model_type
        self.random_state = random_state
        self.target_col = y_train.name  # Pour l'affichage

        self.model = self._init_model()

    def _init_model(self):
        if self.model_type == 'rf':
            return RandomForestRegressor(random_state=self.random_state)
        elif self.model_type == 'lr':
            return LinearRegression()
        else:
            raise ValueError("Modèle non supporté. Choisir parmi 'rf', 'lr'.")

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self, on='val'):
        if on == 'val':
            X, y = self.X_val, self.y_val
        elif on == 'test':
            X, y = self.X_test, self.y_test
        else:
            raise ValueError("Choisir 'val' ou 'test' pour l'évaluation")

        preds = self.model.predict(X)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)

        print(f"\n📊 Évaluation ({on.upper()})")
        print(f"MAE  : {mae:.3f}")
        print(f"RMSE : {rmse:.3f}")
        print(f"R²    : {r2:.3f}")

    def plot_predictions(self, on='both'):
        sets = [('val', self.X_val, self.y_val), ('test', self.X_test, self.y_test)] if on == 'both' else [(on, getattr(self, f"X_{on}"), getattr(self, f"y_{on}"))]

        for label, X, y in sets:
            preds = self.model.predict(X)
            df_plot = pd.DataFrame({
                "Vrai": y.values,
                "Prédit": preds
            }, index=y.index)

            plt.figure(figsize=(15, 5))
            plt.plot(df_plot.index, df_plot["Vrai"], label="Vrai", alpha=0.7)
            plt.plot(df_plot.index, df_plot["Prédit"], label="Prédit", alpha=0.7)
            plt.title(f"🎯 Prédiction vs Valeur réelle ({label.upper()})")
            plt.xlabel("Temps")
            plt.ylabel(self.target_col)
            plt.legend()
            plt.tight_layout()
            plt.grid(True)
            plt.show()

    def search_best_params(self, param_grid):
        if not param_grid:
            print("⚠️ Aucun paramètre à rechercher. Utilisation du modèle par défaut.")
            return

        print(f"🔍 Recherche des meilleurs hyperparamètres pour {self.model_type.upper()}...")
        grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search.best_estimator_
        print(f"✅ Meilleurs paramètres: {grid_search.best_params_}")
