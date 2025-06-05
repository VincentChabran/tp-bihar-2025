import sys
import os

# Ajoute le dossier `src/` au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))

from sklearn.model_selection import train_test_split
import argparse
import yaml
import pandas as pd

from ForecastDatabase import ForecastDatabase
from timeseries_forecasting.ArimaTrainer import ArimaTrainer
from timeseries_forecasting.SarimaTrainer import SarimaTrainer
from timeseries_forecasting.SarimaxTrainer import SarimaxTrainer
from timeseries_forecasting.MLRegressorTrainer import MLRegressorTrainer



def parse_range_dicts(param_dict):
    """
    Transforme les dictionnaires de type {'min': 0, 'max': 4} en range Python.
    Appliqué récursivement à tous les paramètres si nécessaire.
    """
    parsed = {}
    for key, val in param_dict.items():
        if isinstance(val, dict) and "min" in val and "max" in val:
            parsed[key] = list(range(val["min"], val["max"] + 1))
        else:
            parsed[key] = val
    return parsed



def split_series(df, target_col, exog_cols=None, ratios=(0.6, 0.2, 0.2)):
    n = len(df)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    target = df[target_col]
    exog = df[exog_cols] if exog_cols else None

    # Assure que les index datetime sont conservés
    train = target.iloc[:n_train].copy()
    val = target.iloc[n_train:n_train + n_val].copy()
    test = target.iloc[n_train + n_val:].copy()

    train_exog = exog.iloc[:n_train].copy() if exog is not None else None
    val_exog = exog.iloc[n_train:n_train + n_val].copy() if exog is not None else None
    test_exog = exog.iloc[n_train + n_val:].copy() if exog is not None else None

    return train, val, test, train_exog, val_exog, test_exog



def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_type = config["model_type"]
    version = config["version"]
    db_path = config["db_path"]
    target_col = config["target_col"]
    exog_cols = config.get("exog_cols", None)

    df = get_or_fetch_weather_data(
        start_date=config["start_date"],
        end_date=config["end_date"],
        lat=config["latitude"],
        lon=config["longitude"],
        db_path=config["db_path"]
    )
    train_ts, val_ts, test_ts, train_exog, val_exog, test_exog = split_series(df, target_col, exog_cols)

    db = ForecastDatabase(db_path)

    if model_type == "arima":
        trainer = ArimaTrainer(train_ts, val_ts, test_ts)
        params = parse_range_dicts(config["arima_params"])
        trainer.search_best_arima(**params)

    elif model_type == "sarima":
        trainer = SarimaTrainer(train_ts, val_ts, test_ts)
        params = parse_range_dicts(config["sarima_params"])
        trainer.search_best_sarima(**params)


    elif model_type == "sarimax":
        trainer = SarimaxTrainer(train_ts, val_ts, test_ts, train_exog, val_exog, test_exog)
        params = parse_range_dicts(config["sarimax_params"])
        trainer.search_best_sarimax(**params)


    elif model_type == "ml":
        X = df[exog_cols]
        y = df[target_col]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, shuffle=False, test_size=0.4)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, shuffle=False, test_size=0.5)

        trainer = MLRegressorTrainer(X_train, y_train, X_val, y_val, X_test, y_test, model_type=config["ml_model"])
        if config.get("ml_params"):
            trainer.search_best_params(config["ml_params"])
        trainer.train()

    else:
        raise ValueError(f"Modèle non reconnu : {model_type}")

    # Évaluation test
    metrics = trainer.evaluate_on_test()

    # Enregistrement du modèle
    params = {"order": getattr(trainer, "best_order", None)}
    if hasattr(trainer, "best_seasonal_order") and trainer.best_seasonal_order:
        params["seasonal_order"] = trainer.best_seasonal_order

    train_period = (train_ts.index[0].isoformat(), train_ts.index[-1].isoformat())
    model_id = db.insert_model(
        model_name=model_type.upper(),
        train_start=train_period[0],
        train_end=train_period[1],
        params_dict=params,
        version=version
    )

    # Prédictions
    df_preds = pd.DataFrame({
        "time": trainer.test_ts.index,
        "target": trainer.test_ts.values,
        "prediction": trainer.test_forecast.values
    })

    db.insert_predictions(
        model_id=model_id,
        timestamps=df_preds["time"].astype(str).tolist(),
        predictions=df_preds["prediction"].tolist(),
        targets=df_preds["target"].tolist(),
        features_list=None
    )

    print(f"✅ Pipeline terminée pour {model_type.upper()}")
    db.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Chemin vers le fichier YAML de config")
    args = parser.parse_args()
    main(args.config)
