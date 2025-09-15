# 🌾 TPT BIHAR 2025 – Projet Final

## 📌 Objectifs

Ce projet regroupe trois volets d’apprentissage automatique :

1. **Classification d’images** (CNNs)
2. **Analyse de sentiments** (Textes avec LSTM/NLP)
3. **Prévision de séries temporelles** (ARIMA, ML, MLOps)

Le projet **séries temporelles** intègre une chaîne MLOps complète avec base de données, API FastAPI, scripts de monitoring et pipeline CI/CD.

---

## 📁 Structure du projet

```
notebooks/
│   ├── image_classification.ipynb         # CNNs sur des images
│   ├── text_classification.ipynb          # LSTM / analyse de sentiments
│   └── timeseries_forecasting.ipynb       # Prévision de température

data/
│   ├── acquisition.py                     # Script de récupération des données récentes
│   ├── forecast_results.db                # Base SQLite stockant les prédictions et targets
│   └── configs/                           # Configs YAML pour l'acquisition ET l'entraînement des modèles
│       ├── acquisition_config.yaml         # Configuration de l'acquisition des données météo
│       ├── arima_config.yaml              # Configuration du modèle ARIMA
│       ├── sarima_config.yaml             # Configuration du modèle SARIMA
│       ├── sarimax_config.yaml            # Configuration du modèle SARIMAX
│       └── ml_config.yaml                 # Configuration pour les modèles de machine learning

model/
│   ├── generate_prediction.py             # Script de génération de prédictions à une date donnée
│   ├── train_pipeline.py                  # Entraînement des modèles (ARIMA, ML, etc.) via fichier de config
│   └── registry/                          # Modèles entraînés enregistrés (pickle, joblib, etc.)

monitoring/
│   ├── compare_predictions.py             # Génération de graphiques comparant prédictions et observations
│   └── output/                            # Graphiques générés automatiquement

api/
│   ├── main.py                            # API FastAPI
│   └── logs/                              # Journaux de l'API (app.log)

.github/
    └── workflows/ci.yml                   # Pipeline CI/CD GitHub Actions
```

---

## ⚙️ Installation

```bash
# Cloner le repo
git clone <repo-url>
cd tp-bihar-2025

# Créer un environnement virtuel
python -m venv venv && source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Lancer l'API

```bash
uvicorn api.main:app --reload
```

Disponible sur `http://localhost:8000`

---

## 🥪 Tester l’API

**Récupérer les prédictions à une date donnée** :

```bash
curl -X GET "http://localhost:8000/predict?date=2024-01-06" | jq

curl -X GET "http://localhost:8000/predict?date=2024-09-06" | jq
```

**Obtenir les prédictions combinées avec targets** :

```bash
curl -X POST "http://localhost:8000/combined" \
     -H "Content-Type: application/json" \
     -d '{"start_date": "2024-01-01", "end_date": "2024-01-07"}' | jq
```

**Version logicielle** :

```bash
curl -X GET "http://localhost:8000/version" | jq
```

---

## 📊 Génération de prédictions (hors API)

```bash
# Exemple : prédictions pour le 2024-01-06
python model/predict.py 2024-01-06
```

Ce script lit directement la base `data/forecast_results.db` et exporte un CSV des résultats.

---

## 📉 Monitoring (visualisation)

```bash
python monitoring/compare_predictions.py --date 2023-12-06
```

Une image de comparaison est générée dans `monitoring/output/`.

---

## 🏠 Acquisition des données

Configurer le fichier `configs/acquisition_config.yaml`, puis exécuter :

```bash
python data/acquisition.py
```

Cela stocke les nouvelles données dans la base SQLite.

##### 📥 Téléchargement des données d’images

Les données de classification d’images ne peuvent pas être versionnées avec le dépôt Git.

Veuillez suivre les étapes suivantes pour préparer les données :

1. Rendez-vous sur Kaggle : [https://www.kaggle.com/datasets/rodrigonuneswessner/labeledcorndataset](https://www.kaggle.com/datasets/rodrigonuneswessner/labeledcorndataset)
2. Cliquez sur **Download** (vous devez avoir un compte Kaggle)
3. Décompressez l’archive téléchargée
4. Placez manuellement le dossier `ImagensTCCRotuladas/` dans le dossier `./data/`

La structure finale attendue est :

```
data/
└── ImagensTCCRotuladas/
    ├── Train/
    ├── Validation/
    └── Test/
```

Ces dossiers contiennent déjà les images triées par classes (`Chao`, `Ervas`, `Milho`, `Milho_ervas`).

---

## 🧱 Entraînement des modèles

```bash
python model/train_pipeline.py --config configs/arima_config.yaml
```

Les fichiers YAML dans `configs/` définissent les hyperparamètres pour chaque type de modèle :

-  `arima_config.yaml`
-  `sarima_config.yaml`
-  `sarimax_config.yaml`
-  `ml_config.yaml`

Les modèles sont sauvegardés automatiquement dans `model/registry/`.

---

## 📦 Dockerisation

```bash
docker build -t forecast-api .
docker run -p 8000:8000 forecast-api
```

---

## 🔁 CI/CD GitHub Actions

La pipeline CI/CD effectue :

-  ✅ Build de l'image Docker
-  🚀 Push sur GitHub Container Registry (GHCR)
-  🨮 Lancement des tests `pytest`
-  🔔 Notifications sur les erreurs

Elle est déclenchée automatiquement à chaque **push** dans le dépôt.

---

## 🐾 Journaux d'exécution (Logging)

L’API écrit ses journaux dans `api/logs/app.log`. Les événements enregistrés incluent :

-  ✅ Requêtes entrantes (`/predict`, `/combined`, `/version`)
-  🧐 Nombre de prédictions chargées
-  ❌ Erreurs de base de données (fichiers absents ou vides)
-  💡 Commit ID de la version en cours

---

## 💲 Dépendances clés

```text
fastapi
uvicorn
pandas
scikit-learn
statsmodels
sqlalchemy
matplotlib
jupyter
pytest
joblib
httpx
```

---
