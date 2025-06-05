# TPT BIHAR 2025 – Projet Final

## 📌 Objectifs

Ce projet regroupe trois sous-projets :

1. **Classification d’images** (CNNs)
2. **Analyse de sentiments** (Textes - LSTM/NLP)
3. **Prévision de séries temporelles** (ARIMA, ML, MLOps)

Un pipeline **MLOps complet** est mis en place pour le projet de **série temporelle**.

---

## 📁 Structure du projet

```
notebooks/               # Jupyter Notebooks (images, textes, séries)
data/                    # Acquisition des données + base SQLite
model/                   # Scripts de prédiction + modèles
monitoring/              # Comparaison prédictions vs données réelles
api/                     # API FastAPI
.github/workflows/       # CI/CD GitHub Actions
```

---

## 📥 Téléchargement des données d’images

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

## ⚙️ Installation & Exécution

```bash
# Créer un environnement virtuel
python -m venv venv && source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API
uvicorn api.main:app --reload
```

---

## 🔁 CI/CD

Un pipeline GitHub Actions est défini dans `.github/workflows/ci.yml` :

-  Création de l'image Docker
-  Envoi vers `ghcr.io`
-  Exécution des tests automatisés de l'API

---

## 💡 Dépendances clés (requirements.txt)

```
fastapi
uvicorn
scikit-learn
pandas
matplotlib
sqlalchemy
joblib
requests
jupyter
```

Ajouter d’autres selon les besoins (LIME, torch, transformers, etc.).
