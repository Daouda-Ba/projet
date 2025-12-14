# Prédiction de prix immobilier

Application Streamlit permettant d'estimer le prix d'une maison à partir de ses caractéristiques et d'explorer le jeu de données utilisé pour l'entraînement.

## Fonctionnalités
- Saisie guidée des caractéristiques clés (surface, quartier, qualité, etc.)
- Inférence avec un modèle de machine learning pré-entraîné
- Visualisations interactives (distributions, corrélations, nuages de points, prix par quartier)
- Exposition des facteurs influents via les importances de variables

## Prérequis
- Python 3.9+
- Dépendances listées dans `requirements.txt`
- Fichiers fournis :
  - `data_cleaned.csv` (jeu de données pour l'exploration)
  - `mon_deuxieme_model.joblib` (modèle entraîné)
  - `one_hot_encoder.joblib` (encodeur pour les variables catégorielles)
  - `features_list.pkl` (liste ordonnée des features attendues par le modèle)

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Sous Windows : .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Lancer l'application
```bash
streamlit run interface_streamlit.py
```

Ouvrez ensuite l'URL affichée par Streamlit (par défaut http://localhost:8501) pour accéder à l'interface.

## Dépannage
- Si les fichiers de modèle ou de données sont absents, l'application affichera un message d'erreur. Vérifiez que les fichiers mentionnés dans la section *Prérequis* sont présents à la racine du projet.
- En cas de problème de dépendances, assurez-vous d'utiliser la version de Python recommandée et de réinstaller les packages : `pip install --upgrade --force-reinstall -r requirements.txt`.

## Structure du projet
- `interface_streamlit.py` : application principale Streamlit
- `data_cleaned.csv` : données préparées pour l'exploration
- `*.joblib` / `features_list.pkl` : artefacts du modèle
- Ressources visuelles : `house_image.jpg`, `house_image2.jpg`, `datascientist.png`

## Licence
Projet fourni sans garantie ; utilisez et modifiez librement dans un cadre pédagogique ou expérimental.
