import streamlit as st 
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import pickle
import seaborn as sns 
import matplotlib.pyplot as plt
import base64
from PIL import Image
from streamlit_option_menu import option_menu

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Prix Immobilier",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour convertir une image locale en base64
def get_base64(image_file):
    with open(image_file, "rb") as file:
        data = file.read()
    return base64.b64encode(data).decode()

# Fonction pour formater les prix en format monétaire
def format_price(price):
    return f"${price:,.2f}"

# Fonction pour charger et préparer les données
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("data_cleaned.csv")
        return data
    except FileNotFoundError:
        st.error("Fichier de données introuvable. Veuillez vérifier le chemin d'accès.")
        return pd.DataFrame()

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    try:
        model = joblib.load(filename="mon_deuxieme_model.joblib")
        with open("features_list.pkl", "rb") as f:
            feature_columns = pickle.load(f)
        return model, feature_columns
    except (FileNotFoundError, pickle.PickleError) as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None, None

# Chargement du modèle et des features
model, feature_columns = load_model()

# Chargement des données
df = load_data()

# Navigation principale
with st.sidebar:
    st.sidebar.image("house_image2.jpg", width=200)
    st.sidebar.title("Prédiction Immobilière")
    
    selected = option_menu(
        menu_title=None,
        options=[" Accueil", " Prédiction", " Visualisation", " À propos"],
        icons=["house", "magic", "graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Section d'accueil
if selected == " Accueil":
    st.title("Application de prédiction des prix immobiliers 🏡")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Comment ça marche?
        Cette application utilise l'intelligence artificielle pour prédire le prix des maisons en fonction de leurs caractéristiques.
        
        #### Pour obtenir une prédiction:
        1. Accéder à l'onglet 'Prédiction'
        2. Saisir les caractéristiques de la maison
        3. Cliquer sur 'Calculer le prix' pour obtenir une estimation
        
        #### Pour explorer les données:
        1. Consulter l'onglet 'Visualisation'
        2. Découvrer les facteurs qui influencent le plus le prix des maisons
        """)
        
        # Statistiques clés
        if not df.empty:
            st.subheader("Statistiques du marché")
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Prix moyen", format_price(df["SalePrice"].mean()))
            with col_stats2:
                st.metric("Prix médian", format_price(df["SalePrice"].median()))
            with col_stats3:
                st.metric("Superficie moyenne", f"{df['GrLivArea'].mean():.1f} pi²")
    
    with col2:
        st.image("datascientist.png", width=200)
        st.markdown("### Développé par:")
        st.markdown("**Daouda Ba**")
        st.markdown("*Master en MLDS*")
        
        # Bouton pour accéder directement à la prédiction
        if st.button("Faire une prédiction maintenant ➡️", key="home_predict_btn"):
            st.session_state.selected = " Prédiction"
            st.rerun()

# Fonction pour obtenir les caractéristiques de l'utilisateur
def user_input_features():
    # Listes de valeurs pour les champs de type sélection
    neighborhoods = sorted(df["Neighborhood"].unique().tolist()) if not df.empty else ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel"]
    sale_conditions = sorted(df["SaleCondition"].unique().tolist()) if not df.empty else ["Normal", "Abnorml", "Partial", "AdjLand", "Alloca", "Family"]
    kitchen_qualities = ["Ex", "Gd", "TA", "Fa", "Po"]
    
    # Utilisation des statistiques du dataframe pour définir des valeurs par défaut réalistes
    if not df.empty:
        default_year = int(df["YearBuilt"].median())
        default_gr_liv_area = int(df["GrLivArea"].median())
        default_lot_area = int(df["LotArea"].median())
        default_total_bsmt = int(df["TotalBsmtSF"].median())
        default_1st_flr = int(df["1stFlrSF"].median())
        default_2nd_flr = int(df["2ndFlrSF"].median())
        default_garage_area = int(df["GarageArea"].median())
    else:
        default_year = 1980
        default_gr_liv_area = 1500
        default_lot_area = 10000
        default_total_bsmt = 1000
        default_1st_flr = 1000
        default_2nd_flr = 350
        default_garage_area = 400
    
    # Organisation des champs en colonnes et sections
    st.subheader("Caractéristiques de la maison")
    
    # Qualité et état
    col1, col2 = st.columns(2)
    with col1:
        overall_qual = st.slider("Qualité globale", 1, 10, 5, help="10 = Excellente, 1 = Très faible")
    with col2:
        overall_cond = st.slider("État général", 1, 10, 5, help="10 = Excellent, 1 = Très mauvais")
    
    # Taille et superficie
    st.subheader("Superficie")
    col1, col2 = st.columns(2)
    with col1:
        gr_liv_area = st.number_input("Surface habitable (pi²)", 300, 6000, default_gr_liv_area)
        first_flr_sf = st.number_input("Surface 1er étage (pi²)", 300, 5000, default_1st_flr)
    with col2:
        lot_area = st.number_input("Superficie du terrain (pi²)", 1000, 220000, default_lot_area)
        second_flr_sf = st.number_input("Surface 2ème étage (pi²)", 0, 3000, default_2nd_flr)
    
    # Caractéristiques du sous-sol et du garage
    st.subheader("Sous-sol et garage")
    col1, col2 = st.columns(2)
    with col1:
        total_bsmt_sf = st.number_input("Surface du sous-sol (pi²)", 0, 7000, default_total_bsmt)
    with col2:
        garage_cars = st.slider("Places de parking", 0, 5, 2)
        garage_area = st.number_input("Surface du garage (pi²)", 0, 2000, default_garage_area)
    
    # Pièces et commodités
    st.subheader("Pièces et commodités")
    col1, col2, col3 = st.columns(3)
    with col1:
        full_bath = st.slider("Salles de bain complètes", 0, 5, 2)
    with col2:
        half_bath = st.slider("Demi-salles de bain", 0, 3, 1)
    with col3:
        bedroom_abv_gr = st.slider("Chambres", 1, 10, 3)
    
    col1, col2 = st.columns(2)
    with col1:
        kitchen_qual = st.select_slider("Qualité de la cuisine", options=kitchen_qualities, value="TA", 
                                      help="Ex=Excellente, Gd=Bonne, TA=Moyenne, Fa=Passable, Po=Mauvaise")
    with col2:
        fireplaces = st.slider("Nombre de cheminées", 0, 5, 1)
    
    # Informations générales
    st.subheader("Informations générales")
    col1, col2 = st.columns(2)
    with col1:
        year_built = st.number_input("Année de construction", 1872, 2025, default_year)
    with col2:
        neighborhood = st.selectbox("Quartier", neighborhoods)
        sale_condition = st.selectbox("Condition de vente", sale_conditions)

    # Création du dictionnaire des valeurs saisies
    user_data = {
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "YearBuilt": year_built,
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt_sf,
        "1stFlrSF": first_flr_sf,
        "2ndFlrSF": second_flr_sf,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "BedroomAbvGr": bedroom_abv_gr,
        "KitchenQual": kitchen_qual,
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "LotArea": lot_area,
        "Fireplaces": fireplaces,
        "Neighborhood": neighborhood,
        "SaleCondition": sale_condition
    }

    return user_data

# 🔹 Fonction d'inférence : compléter les colonnes et faire la prédiction
def inference(user_data):
    if model is None or feature_columns is None:
        st.error("Le modèle n'a pas pu être chargé correctement.")
        return None
        
    # Convertir les entrées utilisateur en DataFrame
    df = pd.DataFrame([user_data])

    # Compléter les colonnes manquantes avec des valeurs par défaut (0 ou "NA")
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # ou une valeur par défaut adaptée

    # Réorganiser les colonnes dans le même ordre que l'entraînement
    df = df[feature_columns]

    # Faire la prédiction
    prediction = model.predict(df)

    return prediction[0]

# Section de prédiction
if selected == " Prédiction":
    st.title("Prédiction du Prix Immobilier")
    
    # Obtenir les caractéristiques saisies par l'utilisateur
    input_data = user_input_features()
    
    # Résumé des caractéristiques (caché par défaut)
    with st.expander("Résumé des caractéristiques renseignées"):
        st.dataframe(pd.DataFrame([input_data]))
    
    # Création du bouton 'Predict' qui retourne la prédiction du modèle
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Calculer le prix", key="predict_btn", use_container_width=True):
            with st.spinner("Calcul en cours..."):
                # Simuler un délai de calcul
                import time
                time.sleep(1)
                prediction = inference(input_data)
                
                if prediction is not None:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>Prix estimé de la maison</h2>
                        <h1 style="color:#1ABC9C; font-size:3rem;">{format_price(prediction)}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Afficher les facteurs qui ont le plus influencé la prédiction
                    if model is not None and hasattr(model, 'feature_importances_'):
                        st.subheader("Facteurs d'influence")
                        features_df = pd.DataFrame({
                            'Feature': feature_columns,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False).head(5)
                        
                        # Générer un graphique à barres pour les facteurs d'influence
                        fig, ax = plt.subplots(figsize=(10, 4))
                        sns.barplot(x='Importance', y='Feature', data=features_df, palette='viridis', ax=ax)
                        ax.set_title('Top 5 des facteurs influençant le prix')
                        st.pyplot(fig)
                        
                        st.info("💡 **Conseil**: Les facteurs ci-dessus ont le plus d'impact sur le prix de votre maison.")

# Section de visualisation des données
if selected == " Visualisation":
    st.title("Visualisation des Données 📊")
    
    # Vérifier si le dataframe est vide
    if df.empty:
        st.error("Aucune donnée disponible pour la visualisation.")
    else:
        # Menu de sélection pour les visualisations
        viz_option = st.radio(
            "Choisissez un type de visualisation:",
            ["Aperçu des données", "Statistiques descriptives", "Distributions", "Corrélations", "Nuage de points", "Prix par quartier"],
            horizontal=True
        )
        
        if viz_option == "Aperçu des données":
            st.subheader("Aperçu du jeu de données")
            
            # Filtres pour afficher les données
            col1, col2 = st.columns(2)
            with col1:
                sample_size = st.slider("Nombre d'observations à afficher", 5, 100, 20)
            with col2:
                sort_by = st.selectbox("Trier par", ["SalePrice", "YearBuilt", "GrLivArea", "OverallQual"], index=0)
                ascending = st.checkbox("Ordre croissant", False)
            
            # Afficher les données filtrées
            st.dataframe(df.sort_values(by=sort_by, ascending=ascending).head(sample_size))
            
        elif viz_option == "Statistiques descriptives":
            st.subheader("Statistiques descriptives")
            
            # Sélectionner les variables à afficher
            selected_cols = st.multiselect(
                "Sélectionnez les variables à analyser",
                df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                default=["SalePrice", "GrLivArea", "OverallQual", "YearBuilt"]
            )
            
            if selected_cols:
                st.dataframe(df[selected_cols].describe().T)
                
                # Afficher un résumé visuel
                fig, ax = plt.subplots(figsize=(12, 6))
                df[selected_cols].median().plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title("Valeurs médianes")
                ax.set_ylabel("Valeur")
                st.pyplot(fig)
            
        elif viz_option == "Distributions":
            st.subheader("Distributions des variables")
            
            # Sélectionner la variable à analyser
            col1, col2 = st.columns([1, 3])
            with col1:
                feature = st.selectbox("Choisissez une variable", df.select_dtypes(include=['int64', 'float64']).columns.tolist())
                kde = st.checkbox("Ajouter une courbe KDE", True)
                bins = st.slider("Nombre de bins", 10, 100, 30)
            
            # Afficher l'histogramme
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[feature], bins=bins, kde=kde, ax=ax, color='steelblue')
                ax.set_title(f"Distribution de {feature}")
                st.pyplot(fig)
            
            # Afficher quelques statistiques
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Moyenne", f"{df[feature].mean():.2f}")
            with col2:
                st.metric("Médiane", f"{df[feature].median():.2f}")
            with col3:
                st.metric("Écart-type", f"{df[feature].std():.2f}")
            with col4:
                st.metric("Max", f"{df[feature].max():.2f}")
            
        elif viz_option == "Corrélations":
            st.subheader("Matrice de corrélation")
            
            # Sélectionner les variables à inclure
            corr_cols = st.multiselect(
                "Sélectionnez les variables à inclure dans la matrice de corrélation",
                df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                default=["SalePrice", "OverallQual", "YearBuilt", "GrLivArea", "TotalBsmtSF", "FullBath"]
            )
            
            if corr_cols:
                # Calculer la matrice de corrélation
                corr_matrix = df[corr_cols].corr()
                
                # Afficher la matrice de corrélation
                fig, ax = plt.subplots(figsize=(10, 8))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap=cmap, 
                            center=0, square=True, linewidths=.5, ax=ax)
                ax.set_title("Matrice de corrélation")
                st.pyplot(fig)
                
                # Afficher les corrélations avec le prix de vente
                if "SalePrice" in corr_cols:
                    st.subheader("Corrélations avec le prix de vente")
                    price_corr = corr_matrix["SalePrice"].sort_values(ascending=False).drop("SalePrice")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=price_corr.values, y=price_corr.index, ax=ax, palette='viridis')
                    ax.set_title("Corrélation avec le prix de vente")
                    ax.set_xlabel("Coefficient de corrélation")
                    st.pyplot(fig)
                
        elif viz_option == "Nuage de points":
            st.subheader("Nuage de points interactif")
            
            # Sélectionner les variables
            col1, col2, col3 = st.columns(3)
            with col1:
                var_x = st.selectbox("Variable X", df.select_dtypes(exclude="object").columns.tolist(), 
                                   index=df.select_dtypes(exclude="object").columns.get_loc("GrLivArea") if "GrLivArea" in df.columns else 0)
            with col2:
                var_y = st.selectbox("Variable Y", df.select_dtypes(exclude="object").columns.tolist(),
                                   index=df.select_dtypes(exclude="object").columns.get_loc("SalePrice") if "SalePrice" in df.columns else 0)
            with col3:
                color_var = st.selectbox("Variable couleur", df.select_dtypes(include="object").columns.tolist(),
                                      index=df.select_dtypes(include="object").columns.get_loc("Neighborhood") if "Neighborhood" in df.columns else 0)
            
            # Créer le nuage de points interactif
            fig = px.scatter(
                df,
                x=var_x,
                y=var_y,
                color=color_var,
                title=f"{var_y} en fonction de {var_x}",
                hover_data=["YearBuilt", "OverallQual"],
                trendline="ols" if st.checkbox("Ajouter une ligne de tendance", True) else None,
                opacity=0.7
            )
            
            # Personnaliser l'apparence
            fig.update_layout(
                xaxis_title=var_x,
                yaxis_title=var_y,
                legend_title=color_var,
                height=600
            )
            
            # Afficher le graphique
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Prix par quartier":
            st.subheader("Prix par quartier")
            
            # Calculer les prix médians par quartier
            neighborhood_prices = df.groupby("Neighborhood")["SalePrice"].agg(["median", "mean", "count"]).reset_index()
            neighborhood_prices = neighborhood_prices.sort_values("median", ascending=False)
            
            # Créer un graphique à barres
            fig = px.bar(
                neighborhood_prices,
                x="Neighborhood",
                y="median",
                color="count",
                labels={"median": "Prix médian ($)", "count": "Nombre de maisons", "Neighborhood": "Quartier"},
                title="Prix médian par quartier",
                color_continuous_scale="viridis",
                text="median"
            )
            
            # Personnaliser l'apparence
            fig.update_traces(texttemplate='%{text:,.0f} $', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            
            # Afficher le graphique
            st.plotly_chart(fig, use_container_width=True)
            
            # Afficher les statistiques détaillées
            with st.expander("Statistiques détaillées par quartier"):
                st.dataframe(neighborhood_prices)

# Section "À propos"
if selected == " À propos":
    st.title("À propos de cette application ℹ️")
    
    st.markdown("""
    ### Présentation du projet
    Cette application a été développée pour prédire le prix des maisons à partir de leurs caractéristiques. 
    Elle utilise un modèle d'apprentissage automatique entraîné sur un ensemble de données immobilières.
    
    ### Auteur
    **Daouda Ba**
    
    ### Méthodologie
    1. **Collecte des données**: Utilisation d'un jeu de données immobilières.
    2. **Prétraitement**: Nettoyage et transformation des données.
    3. **Modélisation**: Entraînement d'un modèle de xgboost pour prédire les prix.
    4. **Déploiement**: Mise en place de cette application web pour permettre aux utilisateurs de faire des prédictions.
    
    
    ### Contact
    Pour toute question ou suggestion, n'hésitez pas à me contacter à l'adresse: daoudaba4500@gmail.com
    """)
    
    # Afficher les informations sur le modèle
    with st.expander("Informations sur le modèle"):
        if model is not None:
            st.write(f"Type de modèle: {type(model).__name__}")
            if hasattr(model, 'feature_importances_'):
                #st.write("Nombre de caractéristiques utilisées:", len(feature_columns))
                
                # Afficher les 10 caractéristiques les plus importantes
                importances = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.write("Top 10 des caractéristiques les plus importantes:")
                st.dataframe(importances.head(10))
                
                # Graphique des importances
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importances.head(10), ax=ax)
                ax.set_title("Importance des caractéristiques")
                st.pyplot(fig)
        else:
            st.write("Informations sur le modèle non disponibles.")

# Pied de page
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; background-color: rgba(0,0,0,0.05); border-radius: 5px;">
    <p style="margin: 0;">© 2025 | Développé par Daouda Ba</p>
</div>
""", unsafe_allow_html=True)