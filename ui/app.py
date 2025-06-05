import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
from io import BytesIO

# Configuration page
st.set_page_config(
    page_title="🎬 CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""

""", unsafe_allow_html=True)

# Configuration API
API_URL = os.getenv('API_URL', 'http://localhost:8000')
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '')
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

def get_tmdb_movie_info(movie_title):
    """Récupère les informations TMDB d'un film"""
    if not TMDB_API_KEY:
        return None
    
    try:
        # Recherche du film
        search_url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'query': movie_title,
            'language': 'fr-FR'
        }
        
        response = requests.get(search_url, params=params)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                movie = results[0]
                return {
                    'poster_path': movie.get('poster_path'),
                    'overview': movie.get('overview'),
                    'release_date': movie.get('release_date'),
                    'vote_average': movie.get('vote_average')
                }
    except Exception as e:
        st.error(f"Erreur TMDB: {e}")
    
    return None

def get_recommendations(user_id, n_recommendations):
    """Appelle l'API pour obtenir les recommandations"""
    try:
        response = requests.get(
            f"{API_URL}/recommend/{user_id}",
            params={'n': n_recommendations},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API: {e}")
        return None

def display_movie_card(movie, tmdb_info=None):
    """Affiche une carte de film"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Affichage du poster
        if tmdb_info and tmdb_info.get('poster_path'):
            poster_url = f"{TMDB_IMAGE_URL}{tmdb_info['poster_path']}"
            try:
                response = requests.get(poster_url)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    st.image(image, width=150)
                else:
                    st.image("https://via.placeholder.com/150x225?text=No+Poster", width=150)
            except:
                st.image("https://via.placeholder.com/150x225?text=No+Poster", width=150)
        else:
            st.image("https://via.placeholder.com/150x225?text=No+Poster", width=150)
    
    with col2:
        st.markdown(f"**{movie['title']}**")
        st.markdown(f"🎭 *{movie.get('genres', 'N/A')}*")
        
        # Score de recommandation
        score = movie['score']
        st.markdown(f"Score: {score:.2f}", unsafe_allow_html=True)
        
        # Informations TMDB
        if tmdb_info:
            if tmdb_info.get('vote_average'):
                st.markdown(f"⭐ Note TMDB: {tmdb_info['vote_average']}/10")
            if tmdb_info.get('release_date'):
                st.markdown(f"📅 Sortie: {tmdb_info['release_date']}")
            if tmdb_info.get('overview'):
                with st.expander("Synopsis"):
                    st.write(tmdb_info['overview'][:200] + "..." if len(tmdb_info['overview']) > 200 else tmdb_info['overview'])

def main():
    # Header
    st.markdown('🎬 CineMatch', unsafe_allow_html=True)
    st.markdown("### Votre système de recommandation de films personnalisé")
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Paramètres")
    
    # Test de connexion API
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            if health_data['model'] == 'loaded':
                st.sidebar.success("✅ API connectée")
                st.sidebar.success("✅ Modèle chargé")
            else:
                st.sidebar.warning("⚠️ Modèle non chargé")
        else:
            st.sidebar.error("❌ API non disponible")
    except:
        st.sidebar.error("❌ Impossible de se connecter à l'API")
    
    # Interface utilisateur
    user_id = st.sidebar.number_input("ID Utilisateur", min_value=1, max_value=6040, value=1)
    n_recommendations = st.sidebar.slider("Nombre de recommandations", min_value=5, max_value=20, value=10)
    
    # Options avancées
    with st.sidebar.expander("Options avancées"):
        show_tmdb_info = st.checkbox("Afficher infos TMDB", value=True)
        show_charts = st.checkbox("Afficher graphiques", value=True)
    
    # Bouton de recommandation
    if st.sidebar.button("🎬 Obtenir des recommandations", type="primary"):
        with st.spinner("Génération des recommandations..."):
            recommendations = get_recommendations(user_id, n_recommendations)
            
            if recommendations:
                st.session_state['recommendations'] = recommendations
                st.success(f"✅ {len(recommendations['recommendations'])} recommandations générées!")
            else:
                st.error("❌ Impossible d'obtenir les recommandations")
    
    # Affichage des recommandations
    if 'recommendations' in st.session_state:
        recs = st.session_state['recommendations']
        
        # Métriques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👤 Utilisateur", recs['user_id'])
        with col2:
            st.metric("🎬 Films recommandés", len(recs['recommendations']))
        with col3:
            avg_score = sum(movie['score'] for movie in recs['recommendations']) / len(recs['recommendations'])
            st.metric("📊 Score moyen", f"{avg_score:.2f}")
        
        # Graphiques
        if show_charts and recs['recommendations']:
            st.markdown("### 📊 Analyse des recommandations")
            
            # Graphique des scores
            df_recs = pd.DataFrame(recs['recommendations'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_bar = px.bar(
                    df_recs.head(10), 
                    x='title', 
                    y='score',
                    title="Top 10 - Scores de recommandation",
                    color='score',
                    color_continuous_scale='viridis'
                )
                fig_bar.update_xaxis(tickangle=45)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Distribution des genres
                all_genres = []
                for movie in recs['recommendations']:
                    if movie.get('genres'):
                        genres = movie['genres'].split('|')
                        all_genres.extend(genres)
                
                if all_genres:
                    genre_counts = pd.Series(all_genres).value_counts()
                    fig_pie = px.pie(
                        values=genre_counts.values,
                        names=genre_counts.index,
                        title="Répartition des genres"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
        
        # Liste des films
        st.markdown("### 🎬 Vos recommandations")
        
        for i, movie in enumerate(recs['recommendations']):
            with st.container():
                st.markdown(f"#### {i+1}. {movie['title']}")
                
                # Informations TMDB si activées
                tmdb_info = None
                if show_tmdb_info and TMDB_API_KEY:
                    tmdb_info = get_tmdb_movie_info(movie['title'])
                
                display_movie_card(movie, tmdb_info)
                st.markdown("---")
    
    # Statistiques globales
    st.markdown("### 📈 Statistiques de l'application")
    
    try:
        stats_response = requests.get(f"{API_URL}/stats", timeout=5)
        if stats_response.status_code == 200:
            stats = stats_response.json()['last_24h']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🔮 Prédictions 24h", stats['total_predictions'])
            with col2:
                st.metric("👥 Utilisateurs uniques", stats['unique_users'])
            with col3:
                st.metric("🎬 Films uniques", stats['unique_movies'])
            with col4:
                st.metric("📊 Score moyen", f"{stats['avg_score']:.2f}")
        else:
            st.info("Statistiques non disponibles")
    except:
        st.info("Impossible de récupérer les statistiques")
    
    # Footer
    st.markdown("---")
    st.markdown("🎬 **CineMatch** - Système de recommandation alimenté par MovieLens 1M | "
               "🚀 Déployé avec Docker & MLflow")

if __name__ == "__main__":
    main()