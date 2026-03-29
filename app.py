"""
app.py
------
The Streamlit web dashboard for the Music-Based Recommender.
Users select their music preferences and get personalised
book and movie recommendations.

Run the app with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model import get_recommendations, train_model, save_model, load_model
from preprocessing import run_preprocessing


# ─────────────────────────────────────────────
# PAGE SETTINGS — must be the very first command
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="wide"
)


# ─────────────────────────────────────────────
# LOAD DATA AND MODEL (runs once, cached)
# ─────────────────────────────────────────────

@st.cache_data
def load_all_data():
    """
    Loads and preprocesses all data once.
    st.cache_data means it only runs once — not every time
    the user clicks a button. Makes the app much faster.
    """
    music_df, books_df, movies_df, feature_matrix, scaler = run_preprocessing()
    return music_df, books_df, movies_df, feature_matrix, scaler


@st.cache_resource
def load_trained_model():
    """
    Loads the saved model from disk.
    If model.pkl does not exist yet, trains it first.
    """
    try:
        kmeans, scaler = load_model()
    except FileNotFoundError:
        # Model not trained yet — train it now
        music_df, books_df, movies_df, feature_matrix, scaler = load_all_data()
        kmeans = train_model(feature_matrix)
        save_model(kmeans, scaler)
    return kmeans, scaler


# Load everything
music_df, books_df, movies_df, feature_matrix, scaler = load_all_data()
kmeans, scaler = load_trained_model()


# ─────────────────────────────────────────────
# SECTION 1 — TITLE AND INTRODUCTION
# ─────────────────────────────────────────────

st.title("🎵 Music-Based Personalised Recommender")
st.markdown(
    "Tell us what music you listen to and we will recommend "
    "**books and movies** that match your music personality."
)
st.markdown("---")


# ─────────────────────────────────────────────
# SECTION 2 — USER INPUT PANEL
# ─────────────────────────────────────────────

st.header("🎧 Your Music Preferences")
st.markdown("Use the sliders and dropdown below to describe your music taste.")

# Split the input area into 3 columns side by side
col1, col2, col3 = st.columns(3)

with col1:
    genre = st.selectbox(
        "Favourite music genre",
        options=[
            "pop", "rock", "hip-hop", "jazz", "classical",
            "electronic", "r&b", "metal", "indie", "country"
        ]
    )
    energy = st.slider(
        "Energy level",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="0 = very calm, 1 = very energetic"
    )

with col2:
    mood = st.selectbox(
        "Current mood",
        options=["Happy", "Chill", "Energetic", "Emotional", "Dark"]
    )
    valence = st.slider(
        "Music positivity (valence)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="0 = very negative/sad, 1 = very positive/happy"
    )

with col3:
    artist = st.text_input(
        "Favourite artist (optional)",
        placeholder="e.g. Taylor Swift"
    )
    tempo = st.slider(
        "Tempo (BPM)",
        min_value=60.0,
        max_value=200.0,
        value=120.0,
        step=1.0,
        help="60 = slow, 120 = moderate, 180+ = very fast"
    )

st.markdown("---")


# ─────────────────────────────────────────────
# SECTION 3 — RECOMMEND BUTTON + RESULTS
# ─────────────────────────────────────────────

st.header("🎯 Your Personalised Recommendations")

if st.button("Get My Recommendations", type="primary"):

    # Run the recommendation engine
    cluster_id, cluster_info, rec_books, rec_movies, similarity_score = get_recommendations(
        energy=energy,
        valence=valence,
        tempo=tempo
    )

    # ── Listener Profile Card ──
    st.subheader("Your Listener Profile")

    profile_col, score_col = st.columns([2, 1])

    with profile_col:
        st.markdown(f"### {cluster_info['name']}")
        st.markdown(cluster_info['description'])
        st.markdown(f"**Matching book genres:** {', '.join(cluster_info['book_genres'])}")
        st.markdown(f"**Matching movie genres:** {', '.join(cluster_info['movie_genres'])}")

    with score_col:
        # Show similarity score as a big metric
        st.metric(
            label="Match Score",
            value=f"{similarity_score}%",
            help="How strongly your taste matches this listener profile"
        )

    st.markdown("---")

    # ── Books and Movies side by side ──
    books_col, movies_col = st.columns(2)

    with books_col:
        st.subheader("📚 Recommended Books")
        for i, row in rec_books.iterrows():
            with st.expander(f"{row['title']}  ⭐ {row['rating']}"):
                st.markdown(f"**Author:** {row['author']}")
                st.markdown(f"**Genre:** {row['genre']}")
                st.markdown(f"**Mood:** {row['mood_tags']}")

    with movies_col:
        st.subheader("🎬 Recommended Movies")
        for i, row in rec_movies.iterrows():
            with st.expander(f"{row['title']}  ⭐ {row['rating']}"):
                st.markdown(f"**Genre:** {row['genre']}")
                st.markdown(f"**Mood:** {row['mood_tags']}")
                st.markdown(f"**Year:** {row['year']}")

else:
    st.info("Set your preferences above and click **Get My Recommendations** to begin.")

st.markdown("---")


# ─────────────────────────────────────────────
# SECTION 4 — DATA ANALYTICS DASHBOARD
# ─────────────────────────────────────────────

st.header("📊 Data Analytics Dashboard")
st.markdown("Explore the data behind the recommendations.")

# Four charts in a 2x2 grid
chart_col1, chart_col2 = st.columns(2)

# ── Chart 1: Top music genres ──
with chart_col1:
    st.subheader("Top music genres")
    genre_counts = music_df['genre'].value_counts().reset_index()
    genre_counts.columns = ['genre', 'count']
    fig1 = px.bar(
        genre_counts,
        x='genre',
        y='count',
        color='genre',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig1.update_layout(
        showlegend=False,
        xaxis_title="Genre",
        yaxis_title="Number of tracks",
        height=350
    )
    st.plotly_chart(fig1, use_container_width=True)

# ── Chart 2: Mood distribution (valence) ──
with chart_col2:
    st.subheader("Mood distribution across tracks")
    fig2 = px.histogram(
        music_df,
        x='valence',
        nbins=30,
        color_discrete_sequence=['#636EFA']
    )
    fig2.update_layout(
        xaxis_title="Valence (0 = sad, 1 = happy)",
        yaxis_title="Number of tracks",
        height=350
    )
    st.plotly_chart(fig2, use_container_width=True)

chart_col3, chart_col4 = st.columns(2)

# ── Chart 3: Energy vs Valence scatter (coloured by genre) ──
with chart_col3:
    st.subheader("Energy vs mood by genre")
    fig3 = px.scatter(
        music_df,
        x='energy',
        y='valence',
        color='genre',
        hover_data=['track_name', 'artist', 'tempo'],
        color_discrete_sequence=px.colors.qualitative.Set2,
        opacity=0.7
    )
    fig3.update_layout(
        xaxis_title="Energy",
        yaxis_title="Valence (mood)",
        height=350
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Chart 4: Book genre distribution ──
with chart_col4:
    st.subheader("Book genre distribution")
    book_genre_counts = books_df['genre'].value_counts().reset_index()
    book_genre_counts.columns = ['genre', 'count']
    fig4 = px.pie(
        book_genre_counts,
        names='genre',
        values='count',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig4.update_layout(height=350)
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")


# ─────────────────────────────────────────────
# SECTION 5 — CLUSTER VISUALISATION
# ─────────────────────────────────────────────

st.header("🔬 Listener Cluster Visualisation")
st.markdown(
    "This chart shows the 4 listener personality clusters "
    "found in the music data. Each dot is one track."
)

# Add cluster labels to music_df for visualisation
music_df_viz = music_df.copy()
scaled_features = scaler.transform(music_df_viz[['energy', 'valence', 'tempo']])
music_df_viz['cluster'] = kmeans.predict(scaled_features)

cluster_names = {
    0: "Chill",
    1: "Energetic",
    2: "Emotional",
    3: "Dark/Alt"
}
music_df_viz['cluster_name'] = music_df_viz['cluster'].map(cluster_names)

fig5 = px.scatter(
    music_df_viz,
    x='energy',
    y='valence',
    color='cluster_name',
    size='tempo',
    hover_data=['genre', 'artist', 'tempo'],
    color_discrete_sequence=['#00CC96', '#EF553B', '#636EFA', '#AB63FA'],
    title="Music tracks grouped into 4 listener personality clusters"
)
fig5.update_layout(
    xaxis_title="Energy",
    yaxis_title="Valence (mood)",
    legend_title="Listener type",
    height=500
)
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")


# ─────────────────────────────────────────────
# SECTION 6 — HOW IT WORKS (MODEL EXPLANATION)
# ─────────────────────────────────────────────

st.header("🧠 How This Works")

exp_col1, exp_col2, exp_col3 = st.columns(3)

with exp_col1:
    st.markdown("### 1. Clustering")
    st.markdown(
        "We use **KMeans clustering** — a machine learning technique "
        "that groups 500 music tracks into 4 personality types "
        "based on their energy, mood (valence), and tempo."
    )

with exp_col2:
    st.markdown("### 2. Matching")
    st.markdown(
        "When you enter your preferences, we use **cosine similarity** "
        "to measure how close your music taste is to each cluster centre "
        "and find your strongest match."
    )

with exp_col3:
    st.markdown("### 3. Recommendations")
    st.markdown(
        "Each cluster maps to specific book and movie genres. "
        "We then find the **top rated** titles in those genres "
        "and return them as your personalised recommendations."
    )

st.markdown("---")
st.markdown(
    "Built with Python · Streamlit · scikit-learn · Plotly &nbsp;|&nbsp; "
    "Portfolio project by [Your Name]"
)