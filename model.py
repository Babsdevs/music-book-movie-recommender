"""
model.py
--------
Trains the KMeans clustering model on music features.
Maps each cluster to book and movie genres.
Recommends top 5 books and movies for a given user input.

Run this file on its own to train and test the model:
    python model.py
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import run_preprocessing


# ─────────────────────────────────────────────
# STEP 1 — DEFINE THE 4 LISTENER CLUSTERS
# ─────────────────────────────────────────────

# This dictionary maps each cluster number (0,1,2,3) to:
#   - a friendly name
#   - which book genres match that listener personality
#   - which movie genres match that listener personality

CLUSTER_MAP = {
    0: {
        "name":          "Chill Listener",
        "description":   "You enjoy calm, relaxing music. Low energy, positive mood.",
        "book_genres":   ["self-help", "poetry"],
        "movie_genres":  ["drama", "documentary", "animation"],
    },
    1: {
        "name":          "Energetic Listener",
        "description":   "You love high-energy, upbeat music. Fast tempo, positive mood.",
        "book_genres":   ["adventure", "thriller"],
        "movie_genres":  ["action", "sport", "comedy"],
    },
    2: {
        "name":          "Emotional Listener",
        "description":   "You connect deeply with emotional music. Mid energy, reflective mood.",
        "book_genres":   ["romance", "literary fiction"],
        "movie_genres":  ["romance", "drama"],
    },
    3: {
        "name":          "Dark/Alternative Listener",
        "description":   "You are drawn to intense, dark, or alternative music. High energy, low valence.",
        "book_genres":   ["horror", "sci-fi"],
        "movie_genres":  ["horror", "thriller", "sci-fi"],
    },
}


# ─────────────────────────────────────────────
# STEP 2 — TRAIN THE KMEANS MODEL
# ─────────────────────────────────────────────

def train_model(feature_matrix):
    """
    Trains a KMeans model on the music feature matrix.
    KMeans groups the 500 music tracks into 4 clusters
    based on their energy, valence, and tempo patterns.

    Returns:
        kmeans : the trained KMeans model
    """
    print("Training KMeans model...")

    kmeans = KMeans(
        n_clusters=4,      # 4 listener personality groups
        random_state=42,   # makes results the same every time
        n_init=10          # tries 10 different starting points, picks best
    )

    # fit() is the actual training step — finds the 4 cluster centres
    kmeans.fit(feature_matrix)

    print(f"  Model trained on {feature_matrix.shape[0]} tracks")
    print(f"  Number of clusters: 4")
    print(f"  Cluster sizes: {dict(zip(*np.unique(kmeans.labels_, return_counts=True)))}")

    return kmeans


# ─────────────────────────────────────────────
# STEP 3 — SAVE THE TRAINED MODEL
# ─────────────────────────────────────────────

def save_model(kmeans, scaler):
    """
    Saves the trained model and scaler to disk as .pkl files.
    This means we don't have to retrain every time the app loads —
    we just load the saved model instantly.
    """
    with open("model.pkl", "wb") as f:
        pickle.dump(kmeans, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\n✓ Model saved as model.pkl")
    print("✓ Scaler saved as scaler.pkl")


# ─────────────────────────────────────────────
# STEP 4 — LOAD THE SAVED MODEL
# ─────────────────────────────────────────────

def load_model():
    """
    Loads the saved model and scaler from disk.
    Called by app.py so it doesn't retrain every time.

    Returns:
        kmeans : the trained KMeans model
        scaler : the MinMaxScaler used during preprocessing
    """
    with open("model.pkl", "rb") as f:
        kmeans = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return kmeans, scaler


# ─────────────────────────────────────────────
# STEP 5 — PREDICT WHICH CLUSTER A USER BELONGS TO
# ─────────────────────────────────────────────

def predict_cluster(energy, valence, tempo, kmeans, scaler):
    """
    Takes a user's music preferences and finds which
    of the 4 listener clusters they belong to.

    Inputs:
        energy  : float 0-1  (how energetic their music is)
        valence : float 0-1  (how positive/happy their music is)
        tempo   : float      (BPM of their music, e.g. 120)
        kmeans  : trained KMeans model
        scaler  : the same scaler used in training

    Returns:
        cluster_id   : int (0, 1, 2, or 3)
        cluster_info : dict with name, description, genres
    """
    # Put user values into a numpy array — same format as training data
    user_input = np.array([[energy, valence, tempo]])

    # Scale the user input the same way we scaled training data
    user_input_scaled = scaler.transform(user_input)

    # predict() finds the nearest cluster centre
    cluster_id = kmeans.predict(user_input_scaled)[0]

    cluster_info = CLUSTER_MAP[cluster_id]

    return cluster_id, cluster_info


# ─────────────────────────────────────────────
# STEP 6 — RECOMMEND BOOKS
# ─────────────────────────────────────────────

def recommend_books(cluster_info, books_df, top_n=5):
    """
    Filters the books dataset to only books that match
    the cluster's genres, then returns the top rated ones.

    Returns a DataFrame of the top N recommended books.
    """
    # Get the list of matching genres for this cluster
    target_genres = cluster_info["book_genres"]

    # Filter books to only matching genres
    matched = books_df[books_df["genre"].isin(target_genres)].copy()

    # Sort by rating (highest first) and return top N
    matched = matched.sort_values("rating", ascending=False)

    return matched.head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────
# STEP 7 — RECOMMEND MOVIES
# ─────────────────────────────────────────────

def recommend_movies(cluster_info, movies_df, top_n=5):
    """
    Filters the movies dataset to only movies that match
    the cluster's genres, then returns the top rated ones.

    Returns a DataFrame of the top N recommended movies.
    """
    target_genres = cluster_info["movie_genres"]

    matched = movies_df[movies_df["genre"].isin(target_genres)].copy()

    matched = matched.sort_values("rating", ascending=False)

    return matched.head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────
# STEP 8 — CALCULATE SIMILARITY SCORE
# ─────────────────────────────────────────────

def get_similarity_score(energy, valence, tempo, kmeans, scaler):
    """
    Calculates how similar the user's music taste is
    to the centre of their assigned cluster.
    Returns a score between 0% and 100%.
    Higher = stronger match.
    """
    user_input = np.array([[energy, valence, tempo]])
    user_scaled = scaler.transform(user_input)

    # Get the cluster centre coordinates
    cluster_id = kmeans.predict(user_scaled)[0]
    cluster_centre = kmeans.cluster_centers_[cluster_id].reshape(1, -1)

    # Cosine similarity gives a value between 0 and 1
    score = cosine_similarity(user_scaled, cluster_centre)[0][0]

    # Convert to a percentage and round to 1 decimal place
    return round(score * 100, 1)


# ─────────────────────────────────────────────
# STEP 9 — ONE FUNCTION THAT RUNS EVERYTHING
# ─────────────────────────────────────────────

def get_recommendations(energy, valence, tempo, top_n=5):
    """
    Master function — given a user's energy, valence, and tempo,
    returns their cluster profile + recommended books and movies.

    This is the function app.py will call.

    Returns:
        cluster_id      : int
        cluster_info    : dict (name, description, genres)
        recommended_books  : DataFrame
        recommended_movies : DataFrame
        similarity_score   : float (percentage)
    """
    # Load the saved model
    kmeans, scaler = load_model()

    # Load cleaned data
    _, books_df, movies_df, _, _ = run_preprocessing()

    # Find the user's cluster
    cluster_id, cluster_info = predict_cluster(
        energy, valence, tempo, kmeans, scaler
    )

    # Get recommendations
    recommended_books  = recommend_books(cluster_info, books_df, top_n)
    recommended_movies = recommend_movies(cluster_info, movies_df, top_n)

    # Get similarity score
    similarity_score = get_similarity_score(
        energy, valence, tempo, kmeans, scaler
    )

    return cluster_id, cluster_info, recommended_books, recommended_movies, similarity_score


# ─────────────────────────────────────────────
# TEST — run this file directly to train + test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Load and preprocess data
    music_df, books_df, movies_df, feature_matrix, scaler = run_preprocessing()

    # 2. Train the model
    kmeans = train_model(feature_matrix)

    # 3. Save model and scaler to disk
    save_model(kmeans, scaler)

    # 4. Test with a sample user
    print("\n── Testing with a sample user ──")
    print("User input: energy=0.8, valence=0.9, tempo=130 (sounds energetic!)\n")

    cluster_id, cluster_info, rec_books, rec_movies, score = get_recommendations(
        energy=0.8,
        valence=0.9,
        tempo=130
    )

    print(f"Cluster       : {cluster_id} — {cluster_info['name']}")
    print(f"Description   : {cluster_info['description']}")
    print(f"Match score   : {score}%")

    print("\nTop recommended books:")
    print(rec_books[['title', 'genre', 'rating']].to_string(index=False))

    print("\nTop recommended movies:")
    print(rec_movies[['title', 'genre', 'rating']].to_string(index=False))