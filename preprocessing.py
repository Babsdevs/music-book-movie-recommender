"""
preprocessing.py
----------------
Loads the 3 CSV files, cleans the data, and prepares
the feature matrix that the ML model will use for clustering.

Run this file on its own to test it:
    python preprocessing.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────
# STEP 1 — LOAD THE DATA
# ─────────────────────────────────────────────

def load_data():
    """
    Loads all 3 CSV files from the data/ folder.
    Returns them as pandas DataFrames.
    """
    print("Loading data files...")

    music_df  = pd.read_csv("data/music_data.csv")
    books_df  = pd.read_csv("data/books_data.csv")
    movies_df = pd.read_csv("data/movies_data.csv")

    print(f"  Music rows  : {len(music_df)}")
    print(f"  Books rows  : {len(books_df)}")
    print(f"  Movies rows : {len(movies_df)}")

    return music_df, books_df, movies_df


# ─────────────────────────────────────────────
# STEP 2 — CLEAN THE MUSIC DATA
# ─────────────────────────────────────────────

def clean_music(music_df):
    """
    Cleans the music DataFrame.
    - Removes any rows with missing values
    - Makes sure numeric columns are actually numbers
    - Removes duplicates
    """
    print("\nCleaning music data...")

    # Remove rows where any value is missing
    music_df = music_df.dropna()

    # Remove duplicate rows
    music_df = music_df.drop_duplicates()

    # Make sure these columns are numeric (sometimes CSV loads them as text)
    numeric_cols = ['energy', 'valence', 'tempo', 'danceability']
    for col in numeric_cols:
        music_df[col] = pd.to_numeric(music_df[col], errors='coerce')

    # Drop any rows that became NaN after the conversion above
    music_df = music_df.dropna()

    # Reset the index so rows are numbered 0, 1, 2, 3...
    music_df = music_df.reset_index(drop=True)

    print(f"  Clean music rows: {len(music_df)}")
    return music_df


# ─────────────────────────────────────────────
# STEP 3 — NORMALISE THE MUSIC FEATURES
# ─────────────────────────────────────────────

def normalise_features(music_df):
    """
    Takes energy, valence, and tempo and scales them all to 0-1.
    This is important because tempo (e.g. 120 BPM) is a much bigger
    number than energy (e.g. 0.8) — without normalising, tempo would
    unfairly dominate the clustering.

    Returns:
        feature_matrix : the scaled numbers the model trains on
        scaler         : saved so we can scale new user input later
    """
    print("\nNormalising features...")

    # These 3 columns are what the KMeans model clusters on
    features = music_df[['energy', 'valence', 'tempo']].copy()

    # MinMaxScaler pushes every value to be between 0 and 1
    scaler = MinMaxScaler()
    feature_matrix = scaler.fit_transform(features)

    print(f"  Feature matrix shape: {feature_matrix.shape}")
    print(f"  Columns used: energy, valence, tempo")

    return feature_matrix, scaler


# ─────────────────────────────────────────────
# STEP 4 — PREPARE BOOKS AND MOVIES
# ─────────────────────────────────────────────

def clean_books(books_df):
    """
    Cleans the books DataFrame.
    Removes missing values and duplicates.
    """
    books_df = books_df.dropna()
    books_df = books_df.drop_duplicates(subset=['title'])
    books_df = books_df.reset_index(drop=True)
    print(f"\nClean books rows: {len(books_df)}")
    return books_df


def clean_movies(movies_df):
    """
    Cleans the movies DataFrame.
    Removes missing values and duplicates.
    """
    movies_df = movies_df.dropna()
    movies_df = movies_df.drop_duplicates(subset=['title'])
    movies_df = movies_df.reset_index(drop=True)
    print(f"Clean movies rows: {len(movies_df)}")
    return movies_df


# ─────────────────────────────────────────────
# STEP 5 — ONE FUNCTION THAT RUNS EVERYTHING
# ─────────────────────────────────────────────

def run_preprocessing():
    """
    Master function — runs all the steps above in order.
    Call this from model.py and app.py.

    Returns:
        music_df       : clean music DataFrame
        books_df       : clean books DataFrame
        movies_df      : clean movies DataFrame
        feature_matrix : normalised numbers ready for KMeans
        scaler         : the scaler object (saved for later use)
    """
    # Load
    music_df, books_df, movies_df = load_data()

    # Clean
    music_df  = clean_music(music_df)
    books_df  = clean_books(books_df)
    movies_df = clean_movies(movies_df)

    # Normalise music features for ML
    feature_matrix, scaler = normalise_features(music_df)

    print("\n✓ Preprocessing complete — data is ready for the model.")
    return music_df, books_df, movies_df, feature_matrix, scaler


# ─────────────────────────────────────────────
# TEST — run this file directly to check it works
# ─────────────────────────────────────────────

if __name__ == "__main__":
    music_df, books_df, movies_df, feature_matrix, scaler = run_preprocessing()

    print("\n── Sample music data ──")
    print(music_df[['track_name', 'genre', 'energy', 'valence', 'tempo']].head(5))

    print("\n── Sample books data ──")
    print(books_df[['title', 'genre', 'mood_tags']].head(5))

    print("\n── Sample movies data ──")
    print(movies_df[['title', 'genre', 'mood_tags']].head(5))

    print("\n── Feature matrix (first 3 rows, normalised) ──")
    print(feature_matrix[:3])