---
name: music-recommender
description: Helps build a cross-domain music-to-book-and-movie recommendation system using Python, KMeans clustering, and Streamlit. Use this skill when the user asks about any part of this project — data cleaning, ML model, Streamlit app, charts, deployment, or GitHub.
---

# Music-based recommender — project context

## What this project does
Takes a user's music preferences (genre, mood, energy) and recommends books and movies that match their listening personality using KMeans clustering and cosine similarity.

## Project folder structure
```
music-book-movie-recommender/
├── skills/
│   └── SKILL.md
├── data/
│   ├── music_data.csv
│   ├── books_data.csv
│   └── movies_data.csv
├── notebooks/
│   └── 01_EDA.ipynb
├── preprocessing.py
├── model.py
├── app.py
├── requirements.txt
└── README.md
```

## Data columns (use these exact names always)
- music_data.csv: track_name, artist, genre, energy, valence, tempo, danceability
- books_data.csv: title, author, genre, rating, mood_tags
- movies_data.csv: title, genre, rating, mood_tags, year

## ML model details
- Algorithm: KMeans clustering (scikit-learn)
- Features used for clustering: energy, valence, tempo (all normalised to 0-1 scale)
- Number of clusters: 4
- Cluster labels: chill, energetic, emotional, dark
- Saved model file: model.pkl
- Similarity method: cosine similarity

## Cluster to genre mapping
- chill → books: self-help, poetry | movies: drama, documentary
- energetic → books: adventure, thriller | movies: action, sport
- emotional → books: romance, literary fiction | movies: romance, drama
- dark → books: horror, sci-fi | movies: horror, thriller

## Python libraries to always use
- pandas — all data loading and cleaning
- scikit-learn — KMeans, MinMaxScaler, cosine_similarity
- plotly — all charts in app.py
- streamlit — the web dashboard
- numpy — numeric operations
- pickle — saving and loading model.pkl

## Coding rules for this project
- Always write beginner-friendly code with clear comments on every step
- Use exact column names listed above — never rename them
- Keep functions small and simple — one function does one thing
- Always include error handling so the app does not crash

## Deployment
- App deployed on Streamlit Cloud
- Code stored on GitHub repository named: music-book-movie-recommender