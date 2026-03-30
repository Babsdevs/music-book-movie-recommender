# 🎵 Music-Based Personalised Recommendation Engine

**Live App:** [Click here to try it](https://YOUR-USERNAME-music-book-movie-recommender-app.streamlit.app)

---

## Project Overview

This project builds a cross-domain personalised recommendation system that suggests **books and movies** based on a user's music listening behaviour. Using clustering and similarity-based machine learning, the system identifies listener personality profiles and maps them to relevant content categories.

---

## Business Problem

Users consume content across multiple platforms — music on Spotify, books on Goodreads, movies on Netflix — but recommendations are always siloed. This project bridges that gap by using music taste as a signal to recommend books and movies, creating a truly cross-domain experience.

---

## Live Demo

1. Select your favourite music genre and mood
2. Adjust the energy, valence, and tempo sliders
3. Click **Get My Recommendations**
4. Receive personalised book and movie suggestions with match scores

---

## Dataset Description

| Dataset | Rows | Key Columns |
|---|---|---|
| Music | 500 | genre, energy, valence, tempo, danceability |
| Books | 80 | title, genre, rating, mood_tags |
| Movies | 100 | title, genre, rating, mood_tags, year |

Music features (energy, valence, tempo) are normalised to 0–1 scale using MinMaxScaler before being passed to the model.

---

## Machine Learning Model

- **Algorithm:** KMeans Clustering (scikit-learn)
- **Features used:** energy, valence, tempo
- **Number of clusters:** 4
- **Similarity scoring:** Cosine similarity

### The 4 Listener Profiles

| Cluster | Name | Book Genres | Movie Genres |
|---|---|---|---|
| 0 | Chill Listener | self-help, poetry | drama, documentary |
| 1 | Energetic Listener | adventure, thriller | action, sport |
| 2 | Emotional Listener | romance, literary fiction | romance, drama |
| 3 | Dark/Alt Listener | horror, sci-fi | horror, thriller |

---

## Personalisation Logic

1. User inputs energy, valence, and tempo values
2. Input is normalised using the saved MinMaxScaler
3. KMeans predicts which of 4 clusters the user belongs to
4. Cosine similarity calculates a match score (0–100%)
5. Top rated books and movies from matching genres are returned

---

## Data Analytics

The dashboard includes:
- Bar chart of top music genres
- Mood (valence) distribution histogram
- Energy vs valence scatter plot coloured by genre
- Book genre pie chart
- Full cluster visualisation — 500 tracks plotted by personality type

---

## Project Structure

```
music-book-movie-recommender/
├── skills/
│   └── SKILL.md           ← Claude AI skill for this project
├── data/
│   ├── music_data.csv
│   ├── books_data.csv
│   └── movies_data.csv
├── preprocessing.py       ← data loading and cleaning
├── model.py               ← KMeans training and recommendations
├── app.py                 ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR-USERNAME/music-book-movie-recommender.git
cd music-book-movie-recommender

# 2. Install libraries
pip install -r requirements.txt

# 3. Train the model
python model.py

# 4. Run the app
streamlit run app.py
```

---

## Skills Demonstrated

- **Data Analytics** — EDA, feature engineering, visualisation
- **Machine Learning** — KMeans clustering, cosine similarity, model persistence
- **Personalisation** — content-based filtering, user profiling
- **Web Development** — interactive Streamlit dashboard
- **MLOps** — model saving, loading, deployment on Streamlit Cloud
- **Version Control** — full Git and GitHub workflow

---

## Future Improvements

- Connect to real Spotify API for live listening data
- Add collaborative filtering for user-based recommendations
- Include more content domains (podcasts, games)
- Add user accounts and recommendation history

---

## Built With

Python · Streamlit · scikit-learn · Pandas · Plotly · NumPy

---

*Built as a portfolio project to demonstrate cross-domain ML personalisation.*