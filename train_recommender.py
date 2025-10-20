# train_recommender.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------- CONFIG ----------
CSV_PATH = "SpotifyFeatures.csv"
MODEL_DIR = "models"
N_NEIGHBORS = 50   # neighbors stored; app will return top 10 filtered suggestions
TOP_GENRES = 30    # one-hot top N genres, others grouped as 'Other'
RANDOM_STATE = 42
# --------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

# Ensure required columns exist
required = ['track_name','artist_name','track_id','genre','popularity',
            'danceability','energy','loudness','speechiness','acousticness',
            'instrumentalness','liveness','valence','tempo','duration_ms']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Dataset missing required columns: {missing}")

# Basic cleaning: drop rows with NA in important audio features
audio_cols = ['danceability','energy','loudness','speechiness','acousticness',
              'instrumentalness','liveness','valence','tempo','duration_ms']
df = df.dropna(subset=audio_cols).reset_index(drop=True)

# Create a display column
df['display'] = df['track_name'].astype(str) + " — " + df['artist_name'].astype(str)

# Create mood tags from valence & energy (rule-based mapping)
def mood_from_valence_energy(v, e):
    # v: valence (0-1), e: energy (0-1)
    if v >= 0.6 and e >= 0.6:
        return 'party'
    if v >= 0.6 and e < 0.6:
        return 'happy'
    if v < 0.4 and e < 0.5:
        return 'sad'
    if v < 0.6 and e < 0.6 and v >= 0.4:
        return 'chill'
    if e >= 0.7 and v >= 0.4:
        return 'workout'
    return 'chill'

df['mood'] = df.apply(lambda r: mood_from_valence_energy(r['valence'], r['energy']), axis=1)

# Limit genres to top-K, group rest into 'Other'
top_genres = df['genre'].value_counts().nlargest(TOP_GENRES).index.tolist()
df['genre_top'] = df['genre'].where(df['genre'].isin(top_genres), other='Other')

# Features to use for similarity
numeric_features = ['danceability','energy','loudness','speechiness','acousticness',
                    'instrumentalness','liveness','valence','tempo','duration_ms']

print("Preparing preprocessing pipeline...")
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

preprocessor = ColumnTransformer(transformers=[
    ('num', scaler, numeric_features),
    ('genre', ohe, ['genre_top'])
], remainder='drop', sparse_threshold=0)

print("Fitting preprocessor to dataset...")
X = preprocessor.fit_transform(df)

print("Feature matrix shape:", X.shape)

print("Fitting NearestNeighbors (cosine)...")
nn = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric='cosine', n_jobs=-1)
nn.fit(X)

# Save artifacts
print("Saving artifacts to models/ ...")
joblib.dump(preprocessor, os.path.join(MODEL_DIR, 'preprocessor.joblib'))
joblib.dump(nn, os.path.join(MODEL_DIR, 'nn.joblib'))

# Save metadata dataframe for lookups and filtering
# We store only necessary columns to reduce size
meta = df[['track_id','track_name','artist_name','display','genre','genre_top','popularity','mood'] + numeric_features].copy()
joblib.dump(meta, os.path.join(MODEL_DIR, 'metadata.joblib'))

print("Done. Artifacts saved:")
print(" - models/preprocessor.joblib")
print(" - models/nn.joblib")
print(" - models/metadata.joblib")
print("You can now run the Streamlit app (app.py).")
