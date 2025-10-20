# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_distances
from difflib import get_close_matches

st.set_page_config(page_title="Music Recommender", layout="wide")
st.title("🎧 Music Recommendation System")
st.write("Search a song (type part of name) or choose a mood — get similar songs instantly.")

# Load artifacts
MODEL_DIR = "models"
try:
    preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor.joblib")
    nn = joblib.load(f"{MODEL_DIR}/nn.joblib")
    meta = joblib.load(f"{MODEL_DIR}/metadata.joblib")
except Exception as e:
    st.error(f"Error loading models/metadata. Run train_recommender.py first. ({e})")
    st.stop()

# Build search list
meta = meta.reset_index(drop=True)
display_list = meta['display'].tolist()

# Sidebar controls
st.sidebar.header("Controls")
top_k = st.sidebar.slider("Number of recommendations to show", min_value=5, max_value=20, value=10)
mood_filter = st.sidebar.selectbox("Filter by mood (optional)", options=["All", "happy", "party", "chill", "workout", "sad"])
similarity_method = st.sidebar.selectbox("Similarity method", options=["NearestNeighbors (fast)", "Cosine distance (fine)"])

# Main UI: search
query = st.text_input("Type song name or artist (partial). Suggestions will appear below:")
suggestions = []
if query:
    q = query.lower().strip()
    substr_matches = [s for s in display_list if q in s.lower()]
    if len(substr_matches) < 15:
        close = get_close_matches(query, display_list, n=15)
        for c in close:
            if c not in substr_matches:
                substr_matches.append(c)
    suggestions = substr_matches[:50]

selected = None
if suggestions:
    selected = st.selectbox("Suggestions (pick a song)", [""] + suggestions, index=0)
else:
    selected = st.selectbox("No suggestions yet — you can still type and press Search", [""])

# Also provide mood-only quick pick
st.markdown("### Or pick a mood to get popular songs of that mood")
mood_choice = st.selectbox("Pick mood (or None)", ["None", "happy", "party", "chill", "workout", "sad"])
mood_only_btn = st.button("Get mood recommendations")

# Helper: get features vector for a row index
def row_to_feature_vector(row):
    # meta contains numeric features in fixed order used during training
    numeric_cols = ['danceability','energy','loudness','speechiness','acousticness',
                    'instrumentalness','liveness','valence','tempo','duration_ms']
    arr = row[numeric_cols].astype(float).values.reshape(1, -1)
    # preprocessor expects the 'genre_top' too, so we build a temp df with same columns
    # But easiest is to create a DataFrame with numeric + genre_top column for ColumnTransformer
    tmp = pd.DataFrame(arr, columns=numeric_cols)
    tmp['genre_top'] = row['genre_top']
    # preprocessor was fitted with ColumnTransformer expecting numeric columns first then 'genre_top'
    X = preprocessor.transform(tmp)
    return X

# Action: search & recommend
if st.button("Search & Recommend") or (selected and selected != ""):
    chosen_display = selected if selected and selected != "" else ""
    if chosen_display == "":
        st.warning("Please select a suggestion from the list or type a song and pick one.")
    else:
        idx = meta[meta['display'] == chosen_display].index
        if len(idx) == 0:
            st.error("Selected song not found in metadata. Try a different suggestion.")
        else:
            idx = idx[0]
            row = meta.iloc[idx]
            st.subheader(f"Selected: {row['display']}")
            st.write(f"Genre: {row['genre']}, Mood: {row['mood']}, Popularity: {row['popularity']}")
            # compute vector
            X_vec = row_to_feature_vector(row)  # shape (1, D)
            # use nearest neighbors or cosine distances
            if similarity_method == "NearestNeighbors (fast)":
                distances, indices = nn.kneighbors(X_vec, n_neighbors= min(nn.n_neighbors, len(meta)))
                distances = distances.flatten()
                indices = indices.flatten()
                # distances are cosine distances (0 good)
                results = pd.DataFrame({'idx': indices, 'dist': distances})
            else:
                # compute cosine distances to full matrix
                # load matrix from preprocessor on full meta
                numeric_cols = ['danceability','energy','loudness','speechiness','acousticness',
                                'instrumentalness','liveness','valence','tempo','duration_ms']
                tmp_all = meta[numeric_cols].copy()
                tmp_all['genre_top'] = meta['genre_top']
                X_all = preprocessor.transform(tmp_all)
                dists = cosine_distances(X_vec, X_all).flatten()
                results = pd.DataFrame({'idx': np.arange(len(meta)), 'dist': dists})
            # Drop the query itself
            results = results[results['idx'] != idx].copy()
            # If mood_filter selected in sidebar, apply it
            if mood_filter != "All":
                mood_idxs = meta[meta['mood'] == mood_filter].index
                results = results[results['idx'].isin(mood_idxs)]
            # Also if user selected mood_choice (mood-only), apply that too (AND)
            if mood_choice != "None":
                results = results[results['idx'].isin(meta[meta['mood'] == mood_choice].index)]
            # Sort by distance ascending (smaller = more similar)
            results = results.sort_values('dist', ascending=True).head(top_k*3)  # keep some room for filtering
            # Map to metadata and compute similarity score
            recs = meta.iloc[results['idx'].astype(int)].copy()
            recs['score'] = 1 - results['dist'].values  # approx similarity
            # Now pick top_k by score, also ensure unique tracks
            recs = recs.sort_values('score', ascending=False).head(top_k)
            # Display recommendations
            st.markdown(f"### Top {len(recs)} Recommendations")
            for i, r in recs.reset_index(drop=True).iterrows():
                st.markdown(f"**{i+1}. {r['track_name']} — {r['artist_name']}**")
                st.write(f"Genre: {r['genre']}, Mood: {r['mood']}, Popularity: {r['popularity']}, Similarity: {r['score']:.3f}")
                st.markdown("---")

# Mood-only recommendations
if mood_only_btn or (mood_choice != "None" and mood_only_btn):
    chosen_mood = mood_choice
    if chosen_mood == "None":
        st.warning("Pick a mood first.")
    else:
        st.subheader(f"Popular songs for mood: {chosen_mood}")
        subset = meta[meta['mood'] == chosen_mood].sort_values('popularity', ascending=False).head(top_k)
        for i, r in subset.reset_index(drop=True).iterrows():
            st.markdown(f"**{i+1}. {r['track_name']} — {r['artist_name']}**")
            st.write(f"Genre: {r['genre']}, Popularity: {r['popularity']}")
            st.markdown("---")

st.markdown("----")
st.markdown("**Notes:** The recommender uses audio features + genre. It uses cosine distance to find songs with similar audio characteristics. Use the sidebar to tune the number of recommendations and optional mood filters.")
