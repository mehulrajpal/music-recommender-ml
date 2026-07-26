"""
app.py -- Streamlit app for the Music Taste Predictor & Recommender.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this repo to GitHub (including the *.pkl artifacts, schema.pkl,
       metrics.json and music_dataset.csv -- they're small enough to commit).
    2. Go to https://share.streamlit.io -> New app -> point it at this repo,
       branch, and app.py.
    3. Done -- no server config needed, requirements.txt is auto-installed.

If the *.pkl artifacts are missing (e.g. fresh clone without them), retrain first:
    python -m src.train --data music_dataset.csv --model xgboost
"""
import glob
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_config

st.set_page_config(page_title="Music Taste Predictor & Recommender", page_icon="🎧", layout="wide")

CONFIG = load_config("config.yaml")
DATA_PATH = "music_dataset.csv"


@st.cache_resource
def load_artifacts():
    art = {}
    try:
        art["schema"] = joblib.load("schema.pkl")
    except FileNotFoundError:
        return art

    g = glob.glob("*_genre_predictor.pkl")
    a = glob.glob("*_artist_predictor.pkl")
    if g:
        art["genre"] = joblib.load(g[0])
        art["genre_model"] = g[0].replace("_genre_predictor.pkl", "")
    if a:
        art["artist"] = joblib.load(a[0])
        art["artist_model"] = a[0].replace("_artist_predictor.pkl", "")
    try:
        art["recommender"] = joblib.load("knn_recommender.pkl")
    except FileNotFoundError:
        pass
    try:
        with open("metrics.json") as f:
            art["metrics"] = json.load(f)
    except FileNotFoundError:
        art["metrics"] = {}
    return art


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


artifacts = load_artifacts()
df = load_data()

st.title("🎧 Music Taste Predictor & Recommender")
st.caption(
    "Genre & artist prediction (KNN · Random Forest · Gradient Boosting · XGBoost) "
    "plus content-based song recommendations — trained on `music_dataset.csv`."
)

if "schema" not in artifacts:
    st.warning("No trained models found. Run this first, then reload the app:")
    st.code("python -m src.train --data music_dataset.csv --model xgboost")
    st.stop()

schema = artifacts["schema"]
feature_num, feature_cat = schema["numeric_features"], schema["categorical_features"]
metrics = artifacts.get("metrics", {})

# ---------------------------------------------------------------- header KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tracks in dataset", len(df))
c2.metric("Genres", df[schema["genre_col"]].nunique() if schema["genre_col"] else "—")
c3.metric(
    "Genre model accuracy",
    f"{metrics.get('genre', {}).get('accuracy', 0):.1%}" if metrics.get("genre") else "—",
    help=f"Model: {artifacts.get('genre_model', '—')}, top-3 accuracy: "
         f"{metrics.get('genre', {}).get('top_3_accuracy', 0):.1%}",
)
c4.metric(
    "Artist model accuracy",
    f"{metrics.get('artist', {}).get('accuracy', 0):.1%}" if metrics.get("artist") else "—",
    help="See the 'About' tab — audio features alone barely constrain artist identity in this dataset.",
)

tab_predict, tab_recommend, tab_explore, tab_about = st.tabs(
    ["🎯 Predict genre / artist", "🔁 Recommend songs", "📊 Explore dataset", "ℹ️ About"]
)

# ---------------------------------------------------------------- Predict tab
with tab_predict:
    st.write("Set audio-feature values for a hypothetical track and predict its genre / artist:")
    inputs = {}
    cols = st.columns(3)
    for i, col in enumerate(feature_num):
        lo, hi, med = float(df[col].min()), float(df[col].max()), float(df[col].median())
        if hi - lo <= 1.5 and df[col].round(4).eq(df[col]).all() and hi <= 1.0:
            inputs[col] = cols[i % 3].slider(col, min_value=0.0, max_value=1.0, value=round(med, 3))
        else:
            inputs[col] = cols[i % 3].number_input(col, value=med)
    for i, col in enumerate(feature_cat):
        opts = sorted(df[col].dropna().unique().tolist())
        inputs[col] = cols[(i + len(feature_num)) % 3].selectbox(col, opts)

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([inputs])
        r1, r2 = st.columns(2)
        if "genre" in artifacts:
            p, enc = artifacts["genre"]["pipeline"], artifacts["genre"]["encoder"]
            pred = enc.inverse_transform(p.predict(row))[0]
            r1.success(f"Predicted genre: **{pred}**")
            if hasattr(p, "predict_proba"):
                proba = p.predict_proba(row)[0]
                top = sorted(zip(enc.classes_, proba), key=lambda x: -x[1])[:5]
                r1.dataframe(
                    pd.DataFrame(top, columns=["genre", "probability"]).assign(
                        probability=lambda d: d["probability"].map("{:.1%}".format)
                    ),
                    hide_index=True, use_container_width=True,
                )
        if "artist" in artifacts:
            p, enc = artifacts["artist"]["pipeline"], artifacts["artist"]["encoder"]
            pred = enc.inverse_transform(p.predict(row))[0]
            r2.info(f"Predicted artist (low confidence, see 'About'): **{pred}**")
            if hasattr(p, "predict_proba"):
                proba = p.predict_proba(row)[0]
                top = sorted(zip(enc.classes_, proba), key=lambda x: -x[1])[:5]
                r2.dataframe(
                    pd.DataFrame(top, columns=["artist", "probability"]).assign(
                        probability=lambda d: d["probability"].map("{:.1%}".format)
                    ),
                    hide_index=True, use_container_width=True,
                )

# ------------------------------------------------------------- Recommend tab
with tab_recommend:
    st.write("Get song recommendations similar to a listener profile:")
    top_n = st.slider("Number of recommendations", 3, 25, 10)
    prefs = {}
    cols = st.columns(3)
    for i, col in enumerate(feature_num):
        lo, hi, med = float(df[col].min()), float(df[col].max()), float(df[col].median())
        if hi - lo <= 1.5 and hi <= 1.0:
            prefs[col] = cols[i % 3].slider(col, 0.0, 1.0, round(med, 3), key=f"rec_{col}")
        else:
            prefs[col] = cols[i % 3].number_input(col, value=med, key=f"rec_{col}")

    if st.button("Recommend songs", type="primary"):
        if "recommender" in artifacts:
            recs = artifacts["recommender"].recommend_from_preferences(prefs, top_n=top_n)
            st.dataframe(recs, use_container_width=True, hide_index=True)
        else:
            st.error("No recommender found. Train first.")

# --------------------------------------------------------------- Explore tab
with tab_explore:
    st.write("Quick look at the bundled dataset:")
    left, right = st.columns(2)
    if schema["genre_col"]:
        genre_counts = df[schema["genre_col"]].value_counts().reset_index()
        genre_counts.columns = ["genre", "count"]
        left.plotly_chart(
            px.bar(genre_counts, x="genre", y="count", title="Tracks per genre"),
            use_container_width=True,
        )
    if "popularity" in df.columns and schema["genre_col"]:
        right.plotly_chart(
            px.box(df, x=schema["genre_col"], y="popularity", title="Popularity by genre"),
            use_container_width=True,
        )
    if "energy" in df.columns and "danceability" in df.columns:
        st.plotly_chart(
            px.scatter(
                df, x="energy", y="danceability",
                color=schema["genre_col"] if schema["genre_col"] else None,
                hover_data=[schema["track_col"]] if schema["track_col"] else None,
                title="Energy vs. danceability by genre",
            ),
            use_container_width=True,
        )
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------- About tab
with tab_about:
    st.markdown(
        """
### How this works
- **`src/schema.py`** auto-detects the genre / artist / track columns and splits the
  rest into numeric vs. categorical features — this app works on any similarly-shaped
  music CSV, not just this one.
- **Genre model** (`{genre_model}`): trained on {n_genre_classes} genres, features are
  the audio characteristics (energy, danceability, tempo, ...). It generalizes well
  because genre is strongly correlated with those audio features.
- **Artist model** (`{artist_model}`): trained on {n_artist_classes} artists using the
  *same* audio features. Its accuracy (~{artist_acc:.0%}, barely above the
  {chance:.0%} random-guess baseline for {n_artist_classes} classes) reflects a real
  property of this dataset, not a bug: an artist's catalogue spans many genres and
  tempos here, so audio features alone don't identify *who* made a track — only
  *what it sounds like*. To improve this meaningfully you'd need artist-level
  features (vocal timbre embeddings, production style, lyrics, or a
  collaborative-filtering signal), not just per-track audio stats.
- **Recommender**: K-Nearest Neighbors similarity search (cosine distance) over the
  same feature space — this is a retrieval task, not classification, so it isn't
  affected by the artist-prediction ceiling above.

### Retraining on your own data
```bash
python -m src.train --data your_dataset.csv --model xgboost
```
Then reload this app — it auto-loads whatever `*_genre_predictor.pkl` /
`*_artist_predictor.pkl` / `knn_recommender.pkl` files exist at the repo root.
        """.format(
            genre_model=artifacts.get("genre_model", "—"),
            artist_model=artifacts.get("artist_model", "—"),
            n_genre_classes=metrics.get("genre", {}).get("n_classes", "—"),
            n_artist_classes=metrics.get("artist", {}).get("n_classes", "—"),
            artist_acc=metrics.get("artist", {}).get("accuracy", 0),
            chance=(1 / metrics.get("artist", {}).get("n_classes", 1))
            if metrics.get("artist", {}).get("n_classes") else 0,
        )
    )
