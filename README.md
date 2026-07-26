# 🎧 Music Taste Predictor & Recommender

Predict the **genre**, **artist**, and **songs** a listener wants to hear next —
using **KNN**, **Bagging (Random Forest)**, **Boosting (Gradient Boosting)**,
**XGBoost**, and a soft-voting **Ensemble** of them all. Works out of the box on
the bundled dataset, and is built to **plug in any other music dataset** without
code changes. Ships as a deployable Streamlit web app.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%3E%3D1.2-orange)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-%3E%3D2.0-brightgreen)]()
[![Streamlit](https://img.shields.io/badge/app-Streamlit-ff4b4b)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()

---

## ✨ Features

- **Dataset-agnostic by design** — `src/schema.py` auto-detects genre/artist/track/user
  columns and numeric vs. categorical features by inspecting the CSV headers and dtypes.
  Point it at a Spotify export, a Last.fm dump, an MCRec-style Kaggle CSV, or your own
  listening logs — no code changes required, just re-run training.
- **Five model options, one interface**:
  - `knn` — K-Nearest Neighbors (instance-based)
  - `bagging` — Random Forest (bagging of decision trees)
  - `boosting` — Gradient Boosting (sklearn boosted decision trees)
  - `xgboost` — XGBoost (optimized gradient-boosted trees, usually fastest + strongest solo model)
  - `ensemble` — soft-voting combination of KNN + Bagging + XGBoost
- **Song recommendation** via KNN similarity search over audio/behavioural features
  (content-based, so "song" doesn't need to be treated as a giant classification target).
- **Deployable Streamlit app** (`app.py`) — reads whatever `.pkl` models exist at the
  repo root and serves an interactive UI with genre/artist prediction, song
  recommendations, and dataset exploration charts. Deploys to Streamlit Community
  Cloud with zero server config.
- **Training notebook** (`train_model.ipynb`) with EDA, side-by-side comparison charts
  for all five model types, and the same save step the CLI (`src/train.py`) performs.
- **Tested**: `pytest` suite that verifies the schema detector generalizes to a *second*,
  differently-named dataset — proving the "works with any dataset" claim rather than
  just asserting it.

---

## 📁 Project structure

```
music-recommender-ml/
├── app.py                        # Streamlit app (loads .pkl models, serves predict/recommend/explore UI)
├── train_model.ipynb             # EDA + interactive training walkthrough / model comparison
├── generate_sample_data.py       # builds the bundled synthetic demo dataset
├── music_dataset.csv             # bundled demo dataset (1200 rows)
├── config.yaml                   # column-detection candidates + model hyperparameters
├── requirements.txt
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── xgboost_genre_predictor.pkl    # trained artifact (regenerate anytime, see below)
├── xgboost_artist_predictor.pkl   # trained artifact
├── knn_recommender.pkl            # trained artifact
├── schema.pkl                     # detected dataset schema, used by app.py / predict.py
├── src/
│   ├── data_loader.py             # CSV loading + generic cleaning
│   ├── schema.py                  # dataset-agnostic column auto-detection
│   ├── preprocessing.py           # ColumnTransformer builder (impute/scale/one-hot)
│   ├── models.py                  # KNN / Bagging / Boosting / XGBoost / Ensemble pipelines
│   ├── recommender.py             # KNN-based song recommender
│   ├── evaluate.py                # shared metric reporting
│   ├── train.py                   # CLI: train models on any CSV
│   └── predict.py                 # CLI: run inference with saved models
└── tests/
    └── test_pipeline.py           # pytest suite, incl. a second synthetic dataset
```

---

## 🚀 Quickstart

### 1. Install

```bash
git clone https://github.com/<your-username>/music-recommender-ml.git
cd music-recommender-ml
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### 2. (Re)generate the bundled demo dataset — optional, one is already included

```bash
python generate_sample_data.py
```

### 3. Train

```bash
python -m src.train --data music_dataset.csv --model xgboost
```

This detects the schema, trains a genre classifier, an artist classifier, and a song
recommender, and saves everything to the repo root:

```
xgboost_genre_predictor.pkl
xgboost_artist_predictor.pkl
knn_recommender.pkl
schema.pkl
metrics.json
```

Train with a different algorithm instead:

```bash
python -m src.train --data music_dataset.csv --model knn
python -m src.train --data music_dataset.csv --model bagging
python -m src.train --data music_dataset.csv --model boosting
python -m src.train --data music_dataset.csv --model ensemble
```

Or do the same thing interactively, comparing all five side by side, in
`train_model.ipynb`.

### 4. Predict on new data from the CLI

```bash
python -m src.predict --input new_listening_rows.csv --model xgboost --out predictions.csv --recommend
```

### 5. Run the deployable web app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (typically **http://localhost:8501**) — predict
genre/artist for a hypothetical listener, get song recommendations, or explore the
dataset, right from the browser. The app auto-loads whichever `*_genre_predictor.pkl` /
`*_artist_predictor.pkl` / `knn_recommender.pkl` files exist at the repo root, so
retraining with a different `--model` and reloading the page is all it takes to swap
models.

---

## 🔁 Using your own dataset

Nothing in `src/` needs to change. Just point `--data` at your CSV (or edit `DATA_PATH`
in `train_model.ipynb`). Under the hood:

1. `schema.py` scans your column headers for genre/artist/track/user-like names
   (see `config.yaml → schema.*_candidates` — extend these lists if your dataset uses
   unusual naming).
2. Every remaining numeric column becomes a model feature; every remaining low-cardinality
   text column (≤ `max_categorical_cardinality`, default 50 unique values) becomes a
   one-hot feature; free-text/ID-like columns (URLs, lyrics, raw IDs) are automatically
   excluded.
3. If your dataset has no genre column, genre training is skipped automatically (same
   for artist) — the recommender still trains on whatever features exist.
4. Retrain (`python -m src.train --data your_dataset.csv --model xgboost`), then
   `python app.py` — the app picks up the new `.pkl` files and dataset automatically.

If your dataset uses very unusual column names that the candidate lists don't catch,
add them to `config.yaml`:

```yaml
schema:
  genre_candidates: ["genre", "genres", "track_genre", "category", "style", "mood_tag"]
```

---

## 🧠 Methodology

| Task                  | Approach                                                             |
|------------------------|-----------------------------------------------------------------------|
| Genre prediction       | Multi-class classification (KNN / Bagging / Boosting / XGBoost / Ensemble) |
| Artist prediction      | Multi-class classification (same models, same pipeline)               |
| Song recommendation    | KNN similarity search over the numeric/categorical feature space      |

**Why KNN + Bagging + Boosting + XGBoost?** Each algorithm has different strengths:
KNN captures local similarity structure well but is sensitive to noise; bagging
(Random Forest) reduces variance by averaging many decorrelated trees; boosting
(Gradient Boosting / XGBoost) reduces bias by sequentially correcting errors, with
XGBoost adding regularization and speed on top of sklearn's Gradient Boosting. The
`ensemble` option soft-votes across KNN + Bagging + XGBoost, generally giving the
most robust result across different datasets — which matters when the tool is meant
to generalize to whatever CSV you hand it.

Metrics reported after training: **accuracy**, **macro F1**, **weighted F1**, and
**top-3 accuracy** (useful in a recommendation-flavored setting where the "correct"
genre is often in the top few predictions even if not the single top one).

**A note on artist accuracy in the bundled dataset:** genre prediction reaches
~85-88% accuracy, but artist prediction stays near the chance baseline (~2.5% for
40 artists), because in this dataset an artist's catalogue is spread fairly evenly
across genres and audio-feature ranges — audio stats alone don't identify *who* made
a track, only *what it sounds like*. This is a property of the data, not a modeling
bug (see the EDA cross-tab in `train_model.ipynb`, section 1). A real artist-ID system
would need artist-level signal such as vocal-timbre embeddings, production style, or
listening co-occurrence data.

---

## 🧪 Testing

```bash
pytest -v
```

The suite trains and evaluates all five model types (including XGBoost) on the bundled
dataset, and separately verifies schema auto-detection against a second, differently-
structured synthetic dataset (different column names, different casing, different
target labels) to confirm the pipeline isn't secretly hard-coded to one CSV's headers.

---

## 🌐 Deployment

**Streamlit Community Cloud (free, recommended):**
1. Push this repo to GitHub — include the `.pkl` artifacts, `schema.pkl`,
   `metrics.json`, and `music_dataset.csv` (all small enough to commit directly).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → point it at
   this repo, branch, and `app.py`.
3. Done — `requirements.txt` is installed automatically, no server config needed.

**Any other Python host (Render / Railway / Fly.io / a VM):**
```bash
pip install -r requirements.txt
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

**Docker:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
```bash
docker build -t music-recommender .
docker run -p 8501:8501 music-recommender
```

---

## 🛣️ Roadmap / ideas for contributors

- Add collaborative-filtering (matrix factorization) as an alternative recommender mode
  when a user-item interaction column is present.
- Add SHAP-based explainability for genre/artist predictions.
- Add hyperparameter search (`GridSearchCV` / `Optuna`) wired to `config.yaml`.
- Add support for streaming/incremental training on very large datasets.

Contributions welcome — open an issue or PR.

---

## 📄 License

MIT — see [LICENSE](LICENSE).
