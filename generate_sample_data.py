"""
generate_sample_data.py
-----------------------
Creates a synthetic music-listening dataset (data/sample_music_data.csv)
so the project works out-of-the-box for demos and tests, without
requiring you to supply a real dataset first.

Run:
    python generate_sample_data.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N = 1200
genres = ["Pop", "Rock", "Hip-Hop", "Jazz", "Classical", "Electronic", "Country", "R&B"]
artists = [f"Artist_{i}" for i in range(1, 41)]

genre_choice = np.random.choice(genres, size=N, p=[0.22, 0.16, 0.16, 0.08, 0.08, 0.14, 0.08, 0.08])

# give each genre a distinct "sound profile" so the classifiers have real signal to learn
genre_profile = {
    "Pop":        dict(energy=0.75, danceability=0.75, valence=0.7, tempo=118, acousticness=0.15, loudness=-5),
    "Rock":       dict(energy=0.85, danceability=0.5,  valence=0.55, tempo=128, acousticness=0.1,  loudness=-4),
    "Hip-Hop":    dict(energy=0.7,  danceability=0.8,  valence=0.5,  tempo=95,  acousticness=0.08, loudness=-6),
    "Jazz":       dict(energy=0.4,  danceability=0.45, valence=0.55, tempo=110, acousticness=0.6,  loudness=-11),
    "Classical":  dict(energy=0.25, danceability=0.3,  valence=0.4,  tempo=90,  acousticness=0.9,  loudness=-16),
    "Electronic": dict(energy=0.8,  danceability=0.72, valence=0.6,  tempo=126, acousticness=0.05, loudness=-6),
    "Country":    dict(energy=0.55, danceability=0.55, valence=0.65, tempo=105, acousticness=0.35, loudness=-7),
    "R&B":        dict(energy=0.55, danceability=0.68, valence=0.5,  tempo=90,  acousticness=0.25, loudness=-8),
}

rows = []
for i in range(N):
    g = genre_choice[i]
    p = genre_profile[g]
    row = {
        "track_id": f"trk_{i:05d}",
        "song_name": f"Song {i}",
        "artist_name": np.random.choice(artists),
        "genre": g,
        "energy": np.clip(np.random.normal(p["energy"], 0.1), 0, 1),
        "danceability": np.clip(np.random.normal(p["danceability"], 0.1), 0, 1),
        "valence": np.clip(np.random.normal(p["valence"], 0.12), 0, 1),
        "tempo": max(40, np.random.normal(p["tempo"], 10)),
        "acousticness": np.clip(np.random.normal(p["acousticness"], 0.08), 0, 1),
        "loudness": np.random.normal(p["loudness"], 1.5),
        "instrumentalness": np.clip(np.random.beta(2, 5), 0, 1),
        "duration_ms": int(np.random.normal(210000, 35000)),
        "popularity": int(np.clip(np.random.normal(50, 20), 0, 100)),
        "explicit": np.random.choice(["Yes", "No"], p=[0.2, 0.8]),
        "key_signature": np.random.choice(["C", "D", "E", "F", "G", "A", "B"]),
    }
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("music_dataset.csv", index=False)
print(f"Wrote music_dataset.csv with {len(df)} rows and {len(df.columns)} columns.")
