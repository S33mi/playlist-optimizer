"""
Playlist Optimizer – Gradio Demo
Run:  python app.py
"""

import os
from io import BytesIO
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import joblib
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# Config
# -------------------------------------------------
BASE = "https://raw.githubusercontent.com/S33mi/spotify-music-recommender/main"

URL_PART1 = f"{BASE}/data/processed/spotify_with_clusters_part1.csv"
URL_PART2 = f"{BASE}/data/processed/spotify_with_clusters_part2.csv"
URL_X     = f"{BASE}/data/processed/X_scaled.npy"
URL_MOOD  = f"{BASE}/data/models/mood_labels.pkl"

LOCAL_PART1 = "data/processed/spotify_with_clusters_part1.csv"
LOCAL_PART2 = "data/processed/spotify_with_clusters_part2.csv"
LOCAL_CSV   = "data/processed/spotify_with_clusters.csv"
LOCAL_X     = "data/processed/X_scaled.npy"
LOCAL_MOOD  = "data/models/mood_labels.pkl"

# -------------------------------------------------
# Load helpers
# -------------------------------------------------
def _download(url: str) -> bytes:
    print(f"Downloading {url} ...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def load_dataframe() -> pd.DataFrame:
    if os.path.exists(LOCAL_CSV):
        print(f"Loading local: {LOCAL_CSV}")
        return pd.read_csv(LOCAL_CSV)
    if os.path.exists(LOCAL_PART1) and os.path.exists(LOCAL_PART2):
        print("Merging local part1 + part2 ...")
        return pd.concat(
            [pd.read_csv(LOCAL_PART1), pd.read_csv(LOCAL_PART2)],
            ignore_index=True,
        )
    print("Downloading split CSVs from GitHub ...")
    with open("_p1.tmp", "wb") as f:
        f.write(_download(URL_PART1))
    with open("_p2.tmp", "wb") as f:
        f.write(_download(URL_PART2))
    df = pd.concat(
        [pd.read_csv("_p1.tmp"), pd.read_csv("_p2.tmp")],
        ignore_index=True,
    )
    os.remove("_p1.tmp")
    os.remove("_p2.tmp")
    return df

def load_npy(path: str, url: str) -> np.ndarray:
    if os.path.exists(path):
        print(f"Loading local: {path}")
        return np.load(path)
    return np.load(BytesIO(_download(url)))

def load_pkl(path: str, url: str):
    if os.path.exists(path):
        print(f"Loading local: {path}")
        return joblib.load(path)
    return joblib.load(BytesIO(_download(url)))

# -------------------------------------------------
# Load once at startup
# -------------------------------------------------
print("Loading artifacts ...")
df = load_dataframe()
X = load_npy(LOCAL_X, URL_X)
mood_labels = load_pkl(LOCAL_MOOD, URL_MOOD)
if "mood" not in df.columns and "cluster" in df.columns:
    df["mood"] = df["cluster"].map(mood_labels)
assert len(df) == len(X)
print(f"Ready – {len(df)} tracks\n")

# -------------------------------------------------
# Clean the dataset (in case of NaN value)
# -------------------------------------------------
print("Original shape:", df.shape)

# 1. Remove rows with missing critical information
critical_cols = ["track_name", "artists", "track_genre"]
df = df.dropna(subset=critical_cols)

# 2. Fill missing numerical values
for col in ["energy", "tempo", "popularity"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# 3. Reset index so iloc works perfectly (very important!)
df = df.reset_index(drop=True)

# 4. Keep X aligned with the cleaned df
# (Assuming X was loaded in the same order as original df)
X = X[df.index] if len(X) != len(df) else X   # safe alignment

print("Cleaned shape:", df.shape)
print("Remaining NaNs in critical columns:")
print(df[critical_cols].isna().sum())

# -------------------------------------------------
# Scoring + SA
# -------------------------------------------------
def smoothness_score(energy, tempo):
    if len(energy) < 2:
        return 1.0
    e_jump = np.abs(np.diff(energy)).mean()
    t_jump = (np.abs(np.diff(tempo)) / 50.0).mean()
    return 1.0 / (1.0 + 0.6 * e_jump + 0.4 * t_jump)

def total_score(seed_idx, indices):
    energy = df.iloc[indices]["energy"].values
    tempo = df.iloc[indices]["tempo"].values
    sims = cosine_similarity(X[seed_idx].reshape(1, -1), X[indices])[0]
    s_sim = float(sims.mean())
    s_smooth = smoothness_score(energy, tempo)
    if len(indices) > 1:
        sim_m = cosine_similarity(X[indices])
        triu = sim_m[np.triu_indices(len(indices), k=1)]
        s_div = float(1.0 - triu.mean())
    else:
        s_div = 1.0
    if "mood" in df.columns:
        seed_mood = df.iloc[seed_idx]["mood"]
        s_mood = float((df.iloc[indices]["mood"] == seed_mood).mean())
    else:
        s_mood = 0.5
    total = 0.30 * s_smooth + 0.30 * s_sim + 0.25 * s_div + 0.15 * s_mood
    return total, {
        "smoothness": s_smooth,
        "similarity": s_sim,
        "diversity": s_div,
        "mood": s_mood,
    }

def nn_playlist(seed_idx, length):
    sims = cosine_similarity(X[seed_idx].reshape(1, -1), X)[0]
    sims[seed_idx] = -np.inf
    return np.argsort(sims)[-length:][::-1].tolist()

def neighbor(state, seed_idx, n_tracks):
    s = state[:]
    length = len(s)
    move = random.random()
    if move < 0.35:
        i, j = random.sample(range(length), 2)
        s[i], s[j] = s[j], s[i]
    elif move < 0.65:
        i, j = random.sample(range(length), 2)
        track = s.pop(i)
        s.insert(j, track)
    else:
        pos = random.randrange(length)
        used = set(s)
        candidates = [t for t in range(n_tracks) if t != seed_idx and t not in used]
        if candidates:
            s[pos] = random.choice(candidates)
    return s

def run_sa(seed_idx, length, steps=2500, T0=0.12, alpha=0.995):
    n_tracks = len(df)
    current = nn_playlist(seed_idx, length)
    cur_score, _ = total_score(seed_idx, current)
    best, best_score = current[:], cur_score
    T = T0
    for _ in range(steps):
        cand = neighbor(current, seed_idx, n_tracks)
        cand_score, _ = total_score(seed_idx, cand)
        delta = cur_score - cand_score
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
            current, cur_score = cand, cand_score
            if cur_score > best_score:
                best, best_score = current[:], cur_score
        T *= alpha
    return best, best_score

# -------------------------------------------------
# Gradio callbacks
# -------------------------------------------------
def search_tracks(query, top_n=8):
    if not query or len(query.strip()) < 2:
        return gr.update(choices=[], value=None)
    q = query.lower().strip()
    mask = (
        df["track_name"].str.lower().str.contains(q, na=False)
        | df["artists"].str.lower().str.contains(q, na=False)
    )
    results = df.loc[mask, ["track_id", "track_name", "artists"]].head(top_n)
    choices = [
        f"{r.track_name} — {r.artists} ({r.track_id})"
        for _, r in results.iterrows()
    ]
    return gr.update(choices=choices, value=None)

def optimize_playlist(selected_track, length, steps):
    if not selected_track:
        return "Please search and select a seed track.", None, None
    try:
        track_id = selected_track.rsplit("(", 1)[-1].replace(")", "").strip()
    except Exception:
        return "Could not parse track id.", None, None

    matches = df[df["track_id"] == track_id]
    if matches.empty:
        return "Track not found.", None, None

    seed_idx = df.index.get_loc(matches.index[0])
    seed = matches.iloc[0]

    best, score = run_sa(seed_idx, int(length), steps=int(steps))
    _, comps = total_score(seed_idx, best)

    info = f"""
**Seed Track**  
**Name**: {seed['track_name']}  
**Artists**: {seed['artists']}  
**Genre**: {seed.get('track_genre', 'N/A')}  
**Mood**: {seed.get('mood', 'N/A')}  
**Popularity**: {seed['popularity']}  

**Optimized Score**: {score:.3f}  
- Smoothness: {comps['smoothness']:.3f}  
- Similarity: {comps['similarity']:.3f}  
- Diversity: {comps['diversity']:.3f}  
- Mood consistency: {comps['mood']:.3f}
"""

    cols = ["track_name", "artists", "track_genre", "popularity"]
    for c in ["mood", "energy", "tempo"]:
        if c in df.columns:
            cols.append(c)
    playlist_df = df.iloc[best][cols].copy()
    playlist_df.index = range(1, len(playlist_df) + 1)

    fig, ax1 = plt.subplots(figsize=(9, 3.5))
    x = np.arange(1, len(best) + 1)
    energy = df.iloc[best]["energy"].values
    tempo = df.iloc[best]["tempo"].values
    ax1.plot(x, energy, "o-", color="steelblue")
    ax1.set_ylabel("Energy", color="steelblue")
    ax1.set_xlabel("Position")
    ax2 = ax1.twinx()
    ax2.plot(x, tempo, "s--", color="darkorange")
    ax2.set_ylabel("Tempo (BPM)", color="darkorange")
    plt.title("Optimized Playlist – Energy & Tempo")
    fig.tight_layout()

    return info, playlist_df, fig

# -------------------------------------------------
# UI
# -------------------------------------------------
with gr.Blocks(title="Playlist Optimizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Playlist Optimizer
        Content-based playlist optimization using **Simulated Annealing**.  
        Balances smoothness, seed similarity, diversity, and mood consistency.
        """
    )
    with gr.Row():
        search_box = gr.Textbox(
            label="Search seed track or artist",
            placeholder="e.g. Blinding Lights, The Weeknd...",
        )
        search_btn = gr.Button("Search", variant="primary")

    track_dropdown = gr.Dropdown(label="Select seed track", choices=[], interactive=True)

    with gr.Row():
        length_slider = gr.Slider(6, 20, value=12, step=1, label="Playlist length")
        steps_slider = gr.Slider(800, 5000, value=2500, step=100, label="SA steps")

    run_btn = gr.Button("Optimize Playlist", variant="primary")

    seed_out = gr.Markdown(label="Seed & Scores")
    table_out = gr.Dataframe(label="Optimized Playlist")
    plot_out = gr.Plot(label="Energy & Tempo progression")

    search_btn.click(fn=search_tracks, inputs=search_box, outputs=track_dropdown)
    run_btn.click(
        fn=optimize_playlist,
        inputs=[track_dropdown, length_slider, steps_slider],
        outputs=[seed_out, table_out, plot_out],
    )

if __name__ == "__main__":
    demo.launch(share=False)
