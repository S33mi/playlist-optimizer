# Playlist Optimizer

Treat playlist creation as an **optimization problem**.

Given a seed track (or a set of preferred tracks), this project builds playlists that balance musical flow, mood consistency, diversity, and practical constraints using metaheuristics (Genetic Algorithm, Simulated Annealing, etc.).

**Builds on**: [spotify-music-recommender](https://github.com/S33mi/spotify-music-recommender)  
(audio features, content-based similarity, and mood clusters)

---

## 🎯 Goal

Create playlists that are not just “similar songs”, but **optimized sequences**:

- Smooth energy / tempo transitions
- Mood consistency (or controlled progression)
- Diversity (avoid near-duplicates)
- Duration and track-count constraints
- Optional genre balance

This connects music recommendation with classical optimization techniques.

---

## 🔑 Key Features

- Load precomputed Spotify audio features + mood clusters
- Define multi-objective playlist quality score
- Optimize playlists with:
  - Genetic Algorithm
  - Simulated Annealing
  - (Optional) simple greedy / random baselines
- Compare optimized playlists vs random / similarity-only playlists
- Export results and simple visualizations of energy/tempo curves

---

## 🛠️ Tech Stack

- **Python**
- **Data**: pandas, numpy
- **Optimization**: custom GA / SA (or DEAP / mealpy later)
- **ML artifacts**: scikit-learn (from previous project)
- **Visualization**: matplotlib, seaborn
- **Notebooks**: Jupyter

---

## 📁 Project Structure
