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

playlist-optimizer/

├── data/                       # links or copies of processed features

├── notebooks/

│   ├── 01_problem_setup.ipynb

│   ├── 02_baseline_playlists.ipynb

│   ├── 03_genetic_algorithm.ipynb

│   └── 04_simulated_annealing.ipynb

├── src/

│   ├── objectives.py           # playlist scoring functions

│   ├── constraints.py

│   ├── ga_optimizer.py

│   ├── sa_optimizer.py

│   └── utils.py

├── models/                     # optional saved optimizers / configs

├── requirements.txt

├── README.md

└── LICENSE

---

## 🚀 Getting Started

1. Clone the repository
```bash
git clone https://github.com/S33mi/playlist-optimizer.git
cd playlist-optimizer
```
2. Create a virtual environment and install dependencies

```Bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
3. Make sure you have the processed artifacts from the recommender project
(or download them from that repo):
scaled feature matrix
track metadata + mood/cluster labels

4. Open the notebooks in order and run them.

---
## 🧠 Approach

1. Problem Formulation
   
    - Decision variables: ordered sequence of tracks
      - Objectives (examples):
        - Minimize sudden jumps in energy / tempo
        - Maximize average similarity to seed / mood consistency
        - Maximize diversity
      -Constraints:
        - Fixed playlist length
        - Duration range
        - Optional genre or popularity limits


2. Baselines
   
    - Random playlists
    - Pure nearest-neighbor / similarity-based playlists

3. Metaheuristics
    - Genetic Algorithm (selection, crossover, mutation on track sequences)
    - Simulated Annealing (neighborhood moves: swap, insert, reverse)

4. Evaluation
    - Objective score comparison
    - Energy / tempo progression plots
    - Qualitative listening checks

---

## 📊 Results

---

## 🔗 Related Project

[spotify-music-recommender](https://github.com/S33mi/spotify-music-recommender) – content-based recommender + mood clustering used as the foundation for this optimizer.


## 🎯 Future Improvements

- Multi-objective optimization (Pareto front)
- User preference weights (energy vs diversity vs mood)
- Interactive Gradio / Streamlit demo
- Integration with Spotify playlist export
- Hybrid methods (GA + local search)

---
## 📄 License
MIT License
---
Author: [S33mi](https://github.com/S33mi)

Open to Data Analyst / Machine Learning opportunities.
