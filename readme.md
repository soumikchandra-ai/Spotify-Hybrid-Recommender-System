#  Spotify Hybrid Song Recommender System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)

A music recommendation engine that combines **Content-Based Filtering**, **Collaborative Filtering**, and a **Hybrid approach** to surface personalized song suggestions — powered by the Spotify Million Song Dataset and deployed via an interactive Streamlit web application.

---

##  Overview

### Problem Statement

Music streaming platforms serve millions of users daily, and the quality of song recommendations directly drives user engagement and retention. Generic popularity-based recommendations fail to capture individual taste. This project addresses that gap by building a multi-strategy recommender that understands both **song characteristics** (audio features, tags, artist) and **user listening behavior** (play counts, interaction history).

### Why the Spotify Million Song Dataset?

The dataset provides rich, real-world data across two complementary dimensions:

- **Music metadata** — audio features (danceability, energy, tempo, valence, etc.), artist, tags, and year, enabling content-based analysis.
- **User listening history** — anonymized play counts per user per track, enabling collaborative filtering based on actual behavior.

This combination is essential for building a robust hybrid system that avoids the cold-start problem of purely collaborative approaches and the narrowness of purely content-based ones.

### Objectives

- Build three recommendation strategies: content-based, collaborative, and hybrid.
- Allow users to tune the **diversity vs. familiarity** trade-off via a weighted blend.
- Deliver an interactive web interface where users can preview recommendations with Spotify audio clips.
- Provide a clean, modular codebase suitable for extension and deployment.

### Expected Outcomes

- Accurate song recommendations based on audio feature similarity.
- User-behavior-informed suggestions leveraging listening patterns.
- A configurable hybrid system whose balance between strategies can be adjusted at inference time.

---

##  Features

- **Three recommendation modes** — Content-Based, Collaborative Filtering, and a Hybrid system selectable at runtime.
- **Diversity slider** — Adjusts the content-based vs. collaborative weight dynamically, giving users control over how exploratory or familiar their recommendations feel.
- **Audio preview integration** — Each recommendation surfaces a Spotify 30-second preview clip directly in the UI.
- **Scalable data pipeline** — Dask-powered ingestion of large user listening history files that exceed in-memory limits.
- **Sparse matrix optimization** — Interaction and feature matrices stored and computed as CSR sparse matrices for memory efficiency.
- **Modular architecture** — Each filtering strategy is an independent, importable module with a clean API.
- **Persistent transformer** — The fitted `ColumnTransformer` is serialized with `joblib` so the app loads instantly without re-training.

---

##  Dataset

### Source

- **Music Info** — Song metadata including audio features, tags, artist, and Spotify preview URLs.
- **User Listening History** — Anonymized user–track play count records from the Million Song Dataset ecosystem.

> The dataset is not included in this repository due to size constraints. Dataset Link: [https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm]

### Key Columns

| Column | Description |
|---|---|
| `track_id` | Unique song identifier (links metadata to listening history) |
| `name` | Song title |
| `artist` | Artist name |
| `year` | Release year |
| `danceability` | Rhythm regularity score (0–1) |
| `energy` | Perceptual intensity and activity (0–1) |
| `tempo` | Estimated beats per minute |
| `loudness` | Overall loudness in decibels |
| `valence` | Musical positivity (0–1) |
| `speechiness` | Presence of spoken words (0–1) |
| `acousticness` | Acoustic confidence score (0–1) |
| `instrumentalness` | Predicts absence of vocals (0–1) |
| `liveness` | Presence of audience in recording (0–1) |
| `key` | Musical key of the track |
| `time_signature` | Estimated time signature |
| `tags` | Free-text genre/mood tags |
| `spotify_preview_url` | 30-second Spotify audio clip URL |
| `user_id` | Anonymized listener ID (listening history) |
| `playcount` | Number of times user played the track |

### Data Preprocessing

- Removed duplicate entries keyed on `spotify_id`.
- Dropped `genre` and `spotify_id` columns (redundant or sparse).
- Imputed missing `tags` values with `"no_tags"`.
- Normalized text fields (`name`, `artist`, `tags`) to lowercase and stripped whitespace.
- Filtered the song catalog to only tracks present in both the metadata and listening history (to enable hybrid recommendations).

### Feature Engineering

| Feature Group | Columns | Transformation |
|---|---|---|
| Audio intensity | `duration_ms`, `loudness`, `tempo` | `StandardScaler` (zero mean, unit variance) |
| Perceptual scores | `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence` | `MinMaxScaler` (0–1 range) |
| Artist / key / signature | `artist`, `key`, `time_signature` | `OneHotEncoder` (sparse, ignores unseen) |
| Release year | `year` | `CountEncoder` (normalized frequency encoding) |
| Tags | `tags` | `TfidfVectorizer` (top 85 features) |
| Interaction signals | `playcount` | Aggregated user–track play counts → CSR sparse matrix |

---

##  Project Architecture / Workflow

```
Raw Data
   │
   ▼
1. Data Cleaning (data_cleaning.py)
   ├── Deduplication, column pruning, null imputation
   └── Outputs: cleaned_data.csv
   │
   ▼
2. Content-Based Pipeline (content_based_filtering.py)
   ├── Feature engineering via ColumnTransformer
   ├── Fit & serialize transformer (transformer.joblib)
   ├── Transform features → sparse matrix
   └── Outputs: transformed_data.npz
   │
   ▼
3. Collaborative Filtering Pipeline (collaborative_filtering.py)
   ├── Load listening history with Dask (handles large files)
   ├── Inner join metadata ↔ history on track_id
   ├── Build user–track interaction matrix (CSR format)
   └── Outputs: interaction_matrix.npz, track_ids.npy, collab_filtered_data.csv
   │
   ▼
4. Hybrid Feature Preparation (transform_filtered_data.py)
   ├── Apply saved transformer to collaborative-filtered subset
   └── Outputs: transformed_hybrid_data.npz
   │
   ▼
5. Recommendation Engine
   ├── Content-Based: cosine similarity on feature vectors
   ├── Collaborative: cosine similarity on interaction vectors
   └── Hybrid: weighted combination of normalized scores
   │
   ▼
6. Streamlit App (app.py)
   └── Interactive UI with song search, mode selection, diversity slider, audio previews
```

### Similarity Computation

All three strategies use **cosine similarity** as the distance metric:

- **Content-Based** — similarity between TF-IDF + audio feature vectors.
- **Collaborative** — similarity between user play-count row vectors.
- **Hybrid** — min-max normalized scores from both methods combined as:

```
final_score = (w_content × norm_content_score) + (w_collab × norm_collab_score)
```

where `w_content + w_collab = 1` and the weights are user-controlled via the diversity slider.

---

##  Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Core language |
| Pandas | Latest | Data manipulation and cleaning |
| NumPy | Latest | Numerical operations, array indexing |
| Scikit-Learn | Latest | Feature transformers, cosine similarity |
| category-encoders | Latest | Count/frequency encoding for `year` |
| SciPy | Latest | Sparse matrix storage (`csr_matrix`, `.npz`) |
| Dask | Latest | Out-of-core loading of large CSV files |
| Joblib | Latest | Transformer serialization and caching |
| Streamlit | Latest | Interactive web application |

---

##  File Structure

```
spotify-hybrid-recommender/
│
├── data/                          # Dataset files (not tracked in git)
│   ├── Music Info.csv             # Raw song metadata with audio features
│   ├── User Listening History.csv # Raw user play count records
│   ├── cleaned_data.csv           # Output of data_cleaning.py
│   ├── collab_filtered_data.csv   # Songs present in both metadata and history
│   ├── track_ids.npy              # Ordered track ID array for matrix indexing
│   ├── interaction_matrix.npz     # Sparse user–track interaction matrix
│   ├── transformed_data.npz       # Content feature matrix (all songs)
│   └── transformed_hybrid_data.npz# Content feature matrix (collab-filtered songs)
│
├── notebooks/                     # Exploratory analysis notebooks
│
├── app.py                         # Streamlit web application (entry point)
├── data_cleaning.py               # Data loading and preprocessing pipeline
├── content_based_filtering.py     # Content-based recommender (training + inference)
├── collaborative_filtering.py     # Collaborative filtering (matrix build + inference)
├── hybrid_recommendation.py       # HybridRecommenderSystem class
├── transform_filtered_data.py     # Applies transformer to collab-filtered subset
│
├── transformer.joblib             # Serialized fitted ColumnTransformer
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

##  Installation

### 1. Clone the Repository

```bash
git clone https://github.com/soumikchandra-ai/Spotify-Hybrid-Recommender-System.git
cd spotify-hybrid-recommender-system
```

### 2. Create a Virtual Environment

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Place the following files inside the `data/` directory:

```
data/
├── Music Info.csv
└── User Listening History.csv
```

---

## How to Run

### Step 1 — Clean the Data

```bash
python data_cleaning.py
```

Outputs `data/cleaned_data.csv`.

### Step 2 — Build the Content-Based Transformer and Features

```bash
python content_based_filtering.py
```

Outputs `transformer.joblib` and `data/transformed_data.npz`.

### Step 3 — Build the Collaborative Filtering Matrix

```bash
python collaborative_filtering.py
```

Outputs `data/collab_filtered_data.csv`, `data/track_ids.npy`, and `data/interaction_matrix.npz`.

### Step 4 — Transform Features for Hybrid Mode

```bash
python transform_filtered_data.py
```

Outputs `data/transformed_hybrid_data.npz`.

### Step 5 — Launch the Web App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

##  Model Details

### Content-Based Filtering

- **Algorithm**: Cosine similarity over a dense feature representation.
- **Pipeline**: `ColumnTransformer` combining TF-IDF (tags), OneHotEncoding (artist, key, time_signature), CountEncoding (year), StandardScaling (duration, loudness, tempo), and MinMaxScaling (7 perceptual audio features).
- **Output**: Sparse feature matrix stored in `.npz` format for fast loading.
- **Inference**: Given a query song, its row vector is compared against all rows; top-k indices are returned.

### Collaborative Filtering

- **Algorithm**: Item-based cosine similarity on a user–track interaction matrix.
- **Matrix**: Tracks × Users CSR sparse matrix where values are aggregated play counts.
- **Filtering**: Only tracks present in both the metadata catalog and the listening history are included, ensuring every recommendation has displayable metadata.
- **Inference**: The row vector for the query track is compared against all track rows; top-k indices are retrieved via `np.argsort`.

### Hybrid Recommender System

- **Strategy**: Weighted linear combination of normalized content and collaborative scores.
- **Normalization**: Min-max normalization applied independently to each score vector before combining, ensuring comparable scales.
- **Weight control**: `weight_content_based` ∈ [0, 1] is set at inference time via the Streamlit diversity slider:

| Diversity Slider | Content Weight | Collaborative Weight | Effect |
|---|---|---|---|
| 1 (low diversity) | 0.90 | 0.10 | Highly similar songs |
| 5 (balanced) | 0.50 | 0.50 | Balanced recommendations |
| 10 (high diversity) | 0.00 | 1.00 | Behavior-driven, exploratory |

---

##  Results

The system produces ranked recommendation lists evaluated qualitatively through the Streamlit interface (audio previews allow immediate human judgment of relevance). Quantitative offline evaluation was not the primary goal of this project; the focus was on building a functional, configurable, and interactive recommendation pipeline.

Key findings:

- Content-based filtering performs well for songs with rich tag information and distinct audio profiles.
- Collaborative filtering surfaces surprising but relevant recommendations by leveraging listener co-occurrence patterns.
- The hybrid approach consistently outperforms either method in isolation when the diversity weight is balanced (0.4–0.6 content weight), reducing the echo-chamber effect of pure content similarity while retaining audio coherence.

---

##  Future Improvements

- **Offline evaluation metrics** — Implement Precision@K, Recall@K, and NDCG using held-out listening history to quantify recommendation quality.
- **Matrix Factorization** — Replace item-item cosine similarity with SVD or ALS (via `implicit`) for scalable latent-factor collaborative filtering.
- **Real-time Spotify API integration** — Fetch live track metadata, album art, and full 30-second previews via the Spotify Web API.
- **User session personalization** — Allow users to build a listening queue in-session and re-rank recommendations based on accumulated preferences.
- **Cold-start handling** — Implement a popularity-based fallback for new songs with no listening history.
- **Containerization** — Dockerize the application for reproducible deployment on cloud platforms (AWS, GCP, Azure).
- **DVC pipeline** — Leverage the existing `dvc` dependency to version datasets and pipeline stages for reproducibility.

---

##  Author

**Soumik Chandra**

---

> *Built as a portfolio project demonstrating end-to-end ML system design: data engineering, feature transformation, multi-strategy recommendation, and interactive deployment.*
