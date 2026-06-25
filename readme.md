# Spotify Hybrid Song Recommender System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-29%20passed-brightgreen?logo=pytest&logoColor=white)
![Coverage](https://img.shields.io/badge/coverage-65%25-yellow)

A music recommendation engine that combines Content-Based Filtering, Collaborative Filtering, and a Hybrid approach to surface personalized song suggestions, powered by the Spotify Million Song Dataset and deployed via an interactive Streamlit web application.

---

## Overview

### Market Gap

Music streaming platforms serve millions of users daily, and the quality of song recommendations directly drives user engagement and retention. Generic popularity-based recommendations fail to capture individual taste. This project addresses that gap by building a multi-strategy recommender that understands both song characteristics (audio features, tags, artist) and user listening behavior (play counts, interaction history).

### Why the Spotify Million Song Dataset?

The dataset provides rich, real-world data across two complementary dimensions:

- **Music metadata** — audio features (danceability, energy, tempo, valence, etc.), artist, tags, and year, enabling content-based analysis.
- **User listening history** — anonymized play counts per user per track, enabling collaborative filtering based on actual behavior.

This combination is essential for building a robust hybrid system that avoids the cold-start problem of purely collaborative approaches and the narrowness of purely content-based ones.

### Objectives

- Build three recommendation strategies: content-based, collaborative, and hybrid.
- Allow users to tune the diversity vs. familiarity trade-off via a weighted blend.
- Deliver an interactive web interface where users can preview recommendations with Spotify audio clips.
- Provide a clean, modular, tested codebase suitable for extension and deployment.

### Expected Outcomes

- Accurate song recommendations based on audio feature similarity.
- User-behavior-informed suggestions leveraging listening patterns.
- A configurable hybrid system whose balance between strategies can be adjusted at inference time.
- A verified recommendation pipeline with a comprehensive automated test suite covering unit, integration, and edge-case scenarios.

---

## Features

- **Three recommendation modes** — Content-Based, Collaborative Filtering, and a Hybrid system selectable at runtime.
- **Diversity slider** — Adjusts the content-based vs. collaborative weight dynamically, giving users control over how exploratory or familiar their recommendations feel.
- **Audio preview integration** — Each recommendation surfaces a Spotify 30-second preview clip directly in the UI.
- **Scalable data pipeline** — Dask-powered ingestion of large user listening history files that exceed in-memory limits.
- **Sparse matrix optimization** — Interaction and feature matrices stored and computed as CSR sparse matrices for memory efficiency.
- **Modular architecture** — Each filtering strategy is an independent, importable module with a clean API.
- **Persistent transformer** — The fitted ColumnTransformer is serialized with joblib so the app loads instantly without re-training.
- **Automated test suite** — 29 pytest tests covering all three recommendation modules with 65% overall code coverage and 96% coverage on the core Hybrid Recommender module.

---

## Dataset

### Source

- **Music Info** — Song metadata including audio features, tags, artist, and Spotify preview URLs.
- **User Listening History** — Anonymized user-track play count records from the Million Song Dataset ecosystem.

> The dataset is not included in this repository due to size constraints. Dataset link: [https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm]

### Key Columns

| Column | Description |
|---|---|
| `track_id` | Unique song identifier (links metadata to listening history) |
| `name` | Song title |
| `artist` | Artist name |
| `year` | Release year |
| `danceability` | Rhythm regularity score (0-1) |
| `energy` | Perceptual intensity and activity (0-1) |
| `tempo` | Estimated beats per minute |
| `loudness` | Overall loudness in decibels |
| `valence` | Musical positivity (0-1) |
| `speechiness` | Presence of spoken words (0-1) |
| `acousticness` | Acoustic confidence score (0-1) |
| `instrumentalness` | Predicts absence of vocals (0-1) |
| `liveness` | Presence of audience in recording (0-1) |
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
- Filtered the song catalog to only tracks present in both the metadata and listening history to enable hybrid recommendations.

### Feature Engineering

| Feature Group | Columns | Transformation |
|---|---|---|
| Audio intensity | `duration_ms`, `loudness`, `tempo` | StandardScaler (zero mean, unit variance) |
| Perceptual scores | `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence` | MinMaxScaler (0-1 range) |
| Artist / key / signature | `artist`, `key`, `time_signature` | OneHotEncoder (sparse, ignores unseen) |
| Release year | `year` | CountEncoder (normalized frequency encoding) |
| Tags | `tags` | TfidfVectorizer (top 85 features) |
| Interaction signals | `playcount` | Aggregated user-track play counts as CSR sparse matrix |

---

## Project Architecture / Workflow

```
Raw Data
   |
   v
1. Data Cleaning (data_cleaning.py)
   |-- Deduplication, column pruning, null imputation
   +-- Outputs: cleaned_data.csv
   |
   v
2. Content-Based Pipeline (content_based_filtering.py)
   |-- Feature engineering via ColumnTransformer
   |-- Fit & serialize transformer (transformer.joblib)
   |-- Transform features to sparse matrix
   +-- Outputs: transformed_data.npz
   |
   v
3. Collaborative Filtering Pipeline (collaborative_filtering.py)
   |-- Load listening history with Dask (handles large files)
   |-- Inner join metadata with history on track_id
   |-- Build user-track interaction matrix (CSR format)
   +-- Outputs: interaction_matrix.npz, track_ids.npy, collab_filtered_data.csv
   |
   v
4. Hybrid Feature Preparation (transform_filtered_data.py)
   |-- Apply saved transformer to collaborative-filtered subset
   +-- Outputs: transformed_hybrid_data.npz
   |
   v
5. Recommendation Engine
   |-- Content-Based: cosine similarity on feature vectors
   |-- Collaborative: cosine similarity on interaction vectors
   +-- Hybrid: weighted combination of normalized scores
   |
   v
6. Streamlit App (app.py)
   +-- Interactive UI with song search, mode selection, diversity slider, audio previews
```

### Similarity Computation

All three strategies use cosine similarity as the distance metric:

- **Content-Based** — similarity between TF-IDF + audio feature vectors.
- **Collaborative** — similarity between user play-count row vectors.
- **Hybrid** — min-max normalized scores from both methods combined as:

```
final_score = (w_content x norm_content_score) + (w_collab x norm_collab_score)
```

where `w_content + w_collab = 1` and the weights are user-controlled via the diversity slider.

---

## Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Core language |
| Pandas | Latest | Data manipulation and cleaning |
| NumPy | Latest | Numerical operations, array indexing |
| Scikit-Learn | Latest | Feature transformers, cosine similarity |
| category-encoders | Latest | Count/frequency encoding for `year` |
| SciPy | Latest | Sparse matrix storage (csr_matrix, .npz) |
| Dask | Latest | Out-of-core loading of large CSV files |
| Joblib | Latest | Transformer serialization and caching |
| Streamlit | Latest | Interactive web application |
| pytest | Latest | Automated unit and integration testing |
| pytest-cov | Latest | Code coverage reporting |

---

## File Structure

```
spotify-hybrid-recommender/
|
|-- data/                            # Dataset files (not tracked in git)
|   |-- Music Info.csv               # Raw song metadata with audio features
|   |-- User Listening History.csv   # Raw user play count records
|   |-- cleaned_data.csv             # Output of data_cleaning.py
|   |-- collab_filtered_data.csv     # Songs present in both metadata and history
|   |-- track_ids.npy                # Ordered track ID array for matrix indexing
|   |-- interaction_matrix.npz       # Sparse user-track interaction matrix
|   |-- transformed_data.npz         # Content feature matrix (all songs)
|   +-- transformed_hybrid_data.npz  # Content feature matrix (collab-filtered songs)
|
|-- notebooks/                       # Core recommendation modules
|   |-- data_cleaning.py                 # Data loading and preprocessing pipeline
|   |-- content_based_filtering.py       # Content-based recommender (training + inference)
|   |-- collaborative_filtering.py       # Collaborative filtering (matrix build + inference)
|   |-- hybrid_recommendation.py         # HybridRecommenderSystem class
|   |-- transform_filtered_data.py       # Applies transformer to collab-filtered subset
|
|-- tests/                           # Automated test suite
|   |-- __init__.py
|   |-- conftest.py                  # Shared fixtures (synthetic data, mock matrices)
|   |-- test_hybrid.py               # 13 tests: normalize, weighted combination, give_recommendations
|   |-- test_collaborative.py        # 8 tests: filter_songs_data, collaborative_recommendation
|   +-- test_content.py              # 8 tests: calculate_similarity_scores, content_recommend
|
|-- app.py                           # Streamlit web application (entry point)
|
|-- transformer.joblib               # Serialized fitted ColumnTransformer
|-- requirements.txt                 # Python dependencies
+-- README.md                        # Project documentation
```

---

## Testing

The project includes a comprehensive automated test suite built with pytest, covering all three recommendation modules across 29 test cases.

### Running the Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=notebooks --cov-report=term-missing -v
```

### Test Results

```
tests/test_collaborative.py::TestFilterSongsData::test_only_common_track_ids_remain     PASSED
tests/test_collaborative.py::TestFilterSongsData::test_output_index_is_reset            PASSED
tests/test_collaborative.py::TestFilterSongsData::test_empty_track_ids_returns_empty_df PASSED
tests/test_collaborative.py::TestFilterSongsData::test_file_is_saved                    PASSED
tests/test_collaborative.py::TestCollaborativeRecommendation::test_returns_k_plus_one_rows      PASSED
tests/test_collaborative.py::TestCollaborativeRecommendation::test_output_is_dataframe          PASSED
tests/test_collaborative.py::TestCollaborativeRecommendation::test_score_column_dropped         PASSED
tests/test_collaborative.py::TestCollaborativeRecommendation::test_name_and_artist_columns      PASSED
tests/test_content.py::TestCalculateSimilarityScores::test_output_shape_is_1_x_n        PASSED
tests/test_content.py::TestCalculateSimilarityScores::test_self_similarity_is_highest   PASSED
tests/test_content.py::TestCalculateSimilarityScores::test_scores_between_minus1_and_1  PASSED
tests/test_content.py::TestContentRecommend::test_returns_k_results                     PASSED
tests/test_content.py::TestContentRecommend::test_output_has_name_artist_url_columns    PASSED
tests/test_content.py::TestContentRecommend::test_song_not_in_dataset_raises_value_error PASSED
tests/test_content.py::TestContentRecommend::test_index_is_reset_in_output              PASSED
tests/test_content.py::TestContentRecommend::test_input_song_not_in_recommendations     PASSED
tests/test_hybrid.py::TestNormalizeSimilarities::test_output_range_between_0_and_1      PASSED
tests/test_hybrid.py::TestNormalizeSimilarities::test_minimum_becomes_zero              PASSED
tests/test_hybrid.py::TestNormalizeSimilarities::test_maximum_becomes_one               PASSED
tests/test_hybrid.py::TestNormalizeSimilarities::test_uniform_scores_edge_case          PASSED
tests/test_hybrid.py::TestNormalizeSimilarities::test_output_shape_preserved            PASSED
tests/test_hybrid.py::TestWeightedCombination::test_weights_sum_to_one                  PASSED
tests/test_hybrid.py::TestWeightedCombination::test_equal_weights                       PASSED
tests/test_hybrid.py::TestWeightedCombination::test_output_shape_matches_input          PASSED
tests/test_hybrid.py::TestGiveRecommendations::test_returns_correct_number              PASSED
tests/test_hybrid.py::TestGiveRecommendations::test_returns_dataframe                   PASSED
tests/test_hybrid.py::TestGiveRecommendations::test_invalid_song_raises_value_error     PASSED
tests/test_hybrid.py::TestGiveRecommendations::test_result_does_not_contain_track_id    PASSED
tests/test_hybrid.py::TestGiveRecommendations::test_weight_content_based_0_collaborative PASSED

29 passed in 16.54s
```

### What Is Tested

**test_hybrid.py (13 tests)**
- `__normalize_similarities`: output range, min-max boundary values, uniform input edge case, shape preservation
- `__weighted_combination`: pure content-only mode, equal-weight averaging, output shape
- `give_recommendations`: return type, output size, invalid song error handling, column cleanup, pure collaborative mode

**test_collaborative.py (8 tests)**
- `filter_songs_data`: track ID filtering correctness, index reset, empty input handling, file persistence
- `collaborative_recommendation`: output size, DataFrame return type, column cleanup, required column presence

**test_content.py (8 tests)**
- `calculate_similarity_scores`: output shape, self-similarity maximality, score range bounds
- `content_recommend`: k-result count, required columns, missing song error, index reset, input song exclusion

### Edge Cases Covered

- Songs not present in the dataset raise `ValueError` with a descriptive message
- Empty track ID lists produce an empty filtered DataFrame without error
- Uniform similarity scores (where max equals min) do not cause division by zero in normalization
- The queried song is excluded from its own recommendation output
- Output DataFrames never expose internal columns (`track_id`, `score`) to the caller
- Weight boundary conditions: `weight_content_based = 0.0` and `weight_content_based = 1.0`

---

## Installation

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
|-- Music Info.csv
+-- User Listening History.csv
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

## Model Details

### Content-Based Filtering

- **Algorithm**: Cosine similarity over a dense feature representation.
- **Pipeline**: ColumnTransformer combining TF-IDF (tags), OneHotEncoding (artist, key, time_signature), CountEncoding (year), StandardScaling (duration, loudness, tempo), and MinMaxScaling (7 perceptual audio features).
- **Output**: Sparse feature matrix stored in .npz format for fast loading.
- **Inference**: Given a query song, its row vector is compared against all rows and top-k indices are returned.

### Collaborative Filtering

- **Algorithm**: Item-based cosine similarity on a user-track interaction matrix.
- **Matrix**: Tracks x Users CSR sparse matrix where values are aggregated play counts.
- **Filtering**: Only tracks present in both the metadata catalog and the listening history are included, ensuring every recommendation has displayable metadata.
- **Inference**: The row vector for the query track is compared against all track rows and top-k indices are retrieved via `np.argsort`.

### Hybrid Recommender System

- **Strategy**: Weighted linear combination of normalized content and collaborative scores.
- **Normalization**: Min-max normalization applied independently to each score vector before combining, ensuring comparable scales.
- **Weight control**: `weight_content_based` in [0, 1] is set at inference time via the Streamlit diversity slider.

| Diversity Slider | Content Weight | Collaborative Weight | Effect |
|---|---|---|---|
| 1 (low diversity) | 0.90 | 0.10 | Highly similar songs |
| 5 (balanced) | 0.50 | 0.50 | Balanced recommendations |
| 10 (high diversity) | 0.00 | 1.00 | Behavior-driven, exploratory |

---

## Results

The system produces ranked recommendation lists evaluated qualitatively through the Streamlit interface (audio previews allow immediate human judgment of relevance). Quantitative offline evaluation was not the primary goal of this project; the focus was on building a functional, configurable, tested, and interactive recommendation pipeline.

Key findings:

- Content-based filtering performs well for songs with rich tag information and distinct audio profiles.
- Collaborative filtering surfaces surprising but relevant recommendations by leveraging listener co-occurrence patterns.
- The hybrid approach consistently outperforms either method in isolation when the diversity weight is balanced (0.4-0.6 content weight), reducing the echo-chamber effect of pure content similarity while retaining audio coherence.

---
---

## Author

**Soumik Chandra**