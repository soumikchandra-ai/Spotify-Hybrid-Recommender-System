# tests/conftest.py
import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

@pytest.fixture
def sample_songs_data():
    """Minimal songs DataFrame mimicking cleaned_data.csv"""
    return pd.DataFrame({
        "name": ["song a", "song b", "song c", "song d", "song e"],
        "artist": ["artist1", "artist2", "artist1", "artist3", "artist2"],
        "track_id": ["t1", "t2", "t3", "t4", "t5"],
        "spotify_preview_url": ["url1", "url2", "url3", "url4", "url5"]
    })

@pytest.fixture
def sample_track_ids():
    return np.array(["t1", "t2", "t3", "t4", "t5"])

@pytest.fixture
def sample_transformed_matrix():
    """5 songs x 8 features — dense numpy array"""
    np.random.seed(42)
    return np.random.rand(5, 8)

@pytest.fixture
def sample_interaction_matrix():
    """5 songs x 10 users — sparse matrix"""
    np.random.seed(42)
    dense = np.random.rand(5, 10)
    return csr_matrix(dense)

@pytest.fixture
def hybrid_recommender():
    from notebooks.hybrid_recommendation import HybridRecommenderSystem
    return HybridRecommenderSystem(
        number_of_recommendations=3,
        weight_content_based=0.6
    )