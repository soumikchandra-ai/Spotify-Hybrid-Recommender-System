# tests/test_content.py
import pytest
import numpy as np
import pandas as pd
from notebooks.content_based_filtering import (
    content_recommend,
    calculate_similarity_scores
)


class TestCalculateSimilarityScores:

    def test_output_shape_is_1_x_n(self, sample_transformed_matrix):
        input_vec = sample_transformed_matrix[0].reshape(1, -1)
        result = calculate_similarity_scores(input_vec, sample_transformed_matrix)
        assert result.shape == (1, len(sample_transformed_matrix))

    def test_self_similarity_is_highest(self, sample_transformed_matrix):
        """A song should be most similar to itself"""
        input_vec = sample_transformed_matrix[2].reshape(1, -1)
        result = calculate_similarity_scores(input_vec, sample_transformed_matrix)
        assert np.argmax(result) == 2

    def test_scores_between_minus1_and_1(self, sample_transformed_matrix):
        input_vec = sample_transformed_matrix[0].reshape(1, -1)
        result = calculate_similarity_scores(input_vec, sample_transformed_matrix)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)


class TestContentRecommend:

    def test_returns_k_results(self, sample_songs_data, sample_transformed_matrix):
        result = content_recommend(
            "song a", "artist1",
            sample_songs_data, sample_transformed_matrix, k=3
        )
        assert len(result) == 3

    def test_output_has_name_artist_url_columns(
        self, sample_songs_data, sample_transformed_matrix
    ):
        result = content_recommend(
            "song b", "artist2",
            sample_songs_data, sample_transformed_matrix, k=2
        )
        assert "name" in result.columns
        assert "artist" in result.columns
        assert "spotify_preview_url" in result.columns

    def test_song_not_in_dataset_raises_value_error(
        self, sample_songs_data, sample_transformed_matrix
    ):
        with pytest.raises(ValueError):
            content_recommend(
                "ghost song", "nobody",
                sample_songs_data, sample_transformed_matrix, k=3
            )

    def test_index_is_reset_in_output(
        self, sample_songs_data, sample_transformed_matrix
    ):
        result = content_recommend(
            "song c", "artist1",
            sample_songs_data, sample_transformed_matrix, k=2
        )
        assert list(result.index) == list(range(len(result)))

    def test_input_song_not_in_recommendations(
        self, sample_songs_data, sample_transformed_matrix
    ):
        """The queried song itself should not appear in results"""
        result = content_recommend(
            "song a", "artist1",
            sample_songs_data, sample_transformed_matrix, k=3
        )
        assert "song a" not in result["name"].values