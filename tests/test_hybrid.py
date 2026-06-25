import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from notebooks.hybrid_recommendation import HybridRecommenderSystem


class TestNormalizeSimilarities:
    """Tests for the __normalize_similarities method"""

    def setup_method(self):
        self.recommender = HybridRecommenderSystem(
            number_of_recommendations=3,
            weight_content_based=0.6
        )
        # Access private method via name mangling
        self.normalize = self.recommender._HybridRecommenderSystem__normalize_similarities

    def test_output_range_between_0_and_1(self):
        scores = np.array([[0.2, 0.8, 0.5, 0.1, 0.9]])
        result = self.normalize(scores)
        assert np.all(result >= 0.0), "Normalized scores should be >= 0"
        assert np.all(result <= 1.0), "Normalized scores should be <= 1"

    def test_minimum_becomes_zero(self):
        scores = np.array([[3.0, 5.0, 7.0]])
        result = self.normalize(scores)
        assert result.min() == pytest.approx(0.0)

    def test_maximum_becomes_one(self):
        scores = np.array([[3.0, 5.0, 7.0]])
        result = self.normalize(scores)
        assert result.max() == pytest.approx(1.0)

    def test_uniform_scores_edge_case(self):
        """All same values — division by zero risk"""
        scores = np.array([[0.5, 0.5, 0.5]])
        # Should not raise ZeroDivisionError
        try:
            result = self.normalize(scores)
        except Exception as e:
            pytest.fail(f"Normalize raised unexpected exception: {e}")

    def test_output_shape_preserved(self):
        scores = np.array([[0.1, 0.4, 0.9, 0.3]])
        result = self.normalize(scores)
        assert result.shape == scores.shape


class TestWeightedCombination:
    """Tests for the __weighted_combination method"""

    def setup_method(self):
        self.recommender = HybridRecommenderSystem(
            number_of_recommendations=3,
            weight_content_based=0.6
        )
        self.combine = self.recommender._HybridRecommenderSystem__weighted_combination

    def test_weights_sum_to_one(self):
        """If CBF=1 and CF=0, output should equal CBF scores"""
        r = HybridRecommenderSystem(3, weight_content_based=1.0)
        combine = r._HybridRecommenderSystem__weighted_combination
        cbf = np.array([[0.2, 0.8, 0.5]])
        cf  = np.array([[0.0, 0.0, 0.0]])
        result = combine(cbf, cf)
        np.testing.assert_array_almost_equal(result, cbf)

    def test_equal_weights(self):
        """With 0.5/0.5 weights, result should be simple average"""
        r = HybridRecommenderSystem(3, weight_content_based=0.5)
        combine = r._HybridRecommenderSystem__weighted_combination
        cbf = np.array([[0.4, 0.6]])
        cf  = np.array([[0.2, 0.8]])
        expected = np.array([[0.3, 0.7]])
        result = combine(cbf, cf)
        np.testing.assert_array_almost_equal(result, expected)

    def test_output_shape_matches_input(self):
        cbf = np.array([[0.1, 0.5, 0.9]])
        cf  = np.array([[0.3, 0.4, 0.6]])
        result = self.combine(cbf, cf)
        assert result.shape == cbf.shape


class TestGiveRecommendations:
    """Integration tests for the full give_recommendations pipeline"""

    def test_returns_correct_number_of_recommendations(
        self, sample_songs_data, sample_track_ids,
        sample_transformed_matrix, sample_interaction_matrix
    ):
        recommender = HybridRecommenderSystem(
            number_of_recommendations=3,
            weight_content_based=0.6
        )
        result = recommender.give_recommendations(
            song_name="song a",
            artist_name="artist1",
            songs_data=sample_songs_data,
            track_ids=sample_track_ids,
            transformed_matrix=sample_transformed_matrix,
            interaction_matrix=sample_interaction_matrix
        )
        # Returns top-k+1 (includes input song), so at most k+1 rows
        assert len(result) <= 4

    def test_returns_dataframe(
        self, sample_songs_data, sample_track_ids,
        sample_transformed_matrix, sample_interaction_matrix
    ):
        recommender = HybridRecommenderSystem(3, 0.6)
        result = recommender.give_recommendations(
            "song a", "artist1",
            sample_songs_data, sample_track_ids,
            sample_transformed_matrix, sample_interaction_matrix
        )
        assert isinstance(result, pd.DataFrame)

    def test_invalid_song_raises_value_error(
        self, sample_songs_data, sample_track_ids,
        sample_transformed_matrix, sample_interaction_matrix
    ):
        recommender = HybridRecommenderSystem(3, 0.6)
        with pytest.raises(ValueError, match="not found"):
            recommender.give_recommendations(
                "nonexistent song", "unknown artist",
                sample_songs_data, sample_track_ids,
                sample_transformed_matrix, sample_interaction_matrix
            )

    def test_result_does_not_contain_track_id_column(
        self, sample_songs_data, sample_track_ids,
        sample_transformed_matrix, sample_interaction_matrix
    ):
        """track_id and score columns should be dropped in final output"""
        recommender = HybridRecommenderSystem(3, 0.6)
        result = recommender.give_recommendations(
            "song a", "artist1",
            sample_songs_data, sample_track_ids,
            sample_transformed_matrix, sample_interaction_matrix
        )
        assert "track_id" not in result.columns
        assert "score" not in result.columns

    def test_weight_content_based_0_uses_only_collaborative(
        self, sample_songs_data, sample_track_ids,
        sample_transformed_matrix, sample_interaction_matrix
    ):
        """Pure collaborative mode should still return valid results"""
        recommender = HybridRecommenderSystem(3, weight_content_based=0.0)
        result = recommender.give_recommendations(
            "song b", "artist2",
            sample_songs_data, sample_track_ids,
            sample_transformed_matrix, sample_interaction_matrix
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0