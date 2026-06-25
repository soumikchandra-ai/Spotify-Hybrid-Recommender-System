import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from notebooks.collaborative_filtering import (
    collaborative_recommendation,
    filter_songs_data,
    save_pandas_data_to_csv
)


class TestFilterSongsData:

    def test_only_common_track_ids_remain(self, sample_songs_data, tmp_path):
        keep_ids = ["t1", "t3"]
        save_path = str(tmp_path / "filtered.csv")
        result = filter_songs_data(sample_songs_data, keep_ids, save_path)
        assert set(result["track_id"].tolist()) == {"t1", "t3"}

    def test_output_index_is_reset(self, sample_songs_data, tmp_path):
        save_path = str(tmp_path / "filtered.csv")
        result = filter_songs_data(sample_songs_data, ["t2", "t4"], save_path)
        assert list(result.index) == list(range(len(result)))

    def test_empty_track_ids_returns_empty_df(self, sample_songs_data, tmp_path):
        save_path = str(tmp_path / "filtered.csv")
        result = filter_songs_data(sample_songs_data, [], save_path)
        assert result.empty

    def test_file_is_saved(self, sample_songs_data, tmp_path):
        save_path = str(tmp_path / "out.csv")
        filter_songs_data(sample_songs_data, ["t1"], save_path)
        assert (tmp_path / "out.csv").exists()


class TestCollaborativeRecommendation:

    def test_returns_k_plus_one_rows(
        self, sample_songs_data, sample_track_ids, sample_interaction_matrix
    ):
        result = collaborative_recommendation(
            "song a", "artist1",
            sample_track_ids, sample_songs_data,
            sample_interaction_matrix, k=3
        )
        assert len(result) <= 4  # k+1 includes input song

    def test_output_is_dataframe(
        self, sample_songs_data, sample_track_ids, sample_interaction_matrix
    ):
        result = collaborative_recommendation(
            "song b", "artist2",
            sample_track_ids, sample_songs_data,
            sample_interaction_matrix, k=2
        )
        assert isinstance(result, pd.DataFrame)

    def test_score_column_dropped_in_output(
        self, sample_songs_data, sample_track_ids, sample_interaction_matrix
    ):
        result = collaborative_recommendation(
            "song a", "artist1",
            sample_track_ids, sample_songs_data,
            sample_interaction_matrix, k=3
        )
        assert "score" not in result.columns

    def test_name_and_artist_columns_present(
        self, sample_songs_data, sample_track_ids, sample_interaction_matrix
    ):
        result = collaborative_recommendation(
            "song c", "artist1",
            sample_track_ids, sample_songs_data,
            sample_interaction_matrix, k=2
        )
        assert "name" in result.columns
        assert "artist" in result.columns