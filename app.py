import download_data
import streamlit as st
from notebooks.content_based_filtering import content_recommend
import pandas as pd
from scipy.sparse import load_npz
from notebooks.collaborative_filtering import collaborative_recommendation
from numpy import load
from notebooks.hybrid_recommendation import HybridRecommenderSystem as hrs

@st.cache_resource
def load_all_data():
    songs_data = pd.read_csv("data/cleaned_data.csv")
    transformed_data = load_npz("data/transformed_data.npz")
    track_ids = load("data/track_ids.npy", allow_pickle=True)
    filtered_data = pd.read_csv("data/collab_filtered_data.csv")
    interaction_matrix = load_npz("data/interaction_matrix.npz")
    transformed_hybrid_data = load_npz("data/transformed_hybrid_data.npz")

    for df in [songs_data, filtered_data]:
        for col in ["name", "artist"]:
            df[col] = df[col].astype(str).str.lower().str.strip()
        df.reset_index(drop=True,inplace=True)

    return songs_data, transformed_data, track_ids, filtered_data, interaction_matrix, transformed_hybrid_data

songs_data, transformed_data, track_ids, filtered_data, interaction_matrix, transformed_hybrid_data = load_all_data()

st.title("Welcome to Spotify Song Recommender!")
st.write("### Enter the name of the song and the recommender will suggest you similar songs")

song_input = st.text_input("Enter a song name:")
artist_input = st.text_input("Enter the artist name:")

song_name = song_input.lower().strip()
artist_name = artist_input.lower().strip()

k = st.selectbox("How many recommendations do you need?", [5, 10, 15, 20], index=1)
filtering_type=st.selectbox(
    label="Select the type of filtering",
    options=["Content Based Filtering","Collaborative Filtering","Hybrid Recommender System"],
    index=0
)

diversity = st.slider("Diversity in Recommendation", min_value=1, max_value=10, value=5, step=1)
content_based_weight = 1 - (diversity / 10)

if st.button("Get Recommendations"):

    if filtering_type == "Content Based Filtering":
        if (songs_data["name"] == song_name).any():
            st.write(f"Recommendations for: **{song_name}**")
            recommendations = content_recommend(
                song_name, artist_name, songs_data, transformed_data, k
            )
            for ind, rec in recommendations.iterrows():
                rec_song = rec["name"].title()
                rec_artist = rec["artist"].title()
                if ind == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{rec_song}** by **{rec_artist}**")
                else:
                    st.markdown(f"#### {ind}. **{rec_song}** by **{rec_artist}**")
                st.audio(rec["spotify_preview_url"])
                st.write("---")
        else:
            st.warning(f"Sorry, we couldn't find '{song_name}' in our database.")

    elif filtering_type == "Collaborative Filtering":
        if (
            (filtered_data["name"] == song_name) &
            (filtered_data["artist"] == artist_name)
        ).any():
            recommendations = collaborative_recommendation(
                song_name, artist_name, track_ids, songs_data, interaction_matrix, k
            )
            for ind, rec in recommendations.iterrows():
                rec_song = rec["name"].title()
                rec_artist = rec["artist"].title()
                if ind == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{rec_song}** by **{rec_artist}**")
                else:
                    st.markdown(f"#### {ind}. **{rec_song}** by **{rec_artist}**")
                st.audio(rec["spotify_preview_url"])
                st.write("---")
        else:
            st.warning(f"Sorry, we couldn't find '{song_name}' in our database.")

    elif filtering_type == "Hybrid Recommender System":
        if (
            (filtered_data["name"] == song_name) &
            (filtered_data["artist"] == artist_name)
        ).any():
            try:
                recommender = hrs(
                    number_of_recommendations=k,
                    weight_content_based=content_based_weight
                )
                recommendations = recommender.give_recommendations(
                    song_name=song_name,
                    artist_name=artist_name,
                    songs_data=filtered_data,
                    track_ids=track_ids,
                    transformed_matrix=transformed_hybrid_data,
                    interaction_matrix=interaction_matrix
                )
                for ind, rec in recommendations.iterrows():
                    rec_song = rec["name"].title()
                    rec_artist = rec["artist"].title()
                    if ind == 0:
                        st.markdown("## Currently Playing")
                        st.markdown(f"#### **{rec_song}** by **{rec_artist}**")
                    else:
                        st.markdown(f"#### {ind}. **{rec_song}** by **{rec_artist}**")
                    st.audio(rec["spotify_preview_url"])
                    st.write("---")
            except ValueError as e:
                st.warning(str(e))
            except Exception as e:
                st.error(f"Something went wrong: {e}")
        else:
            st.warning(f"'{song_name}' by '{artist_name}' not available for Hybrid Filtering.")