import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler,StandardScaler,OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from notebooks.data_cleaning import data_for_content_filtering
from scipy.sparse import save_npz

#Cleaned Data Path Specified
CLEANED_DATA_PATH="data/cleaned_data.csv"

#Columns to transform
frequency_encode_cols=['year']
ohe_cols=['artist','time_signature','key']
tfidf_cols='tags'
standard_scale_cols=['duration_ms','loudness','tempo']
min_max_scale_cols=['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence']

def train_transformer(data):
    transformer=ColumnTransformer(transformers=[
        ("frequency_encode",CountEncoder(normalize=True,return_df=True),frequency_encode_cols),
        ("ohe",OneHotEncoder(handle_unknown="ignore"),ohe_cols),
        ("tfidf",TfidfVectorizer(max_features=85),'tags'),
        ("standard_scale",StandardScaler(),standard_scale_cols),
        ("min_max_scale",MinMaxScaler(),min_max_scale_cols)
    ],remainder='drop',n_jobs=-1)

    transformer.fit(data)

    #Save the transformer
    joblib.dump(transformer,"transformer.joblib")

def transform_data(data):
    #Load the transformer
    transformer=joblib.load("transformer.joblib")

    #Transform the Data
    transformed_data=transformer.transform(data)

    return transformed_data

def save_transformed_data(transformed_data,save_path):
    #Save the transformed data
    save_npz(save_path,transformed_data)

def calculate_similarity_scores(input_vector,data):
    similarity_scores=cosine_similarity(input_vector,data)
    return similarity_scores

def content_recommend(song_name,artist_name,songs_data,transformed_data,k=10):

    song_name=song_name.lower()
    artist_name=artist_name.lower()

    song_row=songs_data.loc[songs_data["name"].str.lower()==song_name]

    if song_row.empty:
        raise ValueError(f"{song_name} not found in dataset")

    song_index=song_row.index[0]

    # sparse matrix row already has shape (1,n_features)
    input_vector=transformed_data[song_index]

    similarity_scores=calculate_similarity_scores(input_vector,transformed_data)

    top_k_songs_indexes=np.argsort(similarity_scores.ravel())[::-1]

    # remove queried song
    top_k_songs_indexes=top_k_songs_indexes[top_k_songs_indexes!=song_index]

    top_k_songs_indexes=top_k_songs_indexes[:k]

    top_k_songs_names=songs_data.iloc[top_k_songs_indexes]

    top_k_list=top_k_songs_names[['name','artist','spotify_preview_url']].reset_index(drop=True)

    return top_k_list

def test_recommendations(data_path,song_name,k=10):

    song_name=song_name.lower()

    data=pd.read_csv(data_path)

    data_content_filtering=data_for_content_filtering(data)

    train_transformer(data_content_filtering)

    transformed_data=transform_data(data_content_filtering)

    save_transformed_data(transformed_data,"data/transformed_data.npz")

    song_row=data.loc[data["name"].str.lower()==song_name]

    if song_row.empty:
        raise ValueError(f"{song_name} not found in dataset")

    print(song_row)

    song_index=song_row.index[0]

    input_vector=transformed_data[song_index]

    similarity_scores=calculate_similarity_scores(input_vector,transformed_data)

    top_k_songs_indexes=np.argsort(similarity_scores.ravel())[::-1]

    top_k_songs_indexes=top_k_songs_indexes[top_k_songs_indexes!=song_index]

    top_k_songs_indexes=top_k_songs_indexes[:k]

    top_k_songs=data.iloc[top_k_songs_indexes]

    print(top_k_songs)

if __name__=="__main__":
    test_recommendations(CLEANED_DATA_PATH,"Somebody Told Me")