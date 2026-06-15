import pandas as pd
import dask.dataframe as dd
from scipy.sparse import csr_matrix,save_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#Output Paths
track_ids_save_path="data/track_ids.npy"
filtered_data_save_path="data/collab_filtered_data.csv"
interaction_matrix_save_path="data/interaction_matrix.npz"

#Input Paths
songs_data_path="data/cleaned_data.csv"
user_listening_history_data_path="data/User Listening History.csv"

def filter_songs_data(songs_data:pd.DataFrame,track_ids:list,save_df_path:str)->pd.DataFrame:
    filtered_data=songs_data[songs_data["track_id"].isin(track_ids)]
    filtered_data.reset_index(drop=True,inplace=True)
    
    save_pandas_data_to_csv(filtered_data,save_df_path)
    return filtered_data

def save_pandas_data_to_csv(data:pd.DataFrame,file_path:str)->None:
    data.to_csv(file_path,index=False)
    
def save_sparse_matrix(matrix:csr_matrix,file_path:str)->None:
    save_npz(file_path,matrix)
    
def create_interaction_matrix(history_data, track_ids_save_path, save_matrix_path, filtered_data):
    df = history_data.copy()
    df['playcount'] = df['playcount'].astype(np.float64)
    
    # ✅ filtered_data ke ORDER se track_ids — alphabetical nahi
    ordered_track_ids = filtered_data["track_id"].values
    np.save(track_ids_save_path, ordered_track_ids, allow_pickle=True)
    
    # ✅ Manual mapping — filtered_data ke order mein
    track_id_to_idx = {tid: idx for idx, tid in enumerate(ordered_track_ids)}
    
    df = df.compute() if hasattr(df, 'compute') else df
    df = df[df["track_id"].isin(track_id_to_idx)]
    df['track_idx'] = df['track_id'].map(track_id_to_idx)
    
    user_categories = pd.Categorical(df['user_id'])
    df['user_idx'] = user_categories.codes
    n_users = len(user_categories.categories)
    
    interaction_agg = df.groupby(['track_idx', 'user_idx'])['playcount'].sum().reset_index()
    
    n_tracks = len(ordered_track_ids)
    matrix = csr_matrix(
        (interaction_agg['playcount'], 
         (interaction_agg['track_idx'], interaction_agg['user_idx'])),
        shape=(n_tracks, n_users)
    )
    save_sparse_matrix(matrix, save_matrix_path)
    return matrix
    
def collaborative_recommendation(song_name,artist_name,track_ids,songs_data,interaction_matrix,k=5):
    song_name=song_name.lower()
    artist_name=artist_name.lower()
    
    song_row=songs_data.loc[(songs_data["name"]==song_name) & (songs_data["artist"]==artist_name)]
    
    input_track_id=song_row['track_id'].values.item()
    ind=np.where(track_ids==input_track_id)[0].item()
    
    input_array=interaction_matrix[ind]
    
    
    similarity_scores=cosine_similarity(input_array,interaction_matrix)
    recommendation_indices=np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    recommendation_track_ids=track_ids[recommendation_indices]
    top_scores=np.sort(similarity_scores.ravel())[-k-1:][::-1]
    
    scores_df=pd.DataFrame({"track_id":recommendation_track_ids.tolist(),"score":top_scores})
    top_k_songs=(
        songs_data.loc[songs_data['track_id'].isin(recommendation_track_ids)]
        .merge(scores_df,on="track_id")
        .sort_values(by="score",ascending=False)
        .drop(columns=['track_id','score'])
        .reset_index(drop=True)
    )
    return top_k_songs

def main():
    user_data = dd.read_csv(user_listening_history_data_path)
    songs_data = pd.read_csv(songs_data_path)
    
    # ✅ Common tracks nikalo
    history_track_ids = user_data.loc[:, "track_id"].unique().compute().tolist()
    songs_track_ids = set(songs_data["track_id"].values)
    common_track_ids = [t for t in history_track_ids if t in songs_track_ids]
    
    # ✅ filtered_data banao
    filtered_data = filter_songs_data(songs_data, common_track_ids, filtered_data_save_path)
    
    # ✅ User history filter karo
    user_data_filtered = user_data[user_data["track_id"].isin(common_track_ids)]
    
    # ✅ Interaction matrix — filtered_data ke ORDER mein
    create_interaction_matrix(user_data_filtered, track_ids_save_path, interaction_matrix_save_path, filtered_data)

if __name__ == "__main__":
    main()