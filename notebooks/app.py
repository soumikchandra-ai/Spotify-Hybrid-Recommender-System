import streamlit as st
from content_based_filtering import content_recommend
import pandas as pd
from scipy.sparse import load_npz
from collaborative_filtering import collaborative_recommendation
from numpy import load

#Clean the data
cleaned_data_path="data/cleaned_data.csv"
songs_data=pd.read_csv(cleaned_data_path)

#Load the transformed data
transformed_data_path="data/transformed_data.npz"
transformed_data=load_npz(transformed_data_path)

#Load the track ids
track_ids_path="data/track_ids.npy"
track_ids=load(track_ids_path,allow_pickle=True)

#Load the filtered songs data
filtered_data_path="data/collab_filtered_data.csv"
filtered_data=pd.read_csv(filtered_data_path)

#Load the interactionn matrix
interaction_matrix_path="data/interaction_matrix.npz"
interaction_matrix=load_npz(interaction_matrix_path)

#Heading
st.title("Welcome to Spotify Song Recommender!")
#Sub heading
st.write('### Enter the name of the song and the recommender will suggest you similar songs')

#Song Input
song_name=st.text_input('Enter a song name:')
st.write('You Entered: ',song_name)

#Singer input
artist_name=st.text_input('Enter the artist name:')
st.write('You entered: ',artist_name)

song_name=song_name.lower()
artist_name=artist_name.lower()

#k Recommendations
k=st.selectbox('How many recommendations do you need?',[5,10,15,20],index=1)

#Filtering Type
filtering_type=st.selectbox('Select the type of Filetring:',['Content Based Filtering','Collaborative Filtering'])

if filtering_type=='Content Based Filtering':
    if st.button("Get Recommendations:"):
        if(songs_data["name"]==song_name).any():
            st.write("Recommendations for: "f"**{song_name}**")
            recommendations=content_recommend(song_name,artist_name,songs_data,transformed_data,k)
            
            for ind,recommendation in recommendations.iterrows():
                song_name=recommendation['name'].title()
                artist_name=recommendation['artist'].title()
                
                if ind==0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                    
                elif ind==1:
                    st.markdown("### Next Up")
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write("---")
                    
                else:
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write("---")
                    
        else:
            st.write(f"Sorry , we could't find {song_name} in our Databse. Please try entering another Song")
            
elif filtering_type=="Collaborative Filtering":
    if st.button("Get Recommendations:"):
        if((filtered_data["name"]==song_name) & (filtered_data["artist"]==artist_name)).any():
            st.write("Recommendations for: "f"**{song_name}** by **{artist_name}**")
            recommendations=collaborative_recommendation(song_name,artist_name,track_ids,songs_data,interaction_matrix,k)
            
            for ind,recommendation in recommendations.iterrows():
                song_name=recommendation['name'].title()
                artist_name=recommendation['artist'].title()
                
                if ind==0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                    
                elif ind==1:
                    st.markdown("### Next Up")
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write("---")
                    
                else:
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write("---")
                    
        else:
            st.write(f"Sorry , we could't find {song_name} in our Databse. Please try entering another Song")