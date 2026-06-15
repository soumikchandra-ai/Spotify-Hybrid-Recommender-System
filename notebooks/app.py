import streamlit as st
from content_based_filtering import recommend
import pandas as pd
from scipy.sparse import load_npz

transformed_data_path="data/transformed_data.npz"

cleaned_data_path="data/cleaned_data.csv"

data=pd.read_csv(cleaned_data_path)

transformed_data=load_npz(transformed_data_path)

st.title("Welcome to Spotify Song Recommender!")

st.write('### Enter the name of the song and the recommender will suggest you similar songs')

song_name=st.text_input('Enter a song name:')
st.write('You Entered: ',song_name)

song_name=song_name.lower()

k=st.selectbox('How many recommendations do you need?',[5,10,15,20],index=1)

if st.button("Get Recommendations:"):
    if(data["name"]==song_name).any():
        st.write("Recommendations for: "f"**{song_name}**")
        recommendations=recommend(song_name,data,transformed_data,k)
        
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