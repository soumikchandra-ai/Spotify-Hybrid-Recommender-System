import pandas as pd
from notebooks.data_cleaning import data_for_content_filtering
from notebooks.content_based_filtering import transform_data,save_transformed_data
#Path of filtered Data
filtered_data_path="data/collab_filtered_data.csv"
#Save Path
save_path="data/transformed_hybrid_data.npz"

def main(data_path,save_path):
    filtered_data=pd.read_csv(data_path)
    filtered_data_cleaned=data_for_content_filtering(filtered_data)
    transformed_data=transform_data(filtered_data_cleaned)
    save_transformed_data(transformed_data,save_path)
    
if __name__=="__main__":
    main(filtered_data_path,save_path)