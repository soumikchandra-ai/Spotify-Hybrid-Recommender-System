import gdown
import os

os.makedirs("data", exist_ok=True)

files = {
    "data/interaction_matrix.npz":        "https://drive.google.com/file/d/1Fck-nldK8IlwHC1PREN6ePCF1wfaRBCC/view?usp=drive_link",
    "data/transformed_data.npz":          "https://drive.google.com/file/d/1Fox5HjoT3EWIIs2D95V79Kg2Bq362cZT/view?usp=drive_link",
    "data/transformed_hybrid_data.npz":   "https://drive.google.com/file/d/1ZO6HdbMp5kkh16zLPylYo6jEKacqLz2C/view?usp=drive_link",
    "data/track_ids.npy":                 "https://drive.google.com/file/d/1Lf4Jtu9UBrKMHmqJ-sE94h7roJOQzwB3/view?usp=drive_link",
    "data/cleaned_data.csv":              "https://drive.google.com/file/d/1JZL0SYBQFmxZM0FF19JA4nafW1usOJ9R/view?usp=drive_link",
    "data/collab_filtered_data.csv":      "https://drive.google.com/file/d/1jjlDnFu4VaP-RP5nt6LksWOd1-RC8s47/view?usp=drive_link",
    "transformer.joblib":                 "https://drive.google.com/file/d/1ZQRzKfUNGwvFucXz2I5xyJubseuLY6-P/view?usp=drive_link",
    "data/Music Info.csv":                "https://drive.google.com/file/d/1MEo8ZOhKkZ4qLm75a2xROlj2tM0GpD7E/view?usp=drive_link",
    "data/User Listening History.csv":    "https://drive.google.com/file/d/12rUk7c8sj2xZAmCkudSHQzYm3JDgfU81/view?usp=drive_link"
}

for local_path, file_id in files.items():
    if not os.path.exists(local_path):
        print(f"Downloading {local_path}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", local_path, quiet=False)
    else:
        print(f"Already exists: {local_path}")