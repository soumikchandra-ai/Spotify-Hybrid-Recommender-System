import gdown
import os

os.makedirs("data", exist_ok=True)

files = {
    "data/interaction_matrix.npz":        "1Fck-nldK8IlwHC1PREN6ePCF1wfaRBCC",
    "data/transformed_data.npz":          "1Fox5HjoT3EWIIs2D95V79Kg2Bq362cZT",
    "data/transformed_hybrid_data.npz":   "1ZO6HdbMp5kkh16zLPylYo6jEKacqLz2C",
    "data/track_ids.npy":                 "1Lf4Jtu9UBrKMHmqJ-sE94h7roJOQzwB3",
    "data/cleaned_data.csv":              "1JZL0SYBQFmxZM0FF19JA4nafW1usOJ9R",
    "data/collab_filtered_data.csv":      "1jjlDnFu4VaP-RP5nt6LksWOd1-RC8s47",
    "transformer.joblib":                 "1ZQRzKfUNGwvFucXz2I5xyJubseuLY6-P",
    "data/Music Info.csv":                "1MEo8ZOhKkZ4qLm75a2xROlj2tM0GpD7E",
    "data/User Listening History.csv":    "12rUk7c8sj2xZAmCkudSHQzYm3JDgfU81"
}

for local_path, file_id in files.items():
    if not os.path.exists(local_path):
        print(f"Downloading {local_path}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", local_path, quiet=False)
    else:
        print(f"Already exists: {local_path}")