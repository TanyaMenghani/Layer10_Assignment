import kagglehub
import os
import shutil

# Download dataset
path = kagglehub.dataset_download("wcukierski/enron-email-dataset")

print("Original dataset path:", path)

# Create local data directory
os.makedirs("data", exist_ok=True)

# Copy dataset into project data folder
source_file = os.path.join(path, "emails.csv")
destination_file = os.path.join("data", "emails.csv")

shutil.copy(source_file, destination_file)

print("Dataset copied to:", destination_file)