import zipfile
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
zip_path = ROOT_DIR/ "archive (1).zip"       # Path to your zip file
extract_to = ROOT_DIR/ "custom-llm" / 'data/processed' # Folder where you want to extract

# Create destination folder if it doesn't exist
os.makedirs(extract_to, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f'Extracted all files to {extract_to}')
