import pandas as pd
from pathlib import Path

# Dataset file ka path
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset"

# Excel file automatically find karna
excel_files = list(dataset_folder.glob("*.xlsx"))

if not excel_files:
    print("Dataset file nahi mili! Check the dataset folder.")
else:
    file_path = excel_files[0]
    df = pd.read_excel(file_path)

    print("Dataset successfully loaded!")
    print("Dataset shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())