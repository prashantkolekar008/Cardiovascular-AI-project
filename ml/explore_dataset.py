import pandas as pd
from pathlib import Path

# Project aur dataset folder ka path
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset"

# Excel file find karo
excel_files = list(dataset_folder.glob("*.xlsx"))

if not excel_files:
    print("Excel file nahi mili!")
else:
    df = pd.read_excel(excel_files[0])

    print("=== TARGET COLUMN CHECK ===")

    for column in ["Cath", "LAD", "LCX", "RCA"]:
        print(f"\n{column} ke values:")
        print(df[column].value_counts(dropna=False))

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    print("\n=== DUPLICATE ROWS ===")
    print(df.duplicated().sum())