import pandas as pd
from pathlib import Path

data_path = Path("data/raw/telco_churn_dataset.csv")

def load_data(path: Path) -> pd.DataFrame:
    # loading dataset from csv file
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
    
    df = pd.read_csv(path)
    return df

def validation(df: pd.DataFrame) -> None:
    print("Dataset shape: ", df.shape)
    
    print("\nColumn Names:")
    print(df.columns.to_list())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found in dataset")
    
if __name__ == "__main__":
    df = load_data(data_path)
    validation(df)