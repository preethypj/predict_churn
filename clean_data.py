import pandas as pd
from pathlib import Path

raw_data_path = Path("data/raw/telco_churn_dataset.csv")
processed_data_path = Path("data/processed/clean_telco_churn_dataset.csv")

# Loading the raw data
def load_data_raw(path: Path)->pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found in {raw_data_path}")
    return pd.read_csv(path)


# Cleaning raw data
def clean_raw_data(df: pd.DataFrame)-> pd.DataFrame:
    df = df.copy()

    # removing "customerID" column as it is an identifier and will not contribute to prediction
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # converting "TotalCharges" into numeric datatype
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # handling missing values by median
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


# saving the cleaned data into specfied directory
def save_clean_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df_raw = load_data_raw(raw_data_path)
    df_clean = clean_raw_data(df_raw)

    print("Cleaned dataset shape:", df_clean.shape)
    print("\nMissing values after cleaning:")
    print(df_clean.isnull().sum())

    save_clean_data(df_clean, processed_data_path)