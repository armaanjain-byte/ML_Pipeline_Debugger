import pandas as pd


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Loads dataset from a CSV file.

        Returns:
            pd.DataFrame: Loaded dataset

        Raises:
            ValueError: If file cannot be read or is empty
        """
        try:
            df = pd.read_csv(self.file_path)
        except Exception as e:
            raise ValueError(f"Error loading file: {e}")

        if df.empty:
            raise ValueError("Loaded dataset is empty")

        return df

    def basic_info(self, df: pd.DataFrame) -> dict:
        """
        Returns basic information about dataset
        """
        return {
            "num_rows": df.shape[0],
            "num_columns": df.shape[1],
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
"""loads a dataset into a standardized in-memory structure (DataFrame), validates it,
and generates structured metadata that can be programmatically consumed by other components"""