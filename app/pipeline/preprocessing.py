import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    def __init__(self, target_column: str):
        self.target_column = target_column
        self.encoders = {}

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values:
        - numerical → median
        - categorical → mode
        """
        df = df.copy()

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical columns using Label Encoding
        """
        df = df.copy()

        for col in df.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le

        return df

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2):
        """
        Split dataset into train and test sets
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        return train_test_split(X, y, test_size=test_size, random_state=42)