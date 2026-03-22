import pandas as pd


class DataChecks:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column

    def check_missing_values(self):
        """
        Returns columns with missing values
        """
        missing = self.df.isnull().sum()
        return missing[missing > 0].to_dict()

    def check_class_imbalance(self):
        """
        Check imbalance in target column (classification only)
        """
        if self.target_column not in self.df.columns:
            return None

        value_counts = self.df[self.target_column].value_counts(normalize=True)
        return value_counts.to_dict()

    def check_constant_features(self):
        """
        Detect columns with only one unique value
        """
        constant_cols = [
            col for col in self.df.columns
            if self.df[col].nunique() == 1
        ]
        return constant_cols

    def check_high_correlation(self, threshold: float = 0.9):
        """
        Detect highly correlated features
        """
        corr_matrix = self.df.corr(numeric_only=True).abs()

        high_corr = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > threshold:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    high_corr.append((col1, col2))

        return high_corr