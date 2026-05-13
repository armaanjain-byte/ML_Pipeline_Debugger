import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.ensemble import IsolationForest


class DataChecks:
    """
    Comprehensive data validation.
    Runs BEFORE preprocessing to detect issues.
    """
    
    def run_all_checks(self, df: pd.DataFrame, target_column: str, task_type: str = "classification") -> Dict[str, Any]:
        """
        Run all checks and return structured output.
        """
        return {
            "missing_values": self.check_missing_values(df),
            "constant_features": self.check_constant_features(df),
            "duplicates": self.check_duplicates(df),
            "class_imbalance": self.check_class_imbalance(df, target_column, task_type),
            "high_correlation": self.check_high_correlation(df),
            "target_leakage": self.check_target_leakage(df, target_column),
            "multivariate_outliers": self.check_multivariate_outliers(df),
            "data_types": self.check_data_types(df),
            "issues": self._summarize_issues(df, target_column, task_type)
        }
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        return {k: v for k, v in missing_pct.items() if v > 0}
    
    def check_constant_features(self, df: pd.DataFrame) -> List[str]:
        return [col for col in df.columns if df[col].nunique() <= 1]
    
    def check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        num_duplicates = int(df.duplicated().sum())
        dup_pct = float(num_duplicates / len(df) * 100)
        
        return {
            "total_duplicates": num_duplicates,
            "duplicate_percentage": dup_pct,
            "has_duplicates": num_duplicates > 0
        }
    
    def check_class_imbalance(self, df: pd.DataFrame, target_column: str, task_type: str) -> Dict[str, Any]:
        # Do not check class imbalance for continuous regression targets
        if target_column not in df.columns or task_type == "regression":
            return {}
        
        value_counts = df[target_column].value_counts(normalize=True).to_dict()
        if len(value_counts) == 0:
            return {}
        
        minority_pct = min(value_counts.values()) * 100
        return {
            "distribution": value_counts,
            "minority_class_percentage": float(minority_pct),
            "is_imbalanced": minority_pct < 20,
            "num_classes": len(value_counts)
        }
    
    def check_high_correlation(self, df: pd.DataFrame, threshold: float = 0.9) -> List[tuple]:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return []
        
        corr_matrix = numeric_df.corr().abs()
        high_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        float(corr_matrix.iloc[i, j])
                    ))
        
        return high_corr

    def check_target_leakage(self, df: pd.DataFrame, target_column: str, threshold: float = 0.95) -> List[Dict[str, Any]]:
        """
        Detects features that have suspiciously high correlation with the target.
        """
        leakage_warnings = []
        if target_column not in df.columns:
            return leakage_warnings
            
        numeric_df = df.select_dtypes(include=[np.number])
        if target_column in numeric_df.columns:
            correlations = numeric_df.corr()[target_column].abs()
            
            for col, corr in correlations.items():
                if col != target_column and corr > threshold:
                    leakage_warnings.append({
                        "column": col,
                        "correlation": float(corr)
                    })
                    
        return leakage_warnings

    def check_multivariate_outliers(self, df: pd.DataFrame, contamination: float = 0.05) -> Dict[str, Any]:
        """
        Detects multivariate anomalies using Isolation Forest.
        """
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if numeric_df.empty or len(numeric_df) < 50:
            return {"count": 0, "percentage": 0.0, "has_outliers": False}
        
        iso = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso.fit_predict(numeric_df)
        
        outlier_count = int