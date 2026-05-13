"""
FIXED: app/debugger/data_checks.py

CHANGES:
- Comprehensive issue detection
- Structured output
- Better correlation detection
- Outlier detection
- Duplicate detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


class DataChecks:
    """
    Comprehensive data validation.
    Runs BEFORE preprocessing to detect issues.
    """
    
    def run_all_checks(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Run all checks and return structured output.
        
        Returns:
            dict with keys: missing_values, constant_features, duplicates,
                           class_imbalance, high_correlation, data_types, issues
        """
        return {
            "missing_values": self.check_missing_values(df),
            "constant_features": self.check_constant_features(df),
            "duplicates": self.check_duplicates(df),
            "class_imbalance": self.check_class_imbalance(df, target_column),
            "high_correlation": self.check_high_correlation(df),
            "data_types": self.check_data_types(df),
            "issues": self._summarize_issues(df, target_column)
        }
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Return percentage of missing values per column.
        
        Returns:
            dict: {column_name: missing_percentage}
        """
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        # Only return columns with missing values
        return {k: v for k, v in missing_pct.items() if v > 0}
    
    def check_constant_features(self, df: pd.DataFrame) -> List[str]:
        """
        Detect columns with only one unique value (zero variance).
        
        Returns:
            list: Column names with constant values
        """
        return [col for col in df.columns if df[col].nunique() <= 1]
    
    def check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for duplicate rows.
        
        Returns:
            dict: Total duplicates count and percentage
        """
        num_duplicates = int(df.duplicated().sum())
        dup_pct = float(num_duplicates / len(df) * 100)
        
        return {
            "total_duplicates": num_duplicates,
            "duplicate_percentage": dup_pct,
            "has_duplicates": num_duplicates > 0
        }
    
    def check_class_imbalance(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Check target class distribution (classification only).
        
        Returns:
            dict: Class distribution, minority class %, imbalance status
        """
        if target_column not in df.columns:
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
    
    def check_high_correlation(
        self,
        df: pd.DataFrame,
        threshold: float = 0.9
    ) -> List[tuple]:
        """
        Detect highly correlated numeric columns.
        
        Args:
            df: DataFrame
            threshold: Correlation threshold (0-1)
        
        Returns:
            list: [(col1, col2, correlation_value), ...]
        """
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
    
    def check_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Report data types of all columns.
        
        Returns:
            dict: {column_name: data_type_string}
        """
        return df.dtypes.astype(str).to_dict()
    
    def check_outliers(self, df: pd.DataFrame, iqr_multiplier: float = 1.5) -> Dict[str, int]:
        """
        Detect outliers using IQR method (numeric columns only).
        
        Args:
            df: DataFrame
            iqr_multiplier: Multiplier for IQR (1.5 = standard)
        
        Returns:
            dict: {column_name: outlier_count}
        """
        numeric_df = df.select_dtypes(include=[np.number])
        outliers = {}
        
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - (iqr_multiplier * IQR)
            upper_bound = Q3 + (iqr_multiplier * IQR)
            
            outlier_count = ((numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)).sum()
            
            if outlier_count > 0:
                outliers[col] = int(outlier_count)
        
        return outliers
    
    def _summarize_issues(self, df: pd.DataFrame, target_column: str) -> List[Dict[str, Any]]:
        """
        Summarize all detected issues with severity levels.
        
        Returns:
            list: Issue dictionaries with type, column, severity, description
        """
        issues = []
        
        # Missing values
        for col, pct in self.check_missing_values(df).items():
            severity = "high" if pct > 50 else "medium" if pct > 20 else "low"
            issues.append({
                "type": "missing_values",
                "column": col,
                "severity": severity,
                "description": f"{pct:.1f}% missing values"
            })
        
        # Constant features
        for col in self.check_constant_features(df):
            issues.append({
                "type": "constant_feature",
                "column": col,
                "severity": "high",
                "description": "Feature has only one unique value"
            })
        
        # Duplicates
        dup_info = self.check_duplicates(df)
        if dup_info["has_duplicates"]:
            issues.append({
                "type": "duplicate_rows",
                "column": None,
                "severity": "medium",
                "description": f"{dup_info['total_duplicates']} duplicate rows ({dup_info['duplicate_percentage']:.1f}%)"
            })
        
        # Class imbalance
        imbalance = self.check_class_imbalance(df, target_column)
        if imbalance.get("is_imbalanced"):
            issues.append({
                "type": "class_imbalance",
                "column": target_column,
                "severity": "high",
                "description": f"Minority class: {imbalance['minority_class_percentage']:.1f}%"
            })
        
        # High correlation
        for col1, col2, corr in self.check_high_correlation(df):
            issues.append({
                "type": "high_correlation",
                "column": f"{col1} ↔ {col2}",
                "severity": "medium",
                "description": f"Correlation: {corr:.3f}"
            })
        
        # Outliers
        for col, count in self.check_outliers(df).items():
            pct = (count / len(df)) * 100
            severity = "high" if pct > 10 else "medium" if pct > 5 else "low"
            issues.append({
                "type": "outliers",
                "column": col,
                "severity": severity,
                "description": f"{count} outliers ({pct:.1f}%)"
            })
        
        return issues