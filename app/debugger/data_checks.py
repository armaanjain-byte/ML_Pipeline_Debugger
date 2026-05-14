import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)
class DataChecks:
    """
    Comprehensive data validation.
    Runs BEFORE preprocessing to detect issues.
    """
    
    def run_all_checks(self, df: pd.DataFrame, target_column: str, task_type: str = "classification") -> Dict[str, Any]:
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
        return {
            "total_duplicates": num_duplicates,
            "duplicate_percentage": float(num_duplicates / len(df) * 100) if len(df) > 0 else 0.0,
            "has_duplicates": num_duplicates > 0
        }
    
    def check_class_imbalance(self, df: pd.DataFrame, target_column: str, task_type: str) -> Dict[str, Any]:
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

    # app/debugger/data_checks.py

def check_multivariate_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.empty or len(numeric_df) < 50:
         return {"count": 0, "percentage": 0.0, "has_outliers": False, "outlier_indices": []}
    
    # 'auto' allows the model to decide if anomalies actually exist
    iso = IsolationForest(contamination="auto", random_state=42)
    predictions = iso.fit_predict(numeric_df)
    
    outlier_indices = numeric_df.index[predictions == -1].tolist()
    outlier_count = len(outlier_indices)
    
    return {
        "count": outlier_count,
        "percentage": float((outlier_count / len(numeric_df)) * 100),
        "has_outliers": outlier_count > 0,
        "outlier_indices": outlier_indices[:10] # Return top 10 for inspection
    }
    def check_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        return df.dtypes.astype(str).to_dict()
    
    def check_outliers(self, df: pd.DataFrame, target_column: str, iqr_multiplier: float = 1.5) -> Dict[str, int]:
        numeric_df = df.select_dtypes(include=[np.number])
        outliers = {}
        for col in numeric_df.columns:
            # THE FIX: Skip target column AND any binary columns (like SeniorCitizen)
            if col == target_column or numeric_df[col].nunique() <= 2:
                continue
                
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (iqr_multiplier * IQR)
            upper_bound = Q3 + (iqr_multiplier * IQR)
            
            outlier_count = ((numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)).sum()
            if outlier_count > 0:
                outliers[col] = int(outlier_count)
        return outliers
    
    def _summarize_issues(self, df: pd.DataFrame, target_column: str, task_type: str) -> List[Dict[str, Any]]:
        issues = []
        
        for col, pct in self.check_missing_values(df).items():
            severity = "high" if pct > 50 else "medium" if pct > 20 else "low"
            issues.append({
                "type": "missing_values",
                "column": col,
                "severity": severity,
                "description": f"{pct:.1f}% missing values"
            })
        
        for col in self.check_constant_features(df):
            issues.append({
                "type": "constant_feature",
                "column": col,
                "severity": "high",
                "description": "Feature has only one unique value"
            })
        
        dup_info = self.check_duplicates(df)
        if dup_info and dup_info.get("has_duplicates"):
            issues.append({
                "type": "duplicate_rows",
                "column": "dataset_wide",
                "severity": "medium",
                "description": f"{dup_info['total_duplicates']} duplicate rows ({dup_info['duplicate_percentage']:.1f}%)"
            })
        
        imbalance = self.check_class_imbalance(df, target_column, task_type)
        if imbalance and imbalance.get("is_imbalanced"):
            issues.append({
                "type": "class_imbalance",
                "column": target_column,
                "severity": "high",
                "description": f"Minority class: {imbalance['minority_class_percentage']:.1f}%"
            })
        
        for col1, col2, corr in self.check_high_correlation(df):
            issues.append({
                "type": "high_correlation",
                "column": f"{col1} ↔ {col2}",
                "severity": "medium",
                "description": f"Correlation: {corr:.3f}"
            })

        for leak in self.check_target_leakage(df, target_column):
            issues.append({
                "type": "target_leakage",
                "column": leak["column"],
                "severity": "critical",
                "description": f"Suspected leakage: correlation with target is {leak['correlation']:.3f}"
            })

        multi_outliers = self.check_multivariate_outliers(df)
        if multi_outliers and multi_outliers.get("has_outliers"):
            severity = "high" if multi_outliers["percentage"] > 5 else "medium"
            issues.append({
                "type": "multivariate_outliers",
                "column": "dataset_wide",
                "severity": severity,
                "description": f"Detected {multi_outliers['count']} multivariate outliers ({multi_outliers['percentage']:.1f}%)"
            })
        
        for col, count in self.check_outliers(df, target_column).items():
            pct = (count / len(df)) * 100
            severity = "high" if pct > 10 else "medium" if pct > 5 else "low"
            issues.append({
                "type": "outliers",
                "column": col,
                "severity": severity,
                "description": f"{count} outliers ({pct:.1f}%)"
            })
        
        return issues