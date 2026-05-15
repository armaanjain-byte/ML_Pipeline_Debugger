import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.ensemble import IsolationForest
import logging
from app.core.config import DiagnosticConfig

logger = logging.getLogger(__name__)

class DataChecks:
    """
    Comprehensive data validation engine.
    Performs statistical diagnostics on raw training data to identify 
    architectural flaws before model training.
    """
    
    def run_all_checks(self, df: pd.DataFrame, target_column: str, task_type: str = "classification") -> Dict[str, Any]:
        """Executes the full suite of data diagnostics."""
        # We consolidate outlier detection into one method for efficiency
        outlier_results = self.check_all_outliers(df, target_column)
        
        return {
            "missing_values": self.check_missing_values(df),
            "constant_features": self.check_constant_features(df),
            "duplicates": self.check_duplicates(df),
            "class_imbalance": self.check_class_imbalance(df, target_column, task_type),
            "high_correlation": self.check_high_correlation(df),
            "target_leakage": self.check_target_leakage(df, target_column),
            "outliers": outlier_results,
            "data_types": self.check_data_types(df),
            "issues": self._summarize_issues(df, target_column, task_type, outlier_results)
        }
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """Identifies columns with null values and their percentage."""
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        return {k: v for k, v in missing_pct.items() if v > 0}
    
    def check_constant_features(self, df: pd.DataFrame) -> List[str]:
        """Detects features with zero variance (only one unique value)."""
        return [col for col in df.columns if df[col].nunique() <= 1]
    
    def check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identifies exact row duplicates in the dataset."""
        num_duplicates = int(df.duplicated().sum())
        return {
            "total_duplicates": num_duplicates,
            "duplicate_percentage": float(num_duplicates / len(df) * 100) if len(df) > 0 else 0.0,
            "has_duplicates": num_duplicates > 0
        }
    
    def check_class_imbalance(self, df: pd.DataFrame, target_column: str, task_type: str) -> Dict[str, Any]:
        """Checks for skewed class distributions in classification tasks."""
        if target_column not in df.columns or task_type == "regression":
            return {}
        
        value_counts = df[target_column].value_counts(normalize=True).to_dict()
        if not value_counts:
            return {}
        
        minority_pct = min(value_counts.values()) * 100
        return {
            "distribution": value_counts,
            "minority_class_percentage": float(minority_pct),
            "is_imbalanced": minority_pct < 20,
            "num_classes": len(value_counts)
        }
    
    def check_high_correlation(self, df: pd.DataFrame, threshold: float = 0.9) -> List[tuple]:
        """Detects multicollinearity between independent features."""
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

    def check_target_leakage(self, df: pd.DataFrame, target_column: str) -> List[Dict[str, Any]]:
        """Scans for features that correlate too highly with the target."""
        threshold = DiagnosticConfig.LEAKAGE_THRESHOLD
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

    def check_all_outliers(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Single pass to detect both univariate (IQR) and multivariate (Isolation Forest) outliers."""
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        results = {"univariate": {}, "multivariate": {"count": 0, "percentage": 0.0, "indices": []}}
        
        if numeric_df.empty:
            return results

        # 1. Univariate (IQR Method) - Column by column
        for col in numeric_df.columns:
            if col == target_column or numeric_df[col].nunique() <= 2:
                continue
            q1, q3 = numeric_df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            count = ((numeric_df[col] < (q1 - 1.5 * iqr)) | (numeric_df[col] > (q3 + 1.5 * iqr))).sum()
            if count > 0:
                results["univariate"][col] = int(count)

        # 2. Multivariate (Isolation Forest) - Deep statistical anomalies
        if len(numeric_df) >= 50:
            iso = IsolationForest(contamination="auto", random_state=42)
            preds = iso.fit_predict(numeric_df)
            outlier_indices = numeric_df.index[preds == -1].tolist()
            results["multivariate"] = {
                "count": len(outlier_indices),
                "percentage": float((len(outlier_indices) / len(numeric_df)) * 100),
                "indices": outlier_indices[:10]
            }
        
        return results

    def check_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Returns the schema mapping for all columns."""
        return df.dtypes.astype(str).to_dict()
    
    def _summarize_issues(self, df: pd.DataFrame, target_column: str, task_type: str, outlier_results: Dict) -> List[Dict[str, Any]]:
        """Translates raw check results into a structured list of actionable issues."""
        issues = []
        
        # 1. Missing Values
        for col, pct in self.check_missing_values(df).items():
            severity = "high" if pct > 50 else "medium" if pct > 20 else "low"
            issues.append({
                "type": "missing_values",
                "column": col,
                "severity": severity,
                "description": f"{pct:.1f}% missing values"
            })
        
        # 2. Constant Features
        for col in self.check_constant_features(df):
            issues.append({
                "type": "constant_feature",
                "column": col,
                "severity": "high",
                "description": "Feature has only one unique value"
            })
        
        # 3. Duplicate Rows
        dup_info = self.check_duplicates(df)
        if dup_info.get("has_duplicates"):
            issues.append({
                "type": "duplicate_rows",
                "column": "dataset_wide",
                "severity": "medium",
                "description": f"{dup_info['total_duplicates']} duplicate rows ({dup_info['duplicate_percentage']:.1f}%)"
            })
        
        # 4. Class Imbalance
        imbalance = self.check_class_imbalance(df, target_column, task_type)
        if imbalance and imbalance.get("is_imbalanced"):
            issues.append({
                "type": "class_imbalance",
                "column": target_column,
                "severity": "high",
                "description": f"Minority class: {imbalance['minority_class_percentage']:.1f}%"
            })
        
        # 5. High Correlation
        for col1, col2, corr in self.check_high_correlation(df):
            issues.append({
                "type": "high_correlation",
                "column": f"{col1} ↔ {col2}",
                "severity": "medium",
                "description": f"Correlation: {corr:.3f}"
            })

        # 6. Target Leakage
        for leak in self.check_target_leakage(df, target_column):
            issues.append({
                "type": "target_leakage",
                "column": leak["column"],
                "severity": "critical",
                "description": f"Suspected leakage: correlation with target is {leak['correlation']:.3f}"
            })

        # 7. Multivariate Outliers
        multi = outlier_results["multivariate"]
        if multi["count"] > 0:
            severity = "high" if multi["percentage"] > 5 else "medium"
            issues.append({
                "type": "multivariate_outliers",
                "column": "dataset_wide",
                "severity": severity,
                "description": f"Detected {multi['count']} multivariate outliers ({multi['percentage']:.1f}%)"
            })
        
        # 8. Univariate Outliers
        for col, count in outlier_results["univariate"].items():
            pct = (count / len(df)) * 100
            severity = "high" if pct > 10 else "medium" if pct > 5 else "low"
            issues.append({
                "type": "outliers",
                "column": col,
                "severity": severity,
                "description": f"{count} outliers ({pct:.1f}%)"
            })
        
        return issues