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
    Performs deep statistical diagnostics on raw training data to identify 
    architectural flaws before model training.
    """
    
    def run_all_checks(self, df: pd.DataFrame, target_column: str, task_type: str = "classification") -> Dict[str, Any]:
        """Executes the full suite of data diagnostics efficiently."""
        
        # Cache numeric/categorical split to prevent redundant selection operations
        numeric_df = df.select_dtypes(include=[np.number])
        cat_df = df.select_dtypes(exclude=[np.number])
        
        missing_values = self.check_missing_values(df)
        constant_features = self.check_constant_features(df)
        near_constant = self.check_near_constant_features(df)
        high_cardinality = self.check_high_cardinality(cat_df)
        skewed_features = self.check_skewness(numeric_df, target_column)
        
        outlier_results = self.check_all_outliers(numeric_df, target_column)
        
        return {
            "missing_values": missing_values,
            "constant_features": constant_features,
            "near_constant_features": near_constant,
            "high_cardinality": high_cardinality,
            "skewed_features": skewed_features,
            "duplicates": self.check_duplicates(df),
            "class_imbalance": self.check_class_imbalance(df, target_column, task_type),
            "high_correlation": self.check_high_correlation(numeric_df),
            "target_leakage": self.check_target_leakage(numeric_df, target_column),
            "outliers": outlier_results,
            "data_types": self.check_data_types(df),
            "issues": self._summarize_issues(
                df=df, target_column=target_column, task_type=task_type,
                missing_values=missing_values, constant_features=constant_features,
                near_constant=near_constant, high_cardinality=high_cardinality,
                skewed_features=skewed_features, outlier_results=outlier_results
            )
        }
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        return {k: v for k, v in missing_pct.items() if v > 0}
    
    def check_constant_features(self, df: pd.DataFrame) -> List[str]:
        return [col for col in df.columns if df[col].nunique() <= 1]

    def check_near_constant_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Flags features where the most frequent value is overwhelmingly dominant (>99%)."""
        near_constants = {}
        for col in df.columns:
            if df[col].nunique() > 1:
                top_freq = df[col].value_counts(normalize=True).iloc[0]
                if top_freq > 0.99:
                    near_constants[col] = float(top_freq)
        return near_constants

    def check_high_cardinality(self, cat_df: pd.DataFrame) -> Dict[str, int]:
        """Identifies categorical columns with excessive unique values."""
        high_card = {}
        for col in cat_df.columns:
            nunique = cat_df[col].nunique()
            if nunique > 100:  # Bound based on config heuristics
                high_card[col] = nunique
        return high_card

    def check_skewness(self, numeric_df: pd.DataFrame, target_column: str) -> Dict[str, float]:
        """Detects highly skewed numerical distributions requiring transformation."""
        skewed = {}
        for col in numeric_df.columns:
            if col != target_column:
                skew_val = numeric_df[col].skew()
                if abs(skew_val) > 2.0:
                    skewed[col] = float(skew_val)
        return skewed
    
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
        if not value_counts:
            return {}
        
        minority_pct = min(value_counts.values()) * 100
        return {
            "distribution": value_counts,
            "minority_class_percentage": float(minority_pct),
            "is_imbalanced": minority_pct < 20,
            "num_classes": len(value_counts)
        }
    
    def check_high_correlation(self, numeric_df: pd.DataFrame, threshold: float = 0.9) -> List[tuple]:
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

    def check_target_leakage(self, numeric_df: pd.DataFrame, target_column: str) -> List[Dict[str, Any]]:
        threshold = DiagnosticConfig.LEAKAGE_THRESHOLD
        leakage_warnings = []
        
        if target_column in numeric_df.columns:
            correlations = numeric_df.corr()[target_column].abs()
            for col, corr in correlations.items():
                if col != target_column and corr > threshold:
                    leakage_warnings.append({
                        "column": col,
                        "correlation": float(corr)
                    })
        return leakage_warnings

    def check_all_outliers(self, numeric_df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        numeric_df_clean = numeric_df.dropna()
        results = {"univariate": {}, "multivariate": {"count": 0, "percentage": 0.0, "indices": []}}
        
        if numeric_df_clean.empty:
            return results

        # Univariate (IQR)
        for col in numeric_df_clean.columns:
            if col == target_column or numeric_df_clean[col].nunique() <= 2:
                continue
            q1, q3 = numeric_df_clean[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                count = ((numeric_df_clean[col] < (q1 - 1.5 * iqr)) | (numeric_df_clean[col] > (q3 + 1.5 * iqr))).sum()
                if count > 0:
                    results["univariate"][col] = int(count)

        # Multivariate (Isolation Forest) - Bounded safely to prevent false 50%+ anomaly flags
        if len(numeric_df_clean) >= 50:
            # Cap maximum contamination strictly at 5%, scaling down for smaller datasets
            contamination = min(0.05, max(0.01, 100.0 / len(numeric_df_clean)))
            iso = IsolationForest(contamination=contamination, random_state=42)
            preds = iso.fit_predict(numeric_df_clean)
            outlier_indices = numeric_df_clean.index[preds == -1].tolist()
            results["multivariate"] = {
                "count": len(outlier_indices),
                "percentage": float((len(outlier_indices) / len(numeric_df_clean)) * 100),
                "indices": outlier_indices[:10]
            }
        
        return results

    def check_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        return df.dtypes.astype(str).to_dict()
    
    def _summarize_issues(self, df: pd.DataFrame, target_column: str, task_type: str, 
                          missing_values: Dict, constant_features: List, near_constant: Dict,
                          high_cardinality: Dict, skewed_features: Dict, outlier_results: Dict) -> List[Dict[str, Any]]:
        
        issues = []
        
        for col, pct in sorted(missing_values.items(), key=lambda x: x[1], reverse=True):
            severity = "critical" if pct > 50 else "high" if pct > 30 else "medium" if pct > 10 else "low"
            issues.append({
                "type": "missing_values", "column": col, "severity": severity,
                "description": f"{pct:.1f}% missing values detected."
            })
        
        for col in constant_features:
            issues.append({
                "type": "constant_feature", "column": col, "severity": "high",
                "description": "Feature contains only one unique value (zero variance)."
            })
            
        for col, freq in near_constant.items():
            issues.append({
                "type": "near_constant_feature", "column": col, "severity": "medium",
                "description": f"Highly imbalanced feature ({freq*100:.1f}% dominant value)."
            })
            
        for col, count in high_cardinality.items():
            issues.append({
                "type": "high_cardinality", "column": col, "severity": "high",
                "description": f"Categorical feature with excessive cardinality ({count} unique values)."
            })
            
        for col, skew in skewed_features.items():
            issues.append({
                "type": "high_skewness", "column": col, "severity": "medium",
                "description": f"High numerical skewness detected (Skew: {skew:.2f})."
            })

        dup_info = self.check_duplicates(df)
        if dup_info.get("has_duplicates"):
            dup_pct = dup_info['duplicate_percentage']
            severity = "high" if dup_pct > 5 else "medium"
            issues.append({
                "type": "duplicate_rows", "column": "dataset_wide", "severity": severity,
                "description": f"{dup_info['total_duplicates']} exact duplicate rows ({dup_pct:.1f}%)."
            })
        
        imbalance = self.check_class_imbalance(df, target_column, task_type)
        if imbalance and imbalance.get("is_imbalanced"):
            minority_pct = imbalance['minority_class_percentage']
            severity = "critical" if minority_pct < 5 else "high" if minority_pct < 10 else "medium"
            issues.append({
                "type": "class_imbalance", "column": target_column, "severity": severity,
                "description": f"Severe target imbalance (Minority class: {minority_pct:.1f}%)."
            })
        
        # High Correlation
        corr_data = self.check_high_correlation(df.select_dtypes(include=[np.number]))
        for col1, col2, corr in sorted(corr_data, key=lambda x: x[2], reverse=True):
            severity = "high" if corr > 0.95 else "medium"
            issues.append({
                "type": "high_correlation", "column": f"{col1} ↔ {col2}", "severity": severity,
                "description": f"Strong multicollinearity detected (Correlation: {corr:.3f})."
            })

        # Target Leakage
        for leak in self.check_target_leakage(df.select_dtypes(include=[np.number]), target_column):
            issues.append({
                "type": "target_leakage", "column": leak["column"], "severity": "critical",
                "description": f"Suspected target leakage. Correlation with target is abnormally high ({leak['correlation']:.3f})."
            })

        # Outliers
        multi = outlier_results["multivariate"]
        if multi["count"] > 0:
            pct = multi["percentage"]
            severity = "high" if pct > 3 else "medium"  # Since we capped contamination at 5%
            issues.append({
                "type": "multivariate_outliers", "column": "dataset_wide", "severity": severity,
                "description": f"Detected {multi['count']} deep multivariate anomalies ({pct:.1f}%)."
            })
        
        for col, count in sorted(outlier_results["univariate"].items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = (count / len(df)) * 100
            severity = "high" if pct > 10 else "medium" if pct > 5 else "low"
            issues.append({
                "type": "outliers", "column": col, "severity": severity,
                "description": f"{count} univariate outliers detected ({pct:.1f}%)."
            })
        
        return issues