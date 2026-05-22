"""
Feature utilities for deep diagnostic analysis.
Computes VIF, PSI (Feature Drift), Informative Missingness, Volatility, and Overlap Hashing.
"""

import re
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import logging
from scipy.stats import pointbiserialr

logger = logging.getLogger(__name__)

class AdvancedDiagnostics:
    """Deep statistical diagnostics for ML reliability."""
    
    @staticmethod
    def compute_vif(df: pd.DataFrame) -> Dict[str, float]:
        """
        Computes Variance Inflation Factor (VIF) using the pseudo-inverse correlation matrix.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna()
        
        variances = numeric_df.var()
        numeric_df = numeric_df.loc[:, variances > 1e-5]
        
        if numeric_df.shape[1] < 2:
            return {}
            
        try:
            corr = numeric_df.corr().values
            inv_corr = np.linalg.pinv(corr)
            vif_values = np.diag(inv_corr)
            
            vif_dict = {col: float(val) for col, val in zip(numeric_df.columns, vif_values)}
            return dict(sorted(vif_dict.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.warning(f"VIF computation gracefully aborted: {str(e)}")
            return {}

    @staticmethod
    def compute_all_psi(df_expected: pd.DataFrame, df_actual: pd.DataFrame, buckets: int = 10, epsilon: float = 1e-4) -> List[Dict[str, Any]]:
        """Computes PSI for all numeric features and returns a structured list."""
        numeric_cols = df_expected.select_dtypes(include=[np.number]).columns
        results = []
        
        for col in numeric_cols:
            if col in df_actual.columns:
                psi_result = AdvancedDiagnostics._compute_single_psi(
                    df_expected[col].values, df_actual[col].values, buckets, epsilon
                )
                results.append({
                    "Feature": col,
                    "PSI Score": round(psi_result["score"], 4),
                    "Drift Severity": psi_result["severity"].upper()
                })
        
        return sorted(results, key=lambda x: x["PSI Score"], reverse=True)

    @staticmethod
    def _compute_single_psi(expected: np.ndarray, actual: np.ndarray, buckets: int, epsilon: float) -> Dict[str, Any]:
        """Core PSI math with stable quantile binning."""
        try:
            if len(expected) == 0 or len(actual) == 0:
                return {"score": 0.0, "severity": "low"}
                
            expected = expected[~np.isnan(expected) & ~np.isinf(expected)]
            actual = actual[~np.isnan(actual) & ~np.isinf(actual)]
            
            if len(np.unique(expected)) <= 1 or len(np.unique(actual)) <= 1:
                return {"score": 0.0, "severity": "low"}

            breakpoints = np.linspace(0, 100, buckets + 1)
            percentiles = np.percentile(expected, breakpoints)
            percentiles = np.unique(percentiles)
            
            if len(percentiles) < 2:
                return {"score": 0.0, "severity": "low"}

            expected_fractions = np.histogram(expected, bins=percentiles)[0] / len(expected)
            actual_fractions = np.histogram(actual, bins=percentiles)[0] / len(actual)

            expected_fractions = np.where(expected_fractions == 0, epsilon, expected_fractions)
            actual_fractions = np.where(actual_fractions == 0, epsilon, actual_fractions)

            psi = np.sum((actual_fractions - expected_fractions) * np.log(actual_fractions / expected_fractions))
            score = float(psi)
            
            if score < 0.1: severity = "low"
            elif score < 0.2: severity = "medium"
            else: severity = "high"
                
            return {"score": score, "severity": severity}
        except Exception:
            return {"score": 0.0, "severity": "low"}

    @staticmethod
    def compute_informative_missingness(df: pd.DataFrame, target_col: str, task_type: str) -> List[Dict[str, Any]]:
        """
        Detects if the ABSENCE of a value correlates with the target.
        This often indicates structural leakage or biased data collection.
        """
        if target_col not in df.columns:
            return []
            
        warnings = []
        target = df[target_col]
        
        # Only compute for features with actual missing values
        missing_cols = df.columns[df.isnull().sum() > 0]
        
        for col in missing_cols:
            if col == target_col: continue
            
            missing_indicator = df[col].isnull().astype(int)
            
            try:
                if task_type == 'classification' and target.nunique() == 2:
                    # Point biserial correlation for binary target
                    corr, p_value = pointbiserialr(missing_indicator, target.astype(float))
                else:
                    # Standard Pearson for continuous target or multi-class proxy
                    corr = missing_indicator.corr(target)
                
                if pd.notna(corr) and abs(corr) > 0.3:  # 0.3 is a strong signal for just missingness
                    severity = "critical" if abs(corr) > 0.5 else "high"
                    warnings.append({
                        "column": col,
                        "correlation": float(abs(corr)),
                        "severity": severity,
                        "description": f"Missing values in {col} strongly predict the target (Corr: {abs(corr):.2f})."
                    })
            except Exception:
                continue
                
        return sorted(warnings, key=lambda x: x["correlation"], reverse=True)

    @staticmethod
    def compute_train_test_variance_ratio(X_train: pd.DataFrame, X_test: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detects volatility and distribution collapse between splits.
        A ratio >> 1 or << 1 indicates the feature scale breaks down in production.
        """
        volatility = []
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in X_test.columns:
                var_train = X_train[col].var()
                var_test = X_test[col].var()
                
                if var_train > 1e-5 and var_test > 1e-5:
                    ratio = max(var_train / var_test, var_test / var_train)
                    if ratio > 3.0: # Variance changed by more than 3x
                        severity = "critical" if ratio > 10.0 else "high"
                        volatility.append({
                            "column": col,
                            "variance_ratio": float(ratio),
                            "severity": severity,
                            "description": f"Feature variance collapses/explodes across splits (Ratio: {ratio:.1f}x)."
                        })
        return sorted(volatility, key=lambda x: x["variance_ratio"], reverse=True)

    @staticmethod
    def calculate_row_overlap(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[int, float]:
        """Scalable train/test overlap detection using row hashing."""
        if df1.empty or df2.empty:
            return 0, 0.0
        df1_hashes = set(pd.util.hash_pandas_object(df1, index=False))
        df2_hashes = set(pd.util.hash_pandas_object(df2, index=False))
        overlap_count = len(df1_hashes.intersection(df2_hashes))
        overlap_pct = (overlap_count / len(df2_hashes)) * 100 if len(df2_hashes) > 0 else 0.0
        return overlap_count, float(overlap_pct)

class FeatureNameCleaner:
    def clean_feature_name(self, name: str) -> str:
        if not isinstance(name, str): return str(name)
        cleaned = name.replace('num__', '').replace('cat__', '').replace('remainder__', '')
        if '_' in cleaned and not any(c.isdigit() and c == cleaned[0] for c in cleaned):
            parts = cleaned.split('_', 1) 
            if len(parts) == 2:
                cleaned = f"{parts[0].title()} - {parts[1].title()}"
        else:
            cleaned = cleaned.replace('_', ' ').title()
        return cleaned