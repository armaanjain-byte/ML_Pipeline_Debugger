"""
Feature utilities for cleaning transformed feature names and analyzing correlations.
Provides observability for high-cardinality and encoded features.
"""

import re
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


class FeatureNameCleaner:
    """Transforms sklearn-generated feature names into human-readable format."""
    
    def __init__(self):
        self.patterns = {
            r'num__': '',  # Remove 'num__' prefix
            r'cat__': '',  # Remove 'cat__' prefix
            r'_(.+)_': r' - \1',  # Transform x_value_ to x - value
            r'_': ' ',  # Replace remaining underscores with spaces
        }
    
    def clean_feature_name(self, name: str) -> str:
        """
        Clean sklearn-transformed feature names.
        
        Examples:
            'num__tenure' -> 'tenure'
            'cat__gender_Male' -> 'gender - Male'
            'cat__state_CA' -> 'state - CA'
        """
        if not isinstance(name, str):
            return str(name)
        
        # Step 1: Remove numeric prefixes
        cleaned = name.replace('num__', '').replace('cat__', '')
        
        # Step 2: Handle one-hot encoded categorical features
        # Pattern: feature_value becomes feature - value
        if '_' in cleaned and not any(c.isdigit() and c == cleaned[0] for c in cleaned):
            parts = cleaned.split('_')
            if len(parts) == 2:
                cleaned = f"{parts[0]} - {parts[1]}"
            elif len(parts) > 2:
                # Handle cases like postal_code_12345
                cleaned = f"{parts[0]} - {' '.join(parts[1:])}"
        else:
            cleaned = cleaned.replace('_', ' ')
        
        # Step 3: Capitalize properly
        cleaned = self._smart_capitalize(cleaned)
        
        return cleaned
    
    def _smart_capitalize(self, text: str) -> str:
        """Capitalize in a readable way while preserving codes."""
        parts = text.split(' - ')
        result = []
        
        for part in parts:
            # If it looks like a code (all caps or short), keep it
            if len(part) <= 3 and part.isupper():
                result.append(part)
            # If it's numeric, keep it
            elif part.isdigit():
                result.append(part)
            # Otherwise capitalize normally
            else:
                result.append(part.title())
        
        return ' - '.join(result)
    
    def batch_clean(self, feature_names: List[str]) -> Dict[str, str]:
        """Clean multiple feature names at once."""
        return {name: self.clean_feature_name(name) for name in feature_names}


class CorrelationAnalyzer:
    """Analyzes feature correlations and high-cardinality patterns."""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
    
    def compute_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix for numeric features."""
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr().abs()
    
    def find_high_correlations(self, corr_matrix: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Find feature pairs with correlation above threshold."""
        high_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > self.threshold:
                    high_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        float(corr_val)
                    ))
        
        return sorted(high_corr, key=lambda x: x[2], reverse=True)
    
    def identify_high_cardinality_features(self, df: pd.DataFrame, 
                                          threshold: int = 50) -> Dict[str, int]:
        """Identify categorical features with many unique values."""
        high_cardinality = {}
        
        categorical = df.select_dtypes(include=['object', 'category'])
        
        for col in categorical.columns:
            unique_count = df[col].nunique()
            if unique_count > threshold:
                high_cardinality[col] = unique_count
        
        return dict(sorted(high_cardinality.items(), 
                          key=lambda x: x[1], reverse=True))
    
    def compute_feature_redundancy(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute redundancy score for each feature.
        Redundancy = how much information it shares with other features.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()
        
        redundancy_scores = {}
        
        for col in corr_matrix.columns:
            # Average absolute correlation with all other features
            correlations = corr_matrix[col].drop(col)
            redundancy_scores[col] = float(correlations.mean())
        
        return dict(sorted(redundancy_scores.items(), 
                          key=lambda x: x[1], reverse=True))


class FeatureImportanceAnalyzer:
    """Analyzes feature importance with statistical context."""
    
    @staticmethod
    def compute_relative_importance(importance_dict: Dict[str, float]) -> Dict[str, float]:
        """Compute relative importance as percentage of total."""
        if not importance_dict:
            return {}
        
        total = sum(importance_dict.values())
        if total == 0:
            return importance_dict
        
        return {name: (imp / total) * 100 for name, imp in importance_dict.items()}
    
    @staticmethod
    def compute_cumulative_importance(importance_dict: Dict[str, float], 
                                     threshold: float = 0.95) -> Tuple[int, float]:
        """
        Find how many features are needed to explain threshold% of variance.
        Returns (num_features, cumulative_importance)
        """
        sorted_imp = sorted(importance_dict.values(), reverse=True)
        
        if not sorted_imp:
            return 0, 0.0
        
        total = sum(sorted_imp)
        if total == 0:
            return 0, 0.0
        
        cumsum = 0
        for idx, imp in enumerate(sorted_imp, 1):
            cumsum += imp
            if cumsum / total >= threshold:
                return idx, cumsum / total
        
        return len(sorted_imp), 1.0
    
    @staticmethod
    def identify_important_features(importance_dict: Dict[str, float],
                                   n_top: int = 10) -> Dict[str, float]:
        """Get top N important features."""
        return dict(sorted(importance_dict.items(), 
                          key=lambda x: x[1], reverse=True)[:n_top])
    
    @staticmethod
    def compute_feature_variance(importance_dict: Dict[str, float]) -> float:
        """Compute variance in feature importance (higher = more selective)."""
        if len(importance_dict) < 2:
            return 0.0
        
        values = list(importance_dict.values())
        mean = np.mean(values)
        variance = np.mean([(x - mean) ** 2 for x in values])
        
        return float(np.sqrt(variance))


class FeatureGroupAnalyzer:
    """Groups features by type and transformation for insight."""
    
    def __init__(self, cleaner: FeatureNameCleaner = None):
        self.cleaner = cleaner or FeatureNameCleaner()
    
    def group_by_source(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Group features by their original source (numeric vs categorical)."""
        groups = {'numeric': [], 'categorical': []}
        
        for name in feature_names:
            if name.startswith('num__'):
                groups['numeric'].append(self.cleaner.clean_feature_name(name))
            elif name.startswith('cat__'):
                groups['categorical'].append(self.cleaner.clean_feature_name(name))
            else:
                # Assume numeric if no prefix
                groups['numeric'].append(self.cleaner.clean_feature_name(name))
        
        return groups
    
    def group_by_importance_tier(self, importance_dict: Dict[str, float],
                                 tiers: List[float] = None) -> Dict[str, List[Tuple[str, float]]]:
        """
        Group features by importance percentile tiers.
        
        Args:
            importance_dict: Feature importance scores
            tiers: Percentile boundaries, default [0.5, 0.8, 1.0]
        """
        if tiers is None:
            tiers = [0.5, 0.8, 1.0]
        
        # Compute percentiles
        sorted_values = sorted(importance_dict.values(), reverse=True)
        tier_names = ['Top Tier', 'Mid Tier', 'Low Tier']
        
        groups = {name: [] for name in tier_names}
        
        total = sum(sorted_values)
        cumsum = 0
        current_tier = 0
        tier_thresholds = [t * total for t in tiers]
        
        for feature, importance in sorted(importance_dict.items(), 
                                         key=lambda x: x[1], reverse=True):
            cumsum += importance
            
            # Find which tier this feature belongs to
            while current_tier < len(tier_thresholds) - 1 and cumsum > tier_thresholds[current_tier]:
                current_tier += 1
            
            tier_name = tier_names[min(current_tier, len(tier_names) - 1)]
            groups[tier_name].append((self.cleaner.clean_feature_name(feature), importance))
        
        return {k: v for k, v in groups.items() if v}  # Remove empty groups