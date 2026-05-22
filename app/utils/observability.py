"""
Observability utilities for ML Pipeline Debugger.
Produces credible, transparent reliability scoring.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ExecutionMetrics:
    """Pipeline execution metrics."""
    total_rows: int
    numeric_features: int
    categorical_features: int
    missing_value_pct: float
    duplicate_row_count: int
    training_samples: int
    test_samples: int
    

class ReliabilityScorer:
    """
    Computes transparent, justified reliability score (0-100).
    
    Scoring methodology:
    - Data Quality (25%): Missing values, constants, duplicates
    - Leakage Risk (30%): Target correlation detection
    - Class Balance (20%): Minority class representation
    - Outlier Health (15%): Multivariate anomaly contamination
    - Missing Data (10%): NaN imputation difficulty
    
    Each component is independently computed and weighted.
    Penalties are explicit and transparent.
    """
    
    # Score thresholds for each component
    CRITICAL_THRESHOLD = 30
    POOR_THRESHOLD = 50
    FAIR_THRESHOLD = 70
    GOOD_THRESHOLD = 85
    
    def compute_overall_score(self, checks: Dict[str, Any]) -> float:
        """
        Compute weighted overall reliability score.
        
        Args:
            checks: Output from DataChecks.run_all_checks()
            
        Returns:
            float: Score 0-100, where higher is healthier
        """
        components = self.compute_component_scores(checks)
        
        # Weighted sum
        weights = {
            'data_quality': 0.25,
            'leakage_risk': 0.30,
            'balance_health': 0.20,
            'outlier_health': 0.15,
            'missing_health': 0.10
        }
        
        overall = sum(
            components.get(name, 0) * weight
            for name, weight in weights.items()
        )
        
        return max(0, min(100, overall))
    
    def compute_component_scores(self, checks: Dict[str, Any]) -> Dict[str, float]:
        """Compute individual component scores."""
        return {
            'data_quality': self._score_data_quality(checks),
            'leakage_risk': self._score_leakage_risk(checks),
            'balance_health': self._score_class_balance(checks),
            'outlier_health': self._score_outlier_health(checks),
            'missing_health': self._score_missing_health(checks),
        }
    
    def _score_data_quality(self, checks: Dict[str, Any]) -> float:
        """
        Score data quality (0-100).
        Penalties for constant features, duplicates, high correlation.
        """
        base = 100.0
        
        # Constant features: -15 each
        constant_count = len(checks.get('constant_features', []))
        base -= min(100, constant_count * 15)
        
        # Duplicates: -5 to -20 based on percentage
        duplicates = checks.get('duplicates', {})
        dup_pct = duplicates.get('duplicate_percentage', 0)
        if dup_pct > 0:
            base -= min(20, dup_pct / 5)
        
        # High correlation: -5 per pair
        high_corr = len(checks.get('high_correlation', []))
        base -= min(25, high_corr * 5)
        
        return max(0, min(100, base))
    
    def _score_leakage_risk(self, checks: Dict[str, Any]) -> float:
        """
        Score leakage risk (0-100 scale, where 0 = no risk).
        This is inverted: 0 is good, 100 is critical.
        """
        leakage_features = checks.get('target_leakage', [])
        
        if not leakage_features:
            return 0  # No leakage = low risk
        
        # Multiple leaking features = higher risk
        risk = min(100, len(leakage_features) * 50)
        return float(risk)
    
    def _score_class_balance(self, checks: Dict[str, Any]) -> float:
        """
        Score class balance for classification (0-100, higher is better).
        Regression tasks get 100 (not applicable).
        """
        imbalance = checks.get('class_imbalance', {})
        
        if not imbalance:
            return 100  # Regression or no imbalance
        
        minority_pct = imbalance.get('minority_class_percentage', 50)
        
        # Threshold-based scoring
        if minority_pct < 5:
            return 10
        elif minority_pct < 10:
            return 30
        elif minority_pct < 20:
            return 60
        elif minority_pct < 40:
            return 85
        else:
            return 100
    
    def _score_outlier_health(self, checks: Dict[str, Any]) -> float:
        """
        Score based on outlier contamination (0-100, higher is better).
        """
        outliers = checks.get('outliers', {})
        
        multivariate = outliers.get('multivariate', {})
        outlier_pct = multivariate.get('percentage', 0)
        
        univariate_count = len(outliers.get('univariate', {}))
        
        # Multivariate: penalize 5 points per percentage point
        base = 100 - min(50, outlier_pct * 5)
        
        # Univariate: penalize 3 points per affected feature
        base -= min(30, univariate_count * 3)
        
        return max(0, min(100, base))
    
    def _score_missing_health(self, checks: Dict[str, Any]) -> float:
        """
        Score based on missing value impact (0-100, higher is better).
        """
        missing = checks.get('missing_values', {})
        
        if not missing:
            return 100  # No missing values
        
        worst_missing = max(missing.values()) if missing else 0
        
        # Threshold-based penalties
        if worst_missing > 50:
            return 10
        elif worst_missing > 30:
            return 30
        elif worst_missing > 10:
            return 60
        elif worst_missing > 5:
            return 80
        else:
            return 95
    
    def get_health_status(self, score: float) -> str:
        """Get human-readable health status."""
        if score >= self.GOOD_THRESHOLD:
            return "Excellent"
        elif score >= self.FAIR_THRESHOLD:
            return "Good"
        elif score >= self.POOR_THRESHOLD:
            return "Fair"
        elif score >= self.CRITICAL_THRESHOLD:
            return "Poor"
        else:
            return "Critical"
    
    def get_score_color(self, score: float) -> str:
        """Get color for visualization."""
        if score >= self.GOOD_THRESHOLD:
            return "#27ae60"  # Green
        elif score >= self.FAIR_THRESHOLD:
            return "#3498db"  # Blue
        elif score >= self.POOR_THRESHOLD:
            return "#f39c12"  # Orange
        elif score >= self.CRITICAL_THRESHOLD:
            return "#e67e22"  # Dark orange
        else:
            return "#c0392b"  # Red


class PipelineObserver:
    """
    Tracks pipeline execution with real engineering metrics.
    Not about decorative timing - about data flow diagnostics.
    """
    
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
    
    def record_data_load(self, row_count: int, column_count: int) -> None:
        """Record data loading metrics."""
        self.metrics['input_rows'] = row_count
        self.metrics['input_columns'] = column_count
    
    def record_split(self, train_rows: int, test_rows: int) -> None:
        """Record train/test split."""
        self.metrics['train_rows'] = train_rows
        self.metrics['test_rows'] = test_rows
        self.metrics['train_pct'] = (train_rows / (train_rows + test_rows)) * 100
    
    def record_preprocessing(self, numeric_count: int, categorical_count: int) -> None:
        """Record feature preprocessing."""
        self.metrics['numeric_features'] = numeric_count
        self.metrics['categorical_features'] = categorical_count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary."""
        return {
            'data_shape': f"{self.metrics.get('input_rows', 0)} rows × {self.metrics.get('input_columns', 0)} cols",
            'train_test_split': f"{self.metrics.get('train_pct', 0):.1f}% / {100-self.metrics.get('train_pct', 0):.1f}%",
            'feature_breakdown': f"{self.metrics.get('numeric_features', 0)} numeric, {self.metrics.get('categorical_features', 0)} categorical"
        }