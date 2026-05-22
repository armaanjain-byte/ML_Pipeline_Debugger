"""
Observability utilities for ML pipeline monitoring and reliability scoring.
Provides production-grade diagnostics and health metrics for ML systems.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class ExecutionStep:
    """Represents a single pipeline execution step."""
    name: str
    status: str = "completed"  # completed, running, failed
    duration_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status,
            'duration_ms': self.duration_ms,
            'timestamp': self.timestamp.isoformat(),
            'error': self.error
        }


class PipelineObserver:
    """Monitors and records pipeline execution metrics."""
    
    def __init__(self):
        self.steps: List[ExecutionStep] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def record_step(self, step_id: str, step_name: str, duration_ms: int, 
                   status: str = "completed", error: Optional[str] = None) -> None:
        """Record a pipeline execution step."""
        step = ExecutionStep(
            name=step_name,
            status=status,
            duration_ms=duration_ms,
            error=error
        )
        self.steps.append(step)
    
    def get_execution_steps(self) -> List[Dict[str, Any]]:
        """Get all recorded steps."""
        return [step.to_dict() for step in self.steps]
    
    def get_total_duration(self) -> int:
        """Get total pipeline execution time."""
        return sum(step.duration_ms for step in self.steps)
    
    def get_step_breakdown(self) -> Dict[str, int]:
        """Get breakdown of time by step."""
        return {step.name: step.duration_ms for step in self.steps}
    
    def has_failures(self) -> bool:
        """Check if any step failed."""
        return any(step.status == "failed" for step in self.steps)


class ReliabilityScorer:
    """Computes ML pipeline reliability and health scores."""
    
    # Weighting scheme for overall score
    COMPONENT_WEIGHTS = {
        'data_quality': 0.25,      # Data issues
        'leakage_risk': 0.30,      # Target leakage risk
        'balance_health': 0.20,    # Class balance
        'outlier_health': 0.15,    # Outlier contamination
        'missing_health': 0.10     # Missing value impact
    }
    
    # Severity to impact mapping
    SEVERITY_IMPACT = {
        'critical': 1.0,
        'high': 0.7,
        'medium': 0.4,
        'low': 0.1
    }
    
    def compute_overall_score(self, checks_output: Dict[str, Any]) -> float:
        """
        Compute overall pipeline reliability score (0-100).
        
        Considers:
        - Data quality issues
        - Leakage risk
        - Class balance
        - Outlier contamination
        - Missing values
        """
        component_scores = self.compute_component_scores(checks_output)
        
        weighted_score = sum(
            component_scores.get(component, 0) * weight
            for component, weight in self.COMPONENT_WEIGHTS.items()
        )
        
        return max(0, min(100, weighted_score))  # Clamp to [0, 100]
    
    def compute_component_scores(self, checks_output: Dict[str, Any]) -> Dict[str, float]:
        """Compute individual component reliability scores."""
        scores = {}
        
        # Data Quality Score
        scores['data_quality'] = self._compute_data_quality_score(checks_output)
        
        # Leakage Risk Score
        scores['leakage_risk'] = self._compute_leakage_risk_score(checks_output)
        
        # Class Balance Score
        scores['balance_health'] = self._compute_balance_health_score(checks_output)
        
        # Outlier Health Score
        scores['outlier_health'] = self._compute_outlier_health_score(checks_output)
        
        # Missing Data Health Score
        scores['missing_health'] = self._compute_missing_health_score(checks_output)
        
        return scores
    
    def _compute_data_quality_score(self, checks_output: Dict[str, Any]) -> float:
        """
        Score based on general data quality issues.
        Considers constant features, duplicates, data types.
        """
        base_score = 100.0
        
        # Constant features: very problematic
        constant_features = checks_output.get('constant_features', [])
        base_score -= len(constant_features) * 15
        
        # Duplicates: problematic
        duplicates = checks_output.get('duplicates', {})
        dup_pct = duplicates.get('duplicate_percentage', 0)
        base_score -= min(10, dup_pct / 10)
        
        # High correlation: moderate issue
        high_corr = checks_output.get('high_correlation', [])
        base_score -= len(high_corr) * 5
        
        return max(0, min(100, base_score))
    
    def _compute_leakage_risk_score(self, checks_output: Dict[str, Any]) -> float:
        """
        Score based on target leakage risk.
        Returns percentage of risk (0-100 where 0 = no risk, 100 = high risk).
        """
        leakage_features = checks_output.get('target_leakage', [])
        
        if not leakage_features:
            return 0.0  # No risk
        
        # Multiple leaking features = high risk
        risk_score = min(100, len(leakage_features) * 50)
        
        return float(risk_score)
    
    def _compute_balance_health_score(self, checks_output: Dict[str, Any]) -> float:
        """
        Score based on class balance (classification only).
        Returns 0-100 where 100 = perfectly balanced.
        """
        imbalance = checks_output.get('class_imbalance', {})
        
        if not imbalance or imbalance.get('num_classes', 0) == 0:
            return 100.0  # Not applicable or perfectly balanced
        
        # Use minority class percentage as proxy
        minority_pct = imbalance.get('minority_class_percentage', 50)
        
        # Score: higher minority percentage = better balance
        # Penalize heavily below 10%, moderately below 30%
        if minority_pct < 5:
            return 10.0
        elif minority_pct < 10:
            return 30.0
        elif minority_pct < 20:
            return 60.0
        elif minority_pct < 40:
            return 85.0
        else:
            return 100.0
    
    def _compute_outlier_health_score(self, checks_output: Dict[str, Any]) -> float:
        """
        Score based on outlier contamination.
        Returns 0-100 where 100 = no outliers.
        """
        outliers = checks_output.get('outliers', {})
        
        multivariate = outliers.get('multivariate', {})
        outlier_pct = multivariate.get('percentage', 0)
        
        univariate_count = len(outliers.get('univariate', {}))
        
        # Combine multivariate and univariate concerns
        base_penalty = min(50, outlier_pct * 5)  # Each 1% = 5 points penalty
        univariate_penalty = min(30, univariate_count * 3)  # Each feature = 3 points
        
        health_score = 100 - base_penalty - univariate_penalty
        
        return max(0, min(100, health_score))
    
    def _compute_missing_health_score(self, checks_output: Dict[str, Any]) -> float:
        """
        Score based on missing value impact.
        Returns 0-100 where 100 = no missing values.
        """
        missing = checks_output.get('missing_values', {})
        
        if not missing:
            return 100.0
        
        # Penalize based on worst missing percentage
        worst_missing = max(missing.values()) if missing else 0
        
        if worst_missing > 50:
            return 10.0
        elif worst_missing > 30:
            return 30.0
        elif worst_missing > 10:
            return 60.0
        elif worst_missing > 5:
            return 80.0
        else:
            return 95.0
    
    def get_health_status(self, score: float) -> str:
        """Get human-readable health status."""
        if score >= 85:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 50:
            return "Fair"
        elif score >= 30:
            return "Poor"
        else:
            return "Critical"
    
    def get_score_color(self, score: float) -> str:
        """Get color for score visualization."""
        if score >= 85:
            return "#27ae60"  # Green
        elif score >= 70:
            return "#2ecc71"  # Light green
        elif score >= 50:
            return "#f39c12"  # Orange
        elif score >= 30:
            return "#e67e22"  # Dark orange
        else:
            return "#e74c3c"  # Red


class RiskAssessment:
    """Comprehensive risk assessment for ML pipelines."""
    
    def __init__(self):
        self.risks: List[Dict[str, Any]] = []
    
    def assess(self, checks_output: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment."""
        self.risks = []
        
        # Check each category
        self._assess_data_quality(checks_output)
        self._assess_leakage(checks_output)
        self._assess_bias(checks_output)
        self._assess_outliers(checks_output)
        self._assess_missing(checks_output)
        
        return {
            'risks': self.risks,
            'critical_count': sum(1 for r in self.risks if r['severity'] == 'critical'),
            'high_count': sum(1 for r in self.risks if r['severity'] == 'high'),
            'medium_count': sum(1 for r in self.risks if r['severity'] == 'medium'),
            'low_count': sum(1 for r in self.risks if r['severity'] == 'low')
        }
    
    def _assess_data_quality(self, checks_output: Dict[str, Any]) -> None:
        """Assess data quality risks."""
        constant_features = checks_output.get('constant_features', [])
        
        if constant_features:
            self.risks.append({
                'type': 'data_quality',
                'severity': 'high',
                'title': f'{len(constant_features)} Constant Feature(s)',
                'description': 'Features with zero variance provide no predictive signal',
                'impact': 'Model will ignore these features, wasting model capacity'
            })
        
        duplicates = checks_output.get('duplicates', {})
        if duplicates.get('has_duplicates'):
            dup_pct = duplicates.get('duplicate_percentage', 0)
            if dup_pct > 5:
                self.risks.append({
                    'type': 'data_quality',
                    'severity': 'high',
                    'title': f'{dup_pct:.1f}% Duplicate Rows',
                    'description': 'Exact duplicate rows indicate data quality issues',
                    'impact': 'Inflates model confidence on overweighted samples'
                })
    
    def _assess_leakage(self, checks_output: Dict[str, Any]) -> None:
        """Assess target leakage risks."""
        leakage = checks_output.get('target_leakage', [])
        
        for leak in leakage:
            self.risks.append({
                'type': 'leakage',
                'severity': 'critical',
                'title': f'Suspected Target Leakage: {leak["column"]}',
                'description': f'Correlation with target: {leak["correlation"]:.3f}',
                'impact': 'Model will fail in production due to data leakage'
            })
    
    def _assess_bias(self, checks_output: Dict[str, Any]) -> None:
        """Assess class imbalance and bias risks."""
        imbalance = checks_output.get('class_imbalance', {})
        
        if imbalance and imbalance.get('is_imbalanced'):
            minority_pct = imbalance.get('minority_class_percentage', 0)
            if minority_pct < 5:
                self.risks.append({
                    'type': 'bias',
                    'severity': 'critical',
                    'title': f'Severe Class Imbalance ({minority_pct:.1f}% minority)',
                    'description': 'Minority class severely underrepresented',
                    'impact': 'Model may predict majority class exclusively'
                })
            elif minority_pct < 20:
                self.risks.append({
                    'type': 'bias',
                    'severity': 'high',
                    'title': f'Class Imbalance ({minority_pct:.1f}% minority)',
                    'description': 'Unequal class distribution detected',
                    'impact': 'Model accuracy can be misleading'
                })
    
    def _assess_outliers(self, checks_output: Dict[str, Any]) -> None:
        """Assess outlier contamination risks."""
        outliers = checks_output.get('outliers', {})
        
        multivariate = outliers.get('multivariate', {})
        outlier_pct = multivariate.get('percentage', 0)
        
        if outlier_pct > 10:
            self.risks.append({
                'type': 'outliers',
                'severity': 'high',
                'title': f'High Outlier Contamination ({outlier_pct:.1f}%)',
                'description': 'Multivariate outliers detected',
                'impact': 'May bias tree-based models and distance metrics'
            })
        elif outlier_pct > 5:
            self.risks.append({
                'type': 'outliers',
                'severity': 'medium',
                'title': f'Outlier Contamination ({outlier_pct:.1f}%)',
                'description': 'Some multivariate anomalies detected',
                'impact': 'May affect model generalization'
            })
    
    def _assess_missing(self, checks_output: Dict[str, Any]) -> None:
        """Assess missing value impact."""
        missing = checks_output.get('missing_values', {})
        
        for col, pct in missing.items():
            if pct > 50:
                self.risks.append({
                    'type': 'missing',
                    'severity': 'critical',
                    'title': f'Severe Missing Data: {col}',
                    'description': f'{pct:.1f}% of values missing',
                    'impact': 'Consider dropping this feature entirely'
                })
            elif pct > 30:
                self.risks.append({
                    'type': 'missing',
                    'severity': 'high',
                    'title': f'High Missing Data: {col}',
                    'description': f'{pct:.1f}% of values missing',
                    'impact': 'Imputation strategy will heavily influence model'
                })


class MetricsAggregator:
    """Aggregates and computes derived metrics."""
    
    @staticmethod
    def compute_model_health(metrics: Dict[str, float], task_type: str) -> float:
        """
        Compute overall model health score based on metrics.
        Returns 0-100.
        """
        if task_type == "classification":
            # For classification, average of accuracy, precision, recall, f1
            key_metrics = [
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1', 0)
            ]
            return float(np.mean([m for m in key_metrics if m > 0]) * 100)
        else:
            # For regression, use r2 as proxy (0-1 scale)
            r2 = metrics.get('r2', 0)
            return max(0, min(100, r2 * 100))
    
    @staticmethod
    def compute_stability_score(cv_metrics: Dict[str, float]) -> float:
        """
        Compute stability score from cross-validation results.
        Low std = high stability.
        """
        cv_std_keys = [k for k in cv_metrics.keys() if 'std' in k.lower()]
        
        if not cv_std_keys:
            return 100.0
        
        avg_std = np.mean([cv_metrics[k] for k in cv_std_keys])
        
        # Convert std to stability (lower std = higher stability)
        # Assume std > 0.2 is unstable
        stability = max(0, 100 - (avg_std * 500))
        
        return float(min(100, stability))