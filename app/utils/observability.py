"""
Advanced Observability & Reliability Utilities
===============================================

Production-grade observability layer for ML pipeline auditing.
Provides statistical rigor throughout - no generic assessments.

Key Capabilities:
- Multi-dimensional reliability scoring with weighted components
- Dataset-aware diagnostic recommendations
- Generalization gap analysis and interpretation
- Calibration and confidence diagnostics
- Feature stability ranking with impact assessment
- Deployment readiness scoring with thresholds
- Observability telemetry and traceability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# DATA CLASSES FOR STRUCTURED DIAGNOSTICS
# ============================================================================

@dataclass
class GeneralizationDiagnostic:
    """Encapsulates generalization performance across train/test/CV."""
    train_score: float
    test_score: float
    cv_mean: float
    cv_std: float
    cv_scores: List[float]
    metric_name: str
    
    @property
    def train_test_gap(self) -> float:
        """Absolute difference: train - test"""
        return self.train_score - self.test_score
    
    @property
    def cv_test_gap(self) -> float:
        """CV mean vs holdout test set"""
        return self.cv_mean - self.test_score
    
    @property
    def fold_stability_ratio(self) -> float:
        """CV std dev normalized by mean (coefficient of variation)"""
        return self.cv_std / (self.cv_mean + 1e-8) if self.cv_mean > 0 else np.inf
    
    def get_severity_flags(self) -> List[Dict[str, str]]:
        """Extract interpretation-rich flags with domain-specific reasoning."""
        flags = []
        
        # Overfit detection
        if self.train_test_gap > 0.15:
            flags.append({
                'flag': 'High Overfit Risk',
                'severity': 'high',
                'detail': f'Train {self.metric_name.upper()} ({self.train_score:.3f}) exceeds test ({self.test_score:.3f}) by {self.train_test_gap:.3f}',
                'interpretation': 'Model memorizes training distribution; may fail on new data',
                'remediation': [
                    'Increase regularization (L1/L2, dropout)',
                    'Reduce model complexity (fewer parameters)',
                    'Add feature selection to remove noise',
                    'Use early stopping or cross-validation tuning'
                ]
            })
        elif self.train_test_gap > 0.08:
            flags.append({
                'flag': 'Mild Overfit',
                'severity': 'medium',
                'detail': f'Train/test gap: {self.train_test_gap:.3f}',
                'interpretation': 'Some memorization observed but within acceptable range',
                'remediation': ['Monitor performance in production', 'Plan incremental retraining']
            })
        
        # Fold instability
        if self.fold_stability_ratio > 0.10:
            flags.append({
                'flag': 'High Fold Variance',
                'severity': 'high',
                'detail': f'Cross-validation CV={self.cv_std:.3f} (relative std: {self.fold_stability_ratio:.3f})',
                'interpretation': 'Model performance highly dependent on which data points appear in training',
                'remediation': [
                    'Investigate feature importance variance across folds',
                    'Check for feature/target multicollinearity',
                    'Ensure stratification is correct for classification',
                    'Consider ensemble methods for robustness'
                ]
            })
        
        # Generalization decay (CV >> test)
        if self.cv_test_gap > 0.10:
            flags.append({
                'flag': 'Generalization Decay',
                'severity': 'high',
                'detail': f'CV mean ({self.cv_mean:.3f}) exceeds holdout ({self.test_score:.3f}) by {self.cv_test_gap:.3f}',
                'interpretation': 'Possible covariate shift, fold leakage, or train/test distribution mismatch',
                'remediation': [
                    'Verify train/test split is truly random (no temporal/spatial clustering)',
                    'Check for leakage in preprocessing (should be fit on train only)',
                    'Analyze PSI to detect distribution shift',
                    'Inspect if test set is fundamentally different distribution'
                ]
            })
        
        return flags


@dataclass
class MulticollinearityDiagnostic:
    """VIF-based multicollinearity analysis with remediation guidance."""
    vif_scores: Dict[str, float]
    correlation_matrix: Optional[pd.DataFrame] = None
    
    @property
    def critical_features(self) -> List[str]:
        """Features with VIF > 10 (severe multicollinearity)"""
        return [f for f, v in self.vif_scores.items() if v > 10.0]
    
    @property
    def high_vif_features(self) -> List[str]:
        """Features with VIF 5-10 (high multicollinearity)"""
        return [f for f, v in self.vif_scores.items() if 5.0 < v <= 10.0]
    
    @property
    def moderate_vif_features(self) -> List[str]:
        """Features with VIF 2-5 (moderate multicollinearity)"""
        return [f for f, v in self.vif_scores.items() if 2.0 < v <= 5.0]
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Dataset-specific recommendations for multicollinearity remediation."""
        recs = []
        
        if self.critical_features:
            recs.append({
                'severity': 'critical',
                'issue': f'{len(self.critical_features)} features have VIF > 10',
                'features': self.critical_features,
                'impact': 'Coefficients unstable; model highly sensitive to small data changes',
                'actions': [
                    f'Remove one of redundant pair: investigate {self.critical_features[0]}',
                    'Use dimensionality reduction (PCA, autoencoders)',
                    'Apply L1 regularization (Lasso) for automatic feature selection',
                    'Consider domain knowledge to retain most interpretable feature'
                ]
            })
        
        if self.high_vif_features:
            recs.append({
                'severity': 'high',
                'issue': f'{len(self.high_vif_features)} features have VIF 5-10',
                'features': self.high_vif_features,
                'impact': 'Coefficient estimates unreliable; high variance in predictions',
                'actions': [
                    'Create interaction terms to capture joint information',
                    'Consider regularization (Ridge regression)',
                    'Document collinearity for model interpretation'
                ]
            })
        
        return recs


@dataclass
class DriftDiagnostic:
    """PSI-based distribution shift detection with severity assessment."""
    psi_scores: Dict[str, float]
    threshold_low: float = 0.1
    threshold_medium: float = 0.25
    threshold_high: float = 0.5
    
    @property
    def low_drift_features(self) -> List[str]:
        """PSI < 0.1: minimal drift, acceptable"""
        return [f for f, p in self.psi_scores.items() if p < self.threshold_low]
    
    @property
    def medium_drift_features(self) -> List[str]:
        """PSI 0.1-0.25: notable drift, investigate"""
        return [f for f, p in self.psi_scores.items() 
               if self.threshold_low <= p < self.threshold_medium]
    
    @property
    def high_drift_features(self) -> List[str]:
        """PSI 0.25-0.5: high drift, concerning"""
        return [f for f, p in self.psi_scores.items() 
               if self.threshold_medium <= p < self.threshold_high]
    
    @property
    def critical_drift_features(self) -> List[str]:
        """PSI > 0.5: severe drift, critical"""
        return [f for f, p in self.psi_scores.items() if p >= self.threshold_high]
    
    def get_impact_assessment(self) -> Dict[str, str]:
        """Interpret drift severity and production impact."""
        total_drift_features = len(self.critical_drift_features) + len(self.high_drift_features)
        
        if total_drift_features > 5:
            return {
                'severity': 'critical',
                'message': 'Extensive distribution shift detected across features',
                'implication': 'Model trained on fundamentally different distribution',
                'action': 'Hold deployment; investigate data pipeline and source'
            }
        elif total_drift_features > 2:
            return {
                'severity': 'high',
                'message': f'{total_drift_features} features show significant drift',
                'implication': 'Model predictions may degrade on new data',
                'action': 'Deploy with enhanced monitoring; plan urgent retraining'
            }
        else:
            return {
                'severity': 'medium',
                'message': 'Moderate drift detected in subset of features',
                'implication': 'Stable but watch for pattern changes',
                'action': 'Monitor key drifting features in production'
            }


# ============================================================================
# STATISTICAL RIGOR & ANALYTICAL FUNCTIONS
# ============================================================================

class ObservabilityEngine:
    """Advanced observability functions for deployment readiness assessment."""
    
    @staticmethod
    def compute_calibration_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                    n_bins: int = 10) -> Dict[str, float]:
        """
        Compute calibration diagnostics.
        Measures if predicted probabilities match actual frequencies.
        
        Critical for risk-aware applications (lending, healthcare, etc.)
        """
        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        bin_true = np.zeros(n_bins)
        
        for i, (pred, true) in enumerate(zip(y_pred_proba, y_true)):
            bin_idx = min(int(pred * n_bins), n_bins - 1)
            bin_sums[bin_idx] += pred
            bin_counts[bin_idx] += 1
            bin_true[bin_idx] += true
        
        bin_probs = bin_sums / (bin_counts + 1e-8)
        bin_freq = bin_true / (bin_counts + 1e-8)
        
        # Expected Calibration Error
        ece = np.mean(np.abs(bin_probs[bin_counts > 0] - bin_freq[bin_counts > 0]))
        
        return {
            'expected_calibration_error': float(ece),
            'is_well_calibrated': ece < 0.05,
            'bin_centers': bin_centers.tolist(),
            'predicted_probs': bin_probs.tolist(),
            'actual_frequencies': bin_freq.tolist()
        }
    
    @staticmethod
    def compute_confidence_spread(y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Measure prediction confidence distribution.
        Ideally: either very confident (0.1) or very uncertain (0.5)
        Avoid: overconfident but wrong predictions
        """
        max_probs = np.max(y_pred_proba, axis=1) if y_pred_proba.ndim > 1 else y_pred_proba
        
        high_confidence = np.mean(max_probs > 0.8)
        medium_confidence = np.mean((0.5 <= max_probs) & (max_probs <= 0.8))
        low_confidence = np.mean(max_probs < 0.5)
        
        return {
            'high_confidence_pct': float(high_confidence * 100),
            'medium_confidence_pct': float(medium_confidence * 100),
            'low_confidence_pct': float(low_confidence * 100),
            'mean_confidence': float(np.mean(max_probs)),
            'confidence_std': float(np.std(max_probs)),
            'recommendation': 'Good' if high_confidence > 0.7 else 'Fair' if medium_confidence > 0.5 else 'Poor'
        }
    
    @staticmethod
    def compute_feature_stability_ranking(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                         sample_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Rank features by stability (inverse of drift severity).
        Returns actionable stability insights per feature.
        """
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        stability_scores = []
        
        for feature in numeric_features:
            if feature not in X_train.columns or feature not in X_test.columns:
                continue
            
            train_data = X_train[feature].dropna()
            test_data = X_test[feature].dropna()
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            # Distribution similarity via KL divergence
            # Bin both distributions
            bins = np.histogram_bin_edges(
                np.concatenate([train_data, test_data]), bins=30
            )
            
            p, _ = np.histogram(train_data, bins=bins)
            q, _ = np.histogram(test_data, bins=bins)
            
            # Normalize
            p = (p + 1e-6) / (np.sum(p) + 1e-6)
            q = (q + 1e-6) / (np.sum(q) + 1e-6)
            
            # KL divergence (symmetric)
            kl_div = np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))
            kl_div = kl_div / 2.0
            
            # Inverse to get stability score (higher = more stable)
            stability_score = np.exp(-kl_div)
            
            # Additional metrics
            train_mean, train_std = train_data.mean(), train_data.std()
            test_mean, test_std = test_data.mean(), test_data.std()
            
            mean_shift = abs(train_mean - test_mean) / (train_std + 1e-8)
            variance_ratio = test_std / (train_std + 1e-8)
            
            stability_scores.append({
                'feature': feature,
                'stability_score': float(stability_score),
                'kl_divergence': float(kl_div),
                'mean_shift_stdunits': float(mean_shift),
                'variance_ratio': float(variance_ratio),
                'rank': 0  # Will be set after sorting
            })
        
        # Sort by stability
        stability_scores = sorted(stability_scores, key=lambda x: x['stability_score'], reverse=True)
        for i, score_dict in enumerate(stability_scores):
            score_dict['rank'] = i + 1
        
        return stability_scores
    
    @staticmethod
    def compute_leakage_risk_score(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                  y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Sophisticated leakage detection beyond simple overlap.
        Measures: correlation patterns, unique value distributions, temporal indicators.
        """
        leakage_signals = []
        
        # 1. Row overlap (duplicate complete records)
        train_hashes = set()
        for _, row in X_train.iterrows():
            row_hash = hash(tuple(row.values.astype(str)))
            train_hashes.add(row_hash)
        
        overlap_count = 0
        for _, row in X_test.iterrows():
            row_hash = hash(tuple(row.values.astype(str)))
            if row_hash in train_hashes:
                overlap_count += 1
        
        overlap_pct = (overlap_count / len(X_test)) * 100 if len(X_test) > 0 else 0
        
        if overlap_pct > 0.1:
            leakage_signals.append({
                'type': 'exact_row_overlap',
                'pct': overlap_pct,
                'severity': 'critical' if overlap_pct > 1.0 else 'high'
            })
        
        # 2. Feature-target correlation asymmetry
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_features:
            if feature not in X_test.columns:
                continue
            
            train_corr = abs(X_train[feature].corr(y_train))
            test_corr = abs(X_test[feature].corr(y_test))
            
            corr_diff = abs(train_corr - test_corr)
            if corr_diff > 0.3:
                leakage_signals.append({
                    'type': 'feature_target_asymmetry',
                    'feature': feature,
                    'train_corr': float(train_corr),
                    'test_corr': float(test_corr),
                    'difference': float(corr_diff)
                })
        
        return {
            'leakage_signals': leakage_signals,
            'row_overlap_pct': float(overlap_pct),
            'is_high_leakage_risk': overlap_pct > 1.0 or len(leakage_signals) > 3,
            'recommendation': 'HALT DEPLOYMENT' if overlap_pct > 1.0 else 'Investigate signals'
        }
    
    @staticmethod
    def compute_preprocessing_integrity_score(preprocessing_steps: List[Dict]) -> Dict[str, Any]:
        """
        Score preprocessing pipeline for leakage risks.
        Checks: fit on train only, no test contamination, proper ordering.
        """
        score = 100.0
        issues = []
        
        # Check for fit-on-full-data antipatterns
        for step in preprocessing_steps:
            if step.get('applied_to') == 'full_data':
                score -= 25
                issues.append({
                    'type': 'fit_on_full_data',
                    'step': step.get('name', 'unknown'),
                    'impact': 'Test set used to fit transformation (leakage risk)'
                })
            
            if step.get('scaling_method') == 'StandardScaler' and not step.get('fit_train_only'):
                score -= 15
                issues.append({
                    'type': 'scaling_leakage',
                    'step': step.get('name'),
                    'impact': 'Scaler fit statistics may include test data'
                })
        
        return {
            'integrity_score': max(0, score),
            'is_leakage_safe': score >= 80,
            'issues': issues,
            'recommendations': [
                'Ensure all transformers fit exclusively on training set',
                'Apply transformers to test set using fit parameters from training',
                'Use sklearn Pipelines to enforce proper ordering',
                'Document which preprocessing is done train-only vs applied globally'
            ]
        }


class DeploymentReadinessAssessment:
    """
    Comprehensive deployment readiness evaluation with multi-stakeholder perspectives.
    """
    
    @staticmethod
    def generate_deployment_report(
        generalization: GeneralizationDiagnostic,
        multicollinearity: MulticollinearityDiagnostic,
        drift: DriftDiagnostic,
        performance_metric: float,
        issues_by_severity: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Synthesize multi-dimensional assessment into go/no-go recommendation.
        """
        
        score = 100.0
        blocking_issues = []
        warnings = []
        
        # 1. Generalization assessment
        gen_flags = generalization.get_severity_flags()
        if any(f['severity'] == 'high' for f in gen_flags):
            score -= 25
            blocking_issues.append('High generalization risk detected')
        elif gen_flags:
            score -= 10
            warnings.append('Mild generalization concerns')
        
        # 2. Multicollinearity assessment
        if multicollinearity.critical_features:
            score -= 30
            blocking_issues.append(f'{len(multicollinearity.critical_features)} features with critical multicollinearity')
        elif multicollinearity.high_vif_features:
            score -= 15
            warnings.append(f'{len(multicollinearity.high_vif_features)} features with high multicollinearity')
        
        # 3. Drift assessment
        drift_impact = drift.get_impact_assessment()
        if drift_impact['severity'] == 'critical':
            score -= 40
            blocking_issues.append('Critical feature drift detected')
        elif drift_impact['severity'] == 'high':
            score -= 20
            warnings.append('Significant feature drift')
        
        # 4. Performance threshold
        if performance_metric < 0.65:
            score -= 30
            blocking_issues.append(f'Performance below threshold ({performance_metric:.3f})')
        elif performance_metric < 0.75:
            score -= 15
            warnings.append(f'Performance modest ({performance_metric:.3f})')
        
        # 5. Issue accumulation
        total_critical = issues_by_severity.get('critical', 0)
        total_high = issues_by_severity.get('high', 0)
        
        if total_critical > 0:
            score -= min(30, total_critical * 10)
            blocking_issues.append(f'{total_critical} critical data integrity issues')
        
        if total_high > 3:
            score -= (total_high - 3) * 5
            warnings.append(f'{total_high} high-severity issues')
        
        score = max(0, min(100, score))
        
        return {
            'readiness_score': score,
            'can_deploy': score >= 75 and len(blocking_issues) == 0,
            'recommendation': 'GREEN - Deploy with standard monitoring' if score >= 85 else
                            'YELLOW - Deploy with enhanced monitoring' if score >= 70 else
                            'RED - Hold deployment',
            'blocking_issues': blocking_issues,
            'warnings': warnings,
            'approval_rationale': _generate_rationale(score, blocking_issues, warnings)
        }


def _generate_rationale(score: float, blocking: List[str], warnings: List[str]) -> str:
    """Generate human-readable deployment decision rationale."""
    if blocking:
        return f"DEPLOYMENT BLOCKED: {', '.join(blocking)}. Remediate before proceeding."
    elif warnings:
        return f"CONDITIONAL: Model shows {len(warnings)} areas of concern. Deploy with enhanced monitoring and readiness to rollback."
    elif score >= 85:
        return "Model passes reliability audit. Low risk of deployment failure. Proceed with standard production practices."
    else:
        return "Model meets minimum requirements. Deploy in canary mode with careful monitoring."