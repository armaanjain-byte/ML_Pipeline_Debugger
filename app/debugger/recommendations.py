
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)

# ============================================================================
# STRUCTURED RECOMMENDATION TYPES
# ============================================================================

@dataclass
class Recommendation:
    """Structured recommendation with full context and remediation."""
    type: str  # 'leakage', 'drift', 'multicollinearity', 'cardinality', 'missingness', etc.
    severity: str  # 'critical', 'high', 'medium', 'low'
    title: str  # Concise summary
    column: str  # Affected feature(s) or 'dataset_wide'
    description: str  # What was detected
    statistical_evidence: str  # Quantitative proof
    impact: str  # What happens if ignored
    rationale: str  # Why this happens
    actions: List[str]  # Step-by-step remediation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'severity': self.severity,
            'title': self.title,
            'column': self.column,
            'description': self.description,
            'statistical_evidence': self.statistical_evidence,
            'impact': self.impact,
            'rationale': self.rationale,
            'actions': self.actions
        }


# ============================================================================
# CONTEXT-AWARE RECOMMENDATION GENERATORS
# ============================================================================

class HighCardinalityAnalyzer:
    """Generate high-cardinality feature recommendations with specific guidance."""
    
    @staticmethod
    def analyze_and_recommend(df: pd.DataFrame, target: str, task: str) -> List[Recommendation]:
        """
        Analyze high-cardinality features and generate tailored recommendations.
        High cardinality (>100 unique values) introduces specific risks:
        - Tree-based models memorize instead of generalize
        - One-hot encoding explodes feature space
        - Hashing creates collisions
        - Encoding creates spurious ordinal structure
        """
        recommendations = []
        
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        for col in categorical_cols:
            if col == target:
                continue
            
            nunique = df[col].nunique()
            cardinality_pct = (nunique / len(df)) * 100
            
            if nunique > 100:
                # Analyze cardinality severity
                if nunique > 10000:
                    severity = 'critical'
                    context = 'near-perfect unique identifier'
                elif nunique > 1000:
                    severity = 'critical'
                    context = 'extremely high cardinality'
                elif nunique > 500:
                    severity = 'high'
                    context = 'very high cardinality'
                else:
                    severity = 'high'
                    context = 'high cardinality'
                
                # Compute target leakage
                target_by_category = df.groupby(col)[target].nunique()
                categories_with_single_target = (target_by_category == 1).sum()
                single_target_pct = (categories_with_single_target / nunique) * 100
                
                # Compute cardinality usefulness (correlation or variance explanation)
                if task == 'classification':
                    entropy = df.groupby(col)[target].apply(
                        lambda x: -np.sum((x.value_counts() / len(x)) * np.log2(x.value_counts() / len(x) + 1e-8))
                    ).mean()
                    avg_entropy = -np.sum((df[target].value_counts() / len(df)) * np.log2(df[target].value_counts() / len(df) + 1e-8))
                    information_gain = avg_entropy - entropy
                else:
                    information_gain = 0.0  # Would need regression-specific metric
                
                rec = Recommendation(
                    type='high_cardinality',
                    severity=severity,
                    title=f'High-Cardinality Feature: {col}',
                    column=col,
                    description=f'Feature "{col}" has {nunique} unique values ({cardinality_pct:.1f}% of dataset size)',
                    statistical_evidence=f"""
                        Cardinality: {nunique} unique values
                        Cardinality Ratio: {cardinality_pct:.1f}% of dataset size
                        Categories with single target value: {categories_with_single_target} ({single_target_pct:.1f}%)
                        Information gain: {information_gain:.4f} bits
                    """,
                    impact="""
                        - Tree-based models: Creates separate leaf per category, memorizing rather than learning
                        - One-hot encoding: Generates {nunique} binary features, exploding feature space
                        - Sparsity: Most encoded features appear in <1% of samples, creating noise
                        - Generalization: Model fails on unseen category values in production
                    """,
                    rationale=f"""
                        High cardinality features violate the assumption that feature values capture generalizable patterns.
                        With {nunique} categories, tree ensembles build separate paths for each, which is memorization not learning.
                        In production, any new customer ID or category value will be treated as missing/unknown.
                        {f"The feature shows {information_gain:.3f} bits of information gain, suggesting limited predictive value." if information_gain < 0.1 else f"Despite {nunique} values, information gain is low ({information_gain:.3f}), indicating sparse signal."}
                    """,
                    actions=[
                        'OPTION 1 - Drop: If {nunique} >> dataset size, feature is likely an identifier (customer_id, transaction_id). Remove entirely.',
                        f'OPTION 2 - Aggregation: Create derived features from {col} (e.g., frequency encoding, target encoding, hashing into {int(np.sqrt(nunique))} buckets)',
                        f'OPTION 3 - Target Encoding: Map categories to mean target value. Use cross-validation to avoid leakage.',
                        f'OPTION 4 - Domain Engineering: Consult domain experts; may encode business logic (e.g., "VIP", "risky", "frequent").',
                        f'RECOMMENDED: Drop if cardinality_pct > 50%. Otherwise, use target encoding with proper CV fold isolation.'
                    ]
                )
                recommendations.append(rec)
        
        return recommendations


class MulticollinearityRecommender:
    """Generate multicollinearity remediation recommendations."""
    
    @staticmethod
    def analyze_and_recommend(vif_scores: Dict[str, float],
                             correlation_matrix: pd.DataFrame) -> List[Recommendation]:
        """
        Generate multicollinearity remediation with specific feature pair analysis.
        """
        recommendations = []
        
        # Critical VIF (>10)
        critical_vif = {f: v for f, v in vif_scores.items() if v > 10}
        
        for feature, vif in critical_vif.items():
            # Find correlated partners
            if feature in correlation_matrix.columns:
                correlations = correlation_matrix[feature].abs().sort_values(ascending=False)
                top_correlated = correlations[1:4]  # Skip self-correlation
                
                rec = Recommendation(
                    type='multicollinearity',
                    severity='critical',
                    title=f'Severe Multicollinearity: {feature}',
                    column=feature,
                    description=f'Feature "{feature}" has VIF = {vif:.1f} (threshold: 10.0)',
                    statistical_evidence=f"""
                        VIF Score: {vif:.2f}
                        Top correlated features:
                        {chr(10).join([f"  - {fname}: r = {fcorr:.3f}" for fname, fcorr in top_correlated.items()])}
                    """,
                    impact="""
                        - Coefficient estimates are highly unstable (small data changes → large coefficient swings)
                        - Standard errors inflated (wider confidence intervals, weak hypothesis tests)
                        - Feature importance unreliable (may flip sign across models)
                        - Model exhibits high sensitivity to feature scaling
                        - Interpretation misleading: coefficients don't reflect true marginal effects
                    """,
                    rationale=f"""
                        VIF = {vif:.1f} means 1/(1 - R²) = {vif:.1f}, so R² ≈ {1 - 1/vif:.3f}.
                        This feature is explained {(1 - 1/vif)*100:.1f}% by other features, essentially redundant.
                        The feature "{feature}" is highly correlated with {', '.join(top_correlated.index.tolist()[:2])},
                        indicating it captures the same information.
                    """,
                    actions=[
                        f'STEP 1: Compare {feature} with {top_correlated.index[0]} (r={top_correlated.iloc[0]:.3f})',
                        f'STEP 2: Decide which to keep based on: interpretability, domain relevance, data quality',
                        f'STEP 3: Drop the less relevant feature',
                        'STEP 4: Recompute VIF; confirm reduction',
                        'ALTERNATIVE: Apply L1 regularization (Lasso) for automatic selection, or PCA for compression'
                    ]
                )
                recommendations.append(rec)
        
        return recommendations


class DriftRecommender:
    """Generate drift remediation recommendations with severity assessment."""
    
    @staticmethod
    def analyze_and_recommend(psi_scores: Dict[str, float],
                             train_distributions: Dict[str, Dict],
                             test_distributions: Dict[str, Dict]) -> List[Recommendation]:
        """
        Generate drift recommendations with specific distribution shift explanations.
        """
        recommendations = []
        
        for feature, psi in psi_scores.items():
            if psi < 0.25:
                continue  # Low drift, not concerning
            
            severity = 'critical' if psi > 0.5 else 'high' if psi > 0.25 else 'medium'
            
            # Analyze shift characteristics
            train_dist = train_distributions.get(feature, {})
            test_dist = test_distributions.get(feature, {})
            
            shift_type = _characterize_distribution_shift(train_dist, test_dist)
            
            rec = Recommendation(
                type='feature_drift',
                severity=severity,
                title=f'Distribution Drift: {feature}',
                column=feature,
                description=f'Feature "{feature}" shows significant distribution shift (PSI: {psi:.3f})',
                statistical_evidence=f"""
                    Population Stability Index (PSI): {psi:.3f}
                    Shift Pattern: {shift_type}
                    Train Distribution: {train_dist.get('description', 'N/A')}
                    Test Distribution: {test_dist.get('description', 'N/A')}
                """,
                impact=f"""
                    - Model trained on fundamentally different data distribution
                    - Predictions degrade on test/production data ({severity.upper()} severity)
                    - Features weights (importances) may not transfer
                    - Performance metrics from validation inflated vs. production
                """,
                rationale=f"""
                    PSI = {psi:.3f} indicates {shift_type} between train and test distributions.
                    {_interpret_psi_value(psi)}.
                    This could indicate temporal drift (data changes over time), sample bias
                    (test set from different source), or data pipeline issues.
                """,
                actions=[
                    f'INVESTIGATE: Why does {feature} distribution differ between train and test?',
                    '  - Check data collection date/source for test set',
                    '  - Verify preprocessing applied consistently',
                    '  - Confirm no pipeline bugs (encoding errors, missing value handling)',
                    f'MONITOR: Track {feature} distribution in production; set alerts if PSI > 0.2',
                    'REMEDIATE: If confirmed drift, retrain model on combined train+test or recent data',
                    'DOCUMENT: Record drift pattern for model card; inform stakeholders of limitations'
                ]
            )
            recommendations.append(rec)
        
        return recommendations


class LeakageDetector:
    """Generate target leakage recommendations with specific remediation."""
    
    @staticmethod
    def analyze_and_recommend(issues: List[Dict],
                             X_train: pd.DataFrame,
                             X_test: pd.DataFrame,
                             y_train: pd.Series,
                             y_test: pd.Series) -> List[Recommendation]:
        """
        Detect and recommend remediation for target leakage patterns.
        """
        recommendations = []
        
        leakage_issues = [i for i in issues if 'leakage' in i.get('type', '')]
        
        for issue in leakage_issues:
            # Quantify leakage severity
            if issue.get('type') == 'split_overlap':
                overlap_pct = float(issue.get('description', '').split('(')[1].split('%')[0])
                severity = 'critical' if overlap_pct > 1.0 else 'high'
                
                rec = Recommendation(
                    type='target_leakage',
                    severity=severity,
                    title='Train/Test Contamination (Row Overlap)',
                    column='dataset_wide',
                    description=f'{overlap_pct:.2f}% of test set appears identically in training set',
                    statistical_evidence=f"""
                        Test overlap: {overlap_pct:.2f}% of {len(X_test)} test samples
                        Absolute count: ~{int(len(X_test) * overlap_pct / 100)} duplicate rows
                        Expected by chance: <0.01% (assuming random split)
                    """,
                    impact="""
                        - Model achieves artificially high validation scores (memorizes test samples)
                        - Production performance will be significantly lower (gap of 10-20%+ typical)
                        - Model cannot generalize; only replicating training data
                        - Deployment results in immediate performance degradation
                    """,
                    rationale=f"""
                        {overlap_pct:.2f}% overlap far exceeds random chance (<0.01%).
                        This indicates systematic leakage: either:
                        1. Duplicates in source data (same customer/entity appears twice)
                        2. Temporal contamination (train/test not truly separated by date)
                        3. Sampling without replacement error
                        Model performance on overlapped samples is worthless; it has memorized them.
                    """,
                    actions=[
                        'CRITICAL: Do NOT deploy this model',
                        'STEP 1: Identify source of duplicates (data quality issue or sampling error?)',
                        'STEP 2: Remove duplicates from full dataset before splitting',
                        'STEP 3: If temporal data, ensure train/test split is by date, not random',
                        'STEP 4: Re-run entire pipeline on cleaned data with proper split',
                        'STEP 5: Recompute metrics on new holdout; likely to be 5-15% lower',
                        'INVESTIGATE: Document why duplicates existed; prevent in future ingestion'
                    ]
                )
                recommendations.append(rec)
        
        return recommendations


class MissingnessRecommender:
    """Generate missingness remediation recommendations."""
    
    @staticmethod
    def analyze_and_recommend(df: pd.DataFrame,
                             target: str,
                             task: str) -> List[Recommendation]:
        """
        Generate missingness recommendations with informative missing analysis.
        """
        recommendations = []
        
        for col in df.columns:
            if col == target:
                continue
            
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            if missing_pct < 1:
                continue  # Negligible
            
            if missing_pct > 50:
                severity = 'high'
                context = 'majority missing'
            elif missing_pct > 20:
                severity = 'high'
                context = 'substantial missingness'
            else:
                severity = 'medium'
                context = 'minor missingness'
            
            # Check if missingness is informative (correlated with target)
            if task == 'classification':
                missing_indicator = df[col].isnull().astype(int)
                target_corr = abs(missing_indicator.corr(df[target]))
            else:
                target_corr = 0.0
            
            rec = Recommendation(
                type='missing_values',
                severity=severity,
                title=f'Missing Values: {col}',
                column=col,
                description=f'Feature "{col}" has {missing_pct:.1f}% missing values',
                statistical_evidence=f"""
                    Missing percentage: {missing_pct:.1f}%
                    Missing count: {df[col].isnull().sum()} / {len(df)}
                    Correlation with target: {target_corr:.3f}
                    {f"(Informative missing: pattern predicts target)" if target_corr > 0.1 else "(Random missingness)"}
                """,
                impact=f"""
                    - Most imputation methods lose information (MCAR assumption violated)
                    - If >50% missing: feature contains minimal signal
                    - Imputation adds bias if missing not random (MCAR)
                    - Model may accidentally learn from missingness pattern
                """,
                rationale=f"""
                    {missing_pct:.1f}% missing is {'above' if missing_pct > 5 else 'within'} the 5% threshold.
                    {f"The missingness pattern correlates {target_corr:.3f} with target, indicating informative missing." if target_corr > 0.1 else "Missingness appears random."}
                """,
                actions=[
                    f'OPTION 1 - Drop: If {missing_pct:.1f}% > 30%, feature has minimal information. Delete.',
                    f'OPTION 2 - Create indicator: Add binary "{col}_missing" feature to capture missingness pattern.',
                    f'OPTION 3 - Domain fill: Ask domain experts for sensible default (e.g., unknown, N/A, median).',
                    f'OPTION 4 - KNN imputation: Use K-nearest neighbors to borrow values from similar records.',
                    f'RECOMMENDED: {"Drop" if missing_pct > 30 else "Create missing indicator and drop original" if missing_pct > 10 else "Simple median/mode imputation"}'
                ]
            )
            recommendations.append(rec)
        
        return recommendations


# ============================================================================
# HELPER FUNCTIONS FOR INTERPRETATION
# ============================================================================

def _characterize_distribution_shift(train_dist: Dict, test_dist: Dict) -> str:
    """Characterize the type of distribution shift (mean shift, variance change, etc.)"""
    if not train_dist or not test_dist:
        return 'Unknown shift pattern'
    
    train_mean = train_dist.get('mean')
    test_mean = test_dist.get('mean')
    train_std = train_dist.get('std', 1)
    test_std = test_dist.get('std', 1)
    
    if train_mean and test_mean:
        mean_diff = abs(train_mean - test_mean) / (train_std + 1e-8)
        if mean_diff > 1.0:
            return 'Mean shift (location drift)'
    
    if train_std and test_std:
        std_ratio = test_std / (train_std + 1e-8)
        if std_ratio > 1.5 or std_ratio < 0.67:
            return 'Variance change (spread drift)'
    
    return 'General distributional shift'


def _interpret_psi_value(psi: float) -> str:
    """Interpret PSI value magnitude."""
    if psi > 0.5:
        return f"PSI = {psi:.3f} indicates SEVERE shift (distribution changed fundamentally)"
    elif psi > 0.25:
        return f"PSI = {psi:.3f} indicates HIGH shift (distribution notably different)"
    elif psi > 0.1:
        return f"PSI = {psi:.3f} indicates MEDIUM shift (distribution somewhat different)"
    else:
        return f"PSI = {psi:.3f} indicates LOW shift (distributions similar)"


# ============================================================================
# MAIN RECOMMENDATION ENGINE
# ============================================================================

class DatasetAwareRecommendationEngine:
    """
    Master recommendation engine combining all analyzers.
    Generates dataset-specific, statistically-grounded recommendations.
    """
    
    def __init__(self, df: pd.DataFrame, target: str, task: str):
        self.df = df
        self.target = target
        self.task = task
        self.recommendations: List[Recommendation] = []
    
    def generate(self, checks_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive recommendations from diagnostic output.
        """
        
        # 1. High cardinality
        cat_recs = HighCardinalityAnalyzer.analyze_and_recommend(self.df, self.target, self.task)
        self.recommendations.extend(cat_recs)
        
        # 2. Multicollinearity
        vif_data = checks_output.get('high_correlation', {})
        if vif_data:
            corr_matrix = checks_output.get('correlation_matrix')
            collinearity_recs = MulticollinearityRecommender.analyze_and_recommend(
                vif_data, corr_matrix
            )
            self.recommendations.extend(collinearity_recs)
        
        # 3. Missingness
        missing_recs = MissingnessRecommender.analyze_and_recommend(self.df, self.target, self.task)
        self.recommendations.extend(missing_recs)
        
        # 4. Convert to output format
        return {
            'recommendations': [r.to_dict() for r in self.recommendations],
            'total_recommendations': len(self.recommendations),
            'critical_count': len([r for r in self.recommendations if r.severity == 'critical']),
            'high_count': len([r for r in self.recommendations if r.severity == 'high'])
        }

# Backward compatibility alias
RecommendationEngine = DatasetAwareRecommendationEngine
if __name__ == "__main__":
    print("Recommendations module loaded successfully")