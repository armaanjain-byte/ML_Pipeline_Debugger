from typing import Dict, Any, List

class RecommendationEngine:
    """Generates context-aware, structured engineering recommendations."""
    
    RECOMMENDATION_CONTEXT = {
        "target_leakage": {
            "title": "Remove Leaking Feature Pathway",
            "impact": "Prevents catastrophic 0% accuracy in production.",
            "actions": [
                "Drop the feature from the training dataset entirely.",
                "Review data warehouse logic. Ensure this column isn't populated *after* the prediction event."
            ],
            "rationale": "High correlation with the target indicates the feature contains future information. The model will memorize this shortcut."
        },
        "split_overlap": {
            "title": "Resolve Train/Test Overlap (Data Leakage)",
            "impact": "Restores trust in your holdout evaluation metrics.",
            "actions": [
                "Ensure row identifiers are unique.",
                "Use GroupKFold if rows represent the same logical entity across time.",
                "Deduplicate the raw dataset before the train_test_split."
            ],
            "rationale": "Identical rows in both Train and Test sets cause the model to 'memorize' the test set, creating artificially inflated metrics."
        },
        "feature_drift": {
            "title": "Mitigate Distribution Drift (High PSI)",
            "impact": "Prevents concept drift decay in production.",
            "actions": [
                "If the split was temporal, this is valid drift. Use robust scaling.",
                "If the split was random, your seed resulted in a biased sample. Stratify your splits.",
                "Drop the drifting feature if it provides low feature importance."
            ],
            "rationale": "High PSI indicates the feature's statistical distribution changed significantly between training and evaluation spaces."
        },
        "informative_missingness": {
            "title": "Address Informative Missingness",
            "impact": "Prevents the model from learning a structural data-collection bias.",
            "actions": [
                "Investigate WHY the data is missing. Is the absence of the value caused by the target event?",
                "If it's a collection artifact, drop the feature. If it's valid behavior, explicitly create an 'is_missing' indicator."
            ],
            "rationale": "The absence of a value (NaN) correlates strongly with your target. Imputing this blindly will destroy a massive signal, while leaving it unchecked might be leakage."
        },
        "feature_volatility": {
            "title": "Stabilize Feature Variance Collapse",
            "impact": "Ensures distance-based algorithms and neural networks do not explode.",
            "actions": [
                "Apply Log1p or PowerTransformer to compress the scale.",
                "Check for extreme outliers in the Train set that didn't make it to the Test set."
            ],
            "rationale": "The variance of this feature shifts massively (>>3x) between Train and Test sets, meaning its scale is unstable across splits."
        },
        "multicollinearity": {
            "title": "Reduce Multicollinearity (High VIF)",
            "impact": "Stabilizes coefficients and feature importance attributions.",
            "actions": [
                "Drop the feature with the higher VIF score.",
                "Use PCA to compress the correlated feature space."
            ],
            "rationale": "High Variance Inflation Factor (VIF) destabilizes the model, inflates variance, and makes interpretability impossible."
        }
    }
    
    def generate(self, checks_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured recommendations bound to diagnostic evidence."""
        issues = checks_output.get("issues", [])
        recommendations = []
        
        critical_count = 0
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for issue in issues:
            issue_type = issue.get("type", "unknown")
            severity = issue.get("severity", "medium")
            
            rec = {
                "title": issue_type.replace('_', ' ').title(),
                "severity": severity,
                "description": issue.get("description", "Anomaly detected."),
                "rationale": "Data quality anomaly detected requiring engineering investigation.",
                "impact": "Improves overall model stability.",
                "actions": ["Conduct a manual review of this feature."],
                "column": issue.get("column", "unknown"),
                "issue_type": issue_type
            }
            
            if issue_type in self.RECOMMENDATION_CONTEXT:
                context = self.RECOMMENDATION_CONTEXT[issue_type]
                rec.update({
                    "title": context["title"],
                    "rationale": context["rationale"],
                    "impact": context["impact"],
                    "actions": context["actions"]
                })
            
            if severity == "critical": critical_count += 1
            if severity in severity_counts: severity_counts[severity] += 1
            
            recommendations.append(rec)
            
        severity_map = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: severity_map.get(x['severity'], 4))
        
        return {
            "recommendations": recommendations,
            "total_issues": len(issues),
            "critical_issues": critical_count,
            "severity_breakdown": severity_counts
        }