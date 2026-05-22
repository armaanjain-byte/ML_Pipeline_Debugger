from typing import Dict, Any, List

class RecommendationEngine:
    """Generates actionable engineering recommendations based on diagnostic issues."""
    
    # Recommendation templates with context
    RECOMMENDATION_CONTEXT = {
        "target_leakage": {
            "title": "Remove Leaking Feature",
            "severity": "critical",
            "action": "remove_leaking_feature",
            "recommendations": [
                "Drop the feature from the dataset immediately.",
                "Verify if this data point is actually available at prediction time in production.",
                "Review the data collection pipeline to understand how leakage occurred."
            ],
            "rationale": "Features nearly identical to the target indicate data leakage, resulting in artificially high accuracy during training but catastrophic failure in production."
        },
        "multivariate_outliers": {
            "title": "Investigate Multivariate Anomalies",
            "severity": "high",
            "action": "investigate_anomalies",
            "recommendations": [
                "Examine the identified multivariate outliers in detail.",
                "Consider using robust scaling algorithms (e.g., RobustScaler).",
                "Evaluate outlier removal or robust regression techniques."
            ],
            "rationale": "Multivariate outliers represent unusual combinations of features that can severely bias distance-based models and distort predictions."
        },
        "outliers": {
            "title": "Address Univariate Outliers",
            "severity": "medium",
            "action": "investigate_remove",
            "recommendations": [
                "Investigate the root cause of outliers in this feature.",
                "Consider log or Box-Cox transformations for skewed distributions.",
                "Use robust scaling (median/IQR based) instead of standard scaling."
            ],
            "rationale": "High outlier percentages may indicate data quality issues, measurement errors, or extreme right-skew distributions that violate model assumptions."
        },
        "high_correlation": {
            "title": "Reduce Multicollinearity",
            "severity": "medium",
            "action": "reduce_multicollinearity",
            "recommendations": [
                "Drop one of the highly correlated features.",
                "Use PCA (Principal Component Analysis) to compress the feature space.",
                "Apply L1 (Lasso) or L2 (Ridge) regularization for coefficient shrinkage."
            ],
            "rationale": "Highly correlated features inflate model variance, make coefficients unstable, and render feature importance calculations unreliable."
        },
        "class_imbalance": {
            "title": "Handle Class Imbalance",
            "severity": "high",
            "action": "apply_balancing_techniques",
            "recommendations": [
                "Implement SMOTE (Synthetic Minority Over-sampling Technique).",
                "Configure class_weight='balanced' in the estimator.",
                "Evaluate using Precision/Recall, AUC-ROC, or F1-score instead of Accuracy."
            ],
            "rationale": "Imbalanced targets cause models to degenerate into predicting only the majority class, resulting in high nominal accuracy but poor minority class recall."
        },
        "missing_values": {
            "title": "Handle Missing Values",
            "severity": "medium",
            "action": "impute_or_drop",
            "recommendations": [
                "If missing completely at random: use median/mode imputation.",
                "If missing systematically: build a surrogate model for imputation.",
                "If > 50% missing: consider dropping the column entirely."
            ],
            "rationale": "Standard ML algorithms cannot process NaN values natively, and different imputation strategies have varying effects on model behavior."
        },
        "constant_feature": {
            "title": "Remove Constant Features",
            "severity": "high",
            "action": "drop_feature",
            "recommendations": [
                "Drop the feature prior to training.",
                "Verify that constant values are not encoding important metadata."
            ],
            "rationale": "Zero-variance features contain zero information and unnecessarily increase computational overhead without contributing to predictions."
        },
        "duplicate_rows": {
            "title": "Handle Duplicate Rows",
            "severity": "medium",
            "action": "drop_duplicates",
            "recommendations": [
                "Run df.drop_duplicates() early in the pipeline.",
                "Verify whether these duplicates are data quality issues or valid repeating events."
            ],
            "rationale": "Duplicate rows artificially over-weight specific data points, causing the model to overfit on repeated samples."
        }
    }
    
    def generate(self, checks_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable recommendations from diagnostic results."""
        issues = checks_output.get("issues", [])
        recommendations = []
        
        critical_count = 0
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for issue in issues:
            issue_type = issue["type"]
            
            # Start with basic recommendation structure
            rec = {
                "type": issue_type,
                "column": issue["column"],
                "severity": issue["severity"],
                "description": issue["description"]
            }
            
            # Add contextual information
            if issue_type in self.RECOMMENDATION_CONTEXT:
                context = self.RECOMMENDATION_CONTEXT[issue_type]
                rec.update({
                    "title": context["title"],
                    "action": context["action"],
                    "recommendations": context["recommendations"],
                    "rationale": context["rationale"]
                })
            else:
                # Fallback for unknown issue types
                rec.update({
                    "title": issue_type.replace('_', ' ').title(),
                    "action": "review_manually",
                    "recommendations": ["Conduct a manual review of this issue to determine impact."],
                    "rationale": "Data quality anomaly detected that requires investigation."
                })
            
            # Track severity
            if issue["severity"] == "critical":
                critical_count += 1
            severity_counts[issue["severity"]] += 1
            
            recommendations.append(rec)
        
        return {
            "recommendations": recommendations,
            "total_issues": len(issues),
            "critical_issues": critical_count,
            "severity_breakdown": severity_counts
        }