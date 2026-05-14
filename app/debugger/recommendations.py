from typing import Dict, Any, List

class RecommendationEngine:
    """Generates actionable engineering recommendations based on diagnostic issues."""
    
    def generate(self, checks_output: Dict[str, Any]) -> Dict[str, Any]:
        issues = checks_output.get("issues", [])
        recommendations = []
        
        critical_count = 0
        
        for issue in issues:
            rec = {
                "type": issue["type"],
                "column": issue["column"],
                "severity": issue["severity"],
                "description": issue["description"]
            }
            
            if issue["severity"] == "critical":
                critical_count += 1
            
            if issue["type"] == "target_leakage":
                rec["action"] = "remove_leaking_feature"
                rec["recommendations"] = [
                    "Drop the feature from the dataset immediately.",
                    "Verify if this data point is actually available at prediction time in production."
                ]
                rec["rationale"] = "Features nearly identical to the target indicate data leakage, resulting in artificially high accuracy during training but failure in production."

            elif issue["type"] == "multivariate_outliers":
                rec["action"] = "investigate_anomalies"
                rec["recommendations"] = [
                    "Investigate the root cause of these specific outliers.",
                    "Consider robust scaling algorithms (e.g., RobustScaler)."
                ]
                rec["rationale"] = "Multivariate outliers represent strange combinations of normal data that can severely skew distance-based models."

            elif issue["type"] == "outliers":
                rec["action"] = "investigate_remove"
                rec["recommendations"] = [
                    "Investigate root cause of outliers.",
                    "Consider log/box-cox transformation.",
                    "Use robust scaling (median/IQR based)."
                ]
                rec["rationale"] = "High outlier percentages may indicate data quality issues or extreme right-skew distributions."

            elif issue["type"] == "high_correlation":
                rec["action"] = "reduce_multicollinearity"
                rec["recommendations"] = [
                    "Drop one of the highly correlated features.",
                    "Use PCA (Principal Component Analysis) to compress the feature space.",
                    "Apply L1 (Lasso) or L2 (Ridge) regularization."
                ]
                rec["rationale"] = "Highly correlated features inflate model variance and render feature importance calculations unreliable."

            elif issue["type"] == "class_imbalance":
                rec["action"] = "apply_balancing_techniques"
                rec["recommendations"] = [
                    "Implement SMOTE (Synthetic Minority Over-sampling Technique).",
                    "Configure the `class_weight='balanced'` parameter in the estimator.",
                    "Evaluate using Precision/Recall AUC or F1-score instead of standard Accuracy."
                ]
                rec["rationale"] = "Imbalanced targets cause models to degenerate into predicting the majority class exclusively."

            elif issue["type"] == "missing_values":
                rec["action"] = "impute_or_drop"
                rec["recommendations"] = [
                    "If missing completely at random, utilize median/mode imputation via ColumnTransformer.",
                    "If missing systematically, consider building a surrogate model for imputation.",
                    "Drop the column entirely if > 50% of the data is missing."
                ]
                rec["rationale"] = "Standard ML algorithms lack native support for processing NaN structures."

            elif issue["type"] == "constant_feature":
                rec["action"] = "drop_feature"
                rec["recommendations"] = [
                    "Drop the feature prior to training."
                ]
                rec["rationale"] = "Zero-variance features contain zero information and unnecessarily increase computational overhead."

            elif issue["type"] == "duplicate_rows":
                rec["action"] = "drop_duplicates"
                rec["recommendations"] = [
                    "Run `df.drop_duplicates()` early in the pipeline.",
                    "Verify if these duplicates are a result of a messy SQL join or mathematically valid repeating events."
                ]
                rec["rationale"] = "Duplicate rows artificially over-weight specific data points, causing the model to overfit."

            else:
                rec["action"] = "review_manually"
                rec["recommendations"] = ["Conduct a manual review of this feature to determine downstream impact."]
                rec["rationale"] = "General data anomaly detected."
                
            # Ensures EVERY issue caught gets appended to the JSON
            recommendations.append(rec)
            
        return {
            "recommendations": recommendations,
            "total_issues": len(issues),
            "critical_issues": critical_count
        }