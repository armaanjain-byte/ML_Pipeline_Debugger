

from typing import Dict, List, Any


class RecommendationEngine:
    """
    Converts detected data issues into actionable recommendations.
    Rule-based system that generates fix suggestions.
    """
    
    def generate(self, checks_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for generating recommendations.
        
        Args:
            checks_output: Output from DataChecks.run_all_checks()
        
        Returns:
            dict: recommendations and priority-sorted list
        """
        recommendations = []
        issues = checks_output.get("issues", [])
        
        for issue in issues:
            rec = self._generate_for_issue(issue)
            if rec:
                recommendations.append(rec)
        
        # Sort by severity
        sorted_recs = self._prioritize(recommendations)
        
        return {
            "recommendations": sorted_recs,
            "total_issues": len(issues),
            "critical_issues": sum(1 for r in sorted_recs if r.get("severity") == "high")
        }
    
    def _generate_for_issue(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendation for single issue"""
        issue_type = issue.get("type")
        
        if issue_type == "missing_values":
            return self._recommend_missing(issue)
        elif issue_type == "constant_feature":
            return self._recommend_constant(issue)
        elif issue_type == "duplicate_rows":
            return self._recommend_duplicates(issue)
        elif issue_type == "class_imbalance":
            return self._recommend_imbalance(issue)
        elif issue_type == "high_correlation":
            return self._recommend_correlation(issue)
        elif issue_type == "outliers":
            return self._recommend_outliers(issue)
        
        return None
    
    def _recommend_missing(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend strategy for missing values.
        Strategy depends on percentage missing.
        """
        desc = issue.get("description", "")
        pct = float(desc.split("%")[0]) if "%" in desc else 0
        
        if pct > 50:
            action = "drop_column"
            rationale = "Too much data missing to impute reliably. Dropping column recommended."
            urgency = "high"
        elif pct > 20:
            action = "advanced_imputation"
            rationale = "Use KNN or iterative imputation instead of simple median/mode."
            urgency = "medium"
        else:
            action = "simple_imputation"
            rationale = "Simple imputation (median/mode) is sufficient for < 20% missing."
            urgency = "low"
        
        return {
            **issue,
            "action": action,
            "rationale": rationale,
            "urgency": urgency
        }
    
    def _recommend_constant(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend dropping constant features"""
        return {
            **issue,
            "action": "drop_column",
            "rationale": "Features with constant values add no predictive power. Drop immediately.",
            "urgency": "high"
        }
    
    def _recommend_duplicates(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend handling duplicates"""
        return {
            **issue,
            "action": "remove_duplicates",
            "rationale": "Exact duplicate rows inflate metrics and introduce information leakage.",
            "urgency": "high"
        }
    
    def _recommend_imbalance(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend strategy for imbalanced data"""
        return {
            **issue,
            "action": "apply_balancing",
            "recommendations": [
                "Use SMOTE (Synthetic Minority Over-sampling)",
                "Use class_weight parameter in model (automatic balancing)",
                "Use stratified cross-validation",
                "Monitor F1-score instead of accuracy"
            ],
            "rationale": "Imbalanced classes lead to biased models favoring majority class.",
            "urgency": "high"
        }
    
    def _recommend_correlation(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend handling high correlation"""
        col_pair = issue.get("column", "Unknown")
        corr_val = issue.get("description", "").split(":")[-1] if ":" in issue.get("description", "") else "high"
        
        return {
            **issue,
            "action": "reduce_multicollinearity",
            "recommendations": [
                f"Drop one of the correlated features",
                "Use PCA (Principal Component Analysis)",
                "Use regularization (L1/L2)"
            ],
            "rationale": f"Multicollinearity ({corr_val}) increases model complexity without benefit.",
            "urgency": "medium"
        }
    
    def _recommend_outliers(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend handling outliers"""
        desc = issue.get("description", "")
        pct = float(desc.split("(")[1].split("%")[0]) if "(" in desc else 0
        
        if pct > 10:
            action = "investigate_remove"
            rationale = f"High outlier percentage ({pct:.1f}%) may indicate data quality issues."
        else:
            action = "transform_or_robust"
            rationale = "Consider log transformation or robust scaling to reduce outlier impact."
        
        return {
            **issue,
            "action": action,
            "recommendations": [
                "Investigate root cause of outliers",
                "Consider log/box-cox transformation",
                "Use robust scaling (median/IQR based)"
            ],
            "rationale": rationale,
            "urgency": "medium"
        }
    
    def _prioritize(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort recommendations by severity (high → medium → low)"""
        severity_order = {"high": 0, "medium": 1, "low": 2}
        
        return sorted(
            recommendations,
            key=lambda x: severity_order.get(x.get("severity"), 99)
        )