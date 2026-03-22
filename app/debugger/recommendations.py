from typing import Dict, List, Any


class RecommendationEngine:
    """
    Converts detected data issues into actionable recommendations.
    This module is purely rule-based for now.
    """

    def __init__(self):
        pass

    def generate(self, checks_output: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        """
        Main entry point.

        Parameters:
            checks_output: dict returned from data_checks module

        Returns:
            dict:
                {
                    "recommendations": [
                        {
                            "issue_type": str,
                            "feature": str,
                            "problem": str,
                            "recommendation": str
                        }
                    ]
                }
        """

        recommendations = []

        # Missing Values
        recommendations.extend(
            self._handle_missing(checks_output.get("missing", {}))
        )

        # Class Imbalance
        imbalance = checks_output.get("imbalance")
        if imbalance:
            recommendations.append(
                self._handle_imbalance(imbalance)
            )

        # Constant Features
        recommendations.extend(
            self._handle_constant(checks_output.get("constant", []))
        )

        # Correlation
        recommendations.extend(
            self._handle_correlation(checks_output.get("correlation", []))
        )

        return {"recommendations": recommendations}

    def _handle_missing(self, missing: Dict[str, float]) -> List[Dict[str, str]]:
        results = []

        for col, pct in missing.items():
            if pct > 0:
                if pct > 50:
                    rec = "Consider dropping this feature due to excessive missing values."
                elif pct > 20:
                    rec = "Consider advanced imputation (KNN, model-based) instead of simple imputation."
                else:
                    rec = "Simple imputation (mean/median/mode) is sufficient."

                results.append({
                    "issue_type": "missing_values",
                    "feature": col,
                    "problem": f"{pct:.2f}% missing values",
                    "recommendation": rec
                })

        return results

    def _handle_imbalance(self, imbalance: Dict[str, Any]) -> Dict[str, str]:
        ratio = imbalance.get("ratio")

        if ratio and ratio < 0.5:
            rec = "Apply resampling techniques such as SMOTE, undersampling, or class weighting."
        else:
            rec = "Imbalance is mild. Monitor performance metrics like F1-score instead of accuracy."

        return {
            "issue_type": "class_imbalance",
            "feature": imbalance.get("target_column", "target"),
            "problem": f"Class ratio = {ratio}",
            "recommendation": rec
        }

    def _handle_constant(self, constant_cols: List[str]) -> List[Dict[str, str]]:
        return [
            {
                "issue_type": "constant_feature",
                "feature": col,
                "problem": "Feature has constant value",
                "recommendation": "Drop this feature as it adds no predictive value."
            }
            for col in constant_cols
        ]

    def _handle_correlation(self, correlated_pairs: List[tuple]) -> List[Dict[str, str]]:
        results = []

        for col1, col2 in correlated_pairs:
            results.append({
                "issue_type": "high_correlation",
                "feature": f"{col1}, {col2}",
                "problem": "Highly correlated features",
                "recommendation": f"Consider dropping one of '{col1}' or '{col2}' to reduce multicollinearity."
            })

        return results