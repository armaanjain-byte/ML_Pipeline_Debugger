"""
Observability utilities for ML Pipeline Debugger.
Produces conservative, rigorously calibrated reliability scoring, rationales, and readiness metrics.
"""

from typing import Dict, Any, List

class ReliabilityScorer:
    """
    Computes rigorous, trustworthy reliability scores.
    Features compounding penalties and catastrophic caps.
    """
    
    def compute_overall_score(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute heavily penalized overall reliability score."""
        components = self.compute_component_scores(issues)
        
        weights = {
            'Data Quality': 0.15,
            'Leakage Safety': 0.35,
            'Distribution Health': 0.15,
            'Feature Stability': 0.25,
            'Missingness Health': 0.10
        }
        
        overall = sum(components[name]["score"] * weight for name, weight in weights.items())
        
        # Catastrophic Failure Caps (Realism Enforcer)
        if components['Leakage Safety']['score'] < 50:
            overall = min(overall, 40.0)  # Leakage invalidates the entire pipeline
            
        if components['Feature Stability']['score'] < 50:
            overall = min(overall, 60.0)  # Massive drift, volatility, or VIF prevents safe deployment
            
        return {
            "score": max(0.0, min(100.0, overall)),
            "components": components
        }
    
    def compute_component_scores(self, issues: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Compute sub-scores with explicit rationales for dashboard explainability."""
        grouped = {"missing": [], "leakage": [], "stability": [], "quality": [], "distribution": []}
        
        for issue in issues:
            itype = issue.get("type", "")
            if itype in ["missing_values", "informative_missingness"]: grouped["missing"].append(issue)
            elif itype in ["target_leakage", "split_overlap"]: grouped["leakage"].append(issue)
            elif itype in ["feature_drift", "multicollinearity", "high_correlation", "feature_volatility"]: grouped["stability"].append(issue)
            elif itype in ["constant_feature", "near_constant_feature", "high_cardinality", "duplicate_rows"]: grouped["quality"].append(issue)
            elif itype in ["class_imbalance", "high_skewness", "outliers", "multivariate_outliers"]: grouped["distribution"].append(issue)

        return {
            'Data Quality': self._calculate_component(grouped["quality"], 100, 20, 10, "constants/duplicates"),
            'Leakage Safety': self._calculate_component(grouped["leakage"], 100, 60, 30, "leakage/overlap"),
            'Distribution Health': self._calculate_component(grouped["distribution"], 100, 15, 8, "imbalance/outliers"),
            'Feature Stability': self._calculate_component(grouped["stability"], 100, 30, 15, "drift/VIF/volatility"),
            'Missingness Health': self._calculate_component(grouped["missing"], 100, 35, 15, "informative nulls")
        }

    def _calculate_component(self, issue_group: List[Dict[str, Any]], base: float, crit_pen: float, high_pen: float, context: str) -> Dict[str, Any]:
        """Calculates compounding penalty and generates human-readable rationale."""
        score = base
        penalties = []
        
        for issue in issue_group:
            sev = issue.get("severity", "low")
            if sev == "critical":
                score -= crit_pen
                penalties.append(f"-{crit_pen} ({issue['type'].replace('_', ' ')})")
            elif sev == "high":
                score -= high_pen
                penalties.append(f"-{high_pen} ({issue['type'].replace('_', ' ')})")
            elif sev == "medium":
                score -= (high_pen / 2.0)
            else:
                score -= (high_pen / 4.0)
                
        score = max(0.0, min(100.0, score))
        
        if score == 100.0:
            rationale = f"Excellent. No penalizing {context} detected."
        else:
            rationale = f"Score reduced due to: {', '.join(penalties[:3])}" + ("..." if len(penalties) > 3 else "")
            
        return {"score": score, "rationale": rationale}

    def get_deployment_readiness(self, overall_score: float, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Executive summary assessing if the model should be shipped."""
        flags = metrics.get("observability_flags", [])
        
        if overall_score < 50 or any("Leakage" in f["flag"] for f in flags):
            return {
                "status": "DO NOT DEPLOY",
                "color": "#c0392b",
                "message": "Critical leakage, volatility, or structural flaws detected. Deployment will fail in production."
            }
        elif overall_score < 75 or flags:
            return {
                "status": "REQUIRES ENGINEERING REVIEW",
                "color": "#e67e22",
                "message": "Pipeline exhibits instability, drift, or overfitting. Monitor closely if deployed."
            }
        elif overall_score < 90:
            return {
                "status": "PRODUCTION READY (MONITORED)",
                "color": "#f39c12",
                "message": "Solid pipeline. Standard MLOps metric tracking recommended."
            }
        else:
            return {
                "status": "PRODUCTION READY (OPTIMIZED)",
                "color": "#27ae60",
                "message": "Excellent pipeline integrity, generalization capability, and feature stability."
            }

    def get_score_color(self, score: float) -> str:
        if score >= 90: return "#27ae60"  # Green
        elif score >= 75: return "#f39c12"  # Orange
        elif score >= 50: return "#e67e22"  # Dark Orange
        else: return "#c0392b"  # Red

    def get_health_status(self, score: float) -> str:
        if score >= 90: return "Excellent"
        elif score >= 75: return "Fair"
        elif score >= 50: return "Poor"
        else: return "Critical"