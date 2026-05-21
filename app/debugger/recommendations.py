from typing import Dict, Any, List


class RecommendationEngine:
    """
    Generates actionable engineering recommendations
    based on diagnostic findings.
    """

    def generate(
        self,
        checks_output: Dict[str, Any]
    ) -> Dict[str, Any]:

        issues = checks_output.get(
            "issues",
            []
        )

        recommendations = []

        critical_count = 0

        for issue in issues:

            rec = {

                "type":
                    issue.get(
                        "type",
                        "unknown"
                    ),

                "column":
                    issue.get(
                        "column",
                        "unknown"
                    ),

                "severity":
                    issue.get(
                        "severity",
                        "low"
                    ),

                "description":
                    issue.get(
                        "description",
                        ""
                    )
            }

            if rec["severity"] == "critical":

                critical_count += 1

            # =================================================
            # TARGET LEAKAGE
            # =================================================

            if rec["type"] == "target_leakage":

                rec["action"] = (
                    "remove_leaking_feature"
                )

                rec["recommendations"] = [

                    "Drop the feature immediately.",

                    "Verify the feature is available "
                    "at prediction time.",

                    "Re-train the model after removal."
                ]

                rec["rationale"] = (
                    "Target leakage causes "
                    "artificially inflated performance."
                )

            # =================================================
            # MULTIVARIATE OUTLIERS
            # =================================================

            elif rec["type"] == "multivariate_outliers":

                rec["action"] = (
                    "investigate_outliers"
                )

                rec["recommendations"] = [

                    "Inspect anomalous samples.",

                    "Use RobustScaler.",

                    "Consider IsolationForest filtering."
                ]

                rec["rationale"] = (
                    "Multivariate anomalies may "
                    "destabilize model behavior."
                )

            # =================================================
            # HIGH CARDINALITY
            # =================================================

            elif rec["type"] == "high_cardinality":

                rec["action"] = (
                    "reduce_cardinality"
                )

                rec["recommendations"] = [

                    "Use target encoding.",

                    "Remove identifier-like columns.",

                    "Use frequency encoding."
                ]

                rec["rationale"] = (
                    "Very high cardinality may "
                    "cause sparse representations."
                )

            # =================================================
            # HIGH CORRELATION
            # =================================================

            elif rec["type"] == "high_correlation":

                rec["action"] = (
                    "reduce_multicollinearity"
                )

                rec["recommendations"] = [

                    "Drop redundant features.",

                    "Use PCA if needed.",

                    "Use regularization."
                ]

                rec["rationale"] = (
                    "Highly correlated features "
                    "inflate variance."
                )

            # =================================================
            # CLASS BALANCE
            # =================================================

            elif rec["type"] == "class_balance":

                rec["action"] = (
                    "balance_target_distribution"
                )

                rec["recommendations"] = [

                    "Use SMOTE.",

                    "Use class weights.",

                    "Monitor F1-score."
                ]

                rec["rationale"] = (
                    "Imbalanced targets bias "
                    "the model toward majority classes."
                )

            # =================================================
            # MISSING VALUES
            # =================================================

            elif rec["type"] == "missing_data":

                rec["action"] = (
                    "impute_missing_values"
                )

                rec["recommendations"] = [

                    "Use median imputation.",

                    "Drop heavily missing columns.",

                    "Validate upstream data."
                ]

                rec["rationale"] = (
                    "Missing values reduce "
                    "model reliability."
                )

            # =================================================
            # CONSTANT FEATURES
            # =================================================

            elif rec["type"] == "constant_feature":

                rec["action"] = (
                    "drop_constant_feature"
                )

                rec["recommendations"] = [

                    "Remove the feature before training."
                ]

                rec["rationale"] = (
                    "Constant features add "
                    "no predictive value."
                )

            # =================================================
            # DUPLICATES
            # =================================================

            elif rec["type"] == "duplicate_rows":

                rec["action"] = (
                    "remove_duplicates"
                )

                rec["recommendations"] = [

                    "Run df.drop_duplicates().",

                    "Validate ingestion pipeline."
                ]

                rec["rationale"] = (
                    "Duplicate rows bias training."
                )

            # =================================================
            # DEFAULT
            # =================================================

            else:

                rec["action"] = (
                    "manual_review"
                )

                rec["recommendations"] = [

                    "Review issue manually."
                ]

                rec["rationale"] = (
                    "General anomaly detected."
                )

            recommendations.append(rec)

        return {

            "recommendations":
                recommendations,

            "critical_issues":
                critical_count,

            "total_issues":
                len(issues)
        }