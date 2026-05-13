from typing import Dict, Any

from app.pipeline.data_loader import DataLoader
from app.pipeline.preprocessing import Preprocessor
from app.pipeline.model import Model
from app.debugger.data_checks import DataChecks
from app.debugger.recommendations import RecommendationEngine


class PipelineRunner:
    """
    Orchestrates the full ML pipeline:
    - Load data
    - Run data checks
    - Preprocess data
    - Train model
    - Evaluate model
    - Generate recommendations
    """

    def __init__(self, file_path: str, target_column: str, task_type: str):
        """
        Parameters:
            file_path: path to dataset
            target_column: name of target variable
            task_type: 'classification' or 'regression'
        """
        self.file_path = file_path
        self.target_column = target_column
        self.task_type = task_type

        # Initialize components
        self.loader = DataLoader(file_path)
        self.preprocessor = Preprocessor(target_column)
        self.model = Model(task_type)
        self.checker = DataChecks(target_column)
        self.recommender = RecommendationEngine()

    def run(self) -> Dict[str, Any]:
        """
        Executes full pipeline

        Returns:
            dict:
                {
                    "metadata": {},
                    "checks": {},
                    "recommendations": {},
                    "model_metrics": {},
                    "feature_importance": {}
                }
        """

        # Step 1: Load data
        data_output = self.loader.load()
        df = data_output["data"]
        metadata = data_output["metadata"]

        # Step 2: Run checks BEFORE preprocessing
        checks_output = self.checker.run_checks(df)

        # Step 3: Generate recommendations
        recommendations = self.recommender.generate(checks_output)

        # Step 4: Preprocess
        X_train, X_test, y_train, y_test = self.preprocessor.process(df)

        # Step 5: Train model
        self.model.train(X_train, y_train)

        # Step 6: Predict
        predictions = self.model.predict(X_test)

        # Step 7: Evaluate
        metrics = self.model.evaluate(y_test, predictions)

        # Step 8: Feature importance
        feature_importance = self.model.feature_importance()

        return {
            "metadata": metadata,
            "checks": checks_output,
            "recommendations": recommendations,
            "model_metrics": metrics,
            "feature_importance": feature_importance
        }