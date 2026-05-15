from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Dict, Any, List
import numpy as np

from app.pipeline.data_loader import DataLoader
from app.pipeline.model import Model
from app.debugger.data_checks import DataChecks
from app.debugger.recommendations import RecommendationEngine
from app.core.config import PipelineConfig
from app.core.exceptions import (
    DataLoadException,
    PreprocessingException,
    ModelTrainingException
)
from app.utils.logger import get_logger

logger = get_logger()

class PipelineRunner:
    """
    Orchestrates the full ML pipeline with zero-leakage architecture:
    1. Loads and samples data (if dev_mode is active).
    2. Performs a single train/test split (The Single Source of Truth).
    3. Runs diagnostics strictly on the training set.
    4. Trains a robust pipeline (scaling/encoding) without data leakage.
    5. Evaluates using holdout and cross-validation metrics.
    """
    
    def __init__(
        self,
        file_path: str,
        target_column: str,
        task_type: str,
        config: PipelineConfig = None,
        dev_mode: bool = False
    ):
        self.file_path = file_path
        self.target_column = target_column
        self.task_type = task_type
        self.config = config or PipelineConfig()
        self.dev_mode = dev_mode
        
        # Initialize components
        self.loader = DataLoader(file_path)
        self.model = Model(task_type=self.task_type, dev_mode=self.dev_mode)
        self.checker = DataChecks()
        self.recommender = RecommendationEngine()
        
        logger.info(f"PipelineRunner initialized: task_type={task_type}, target={target_column}, dev_mode={dev_mode}")
    
    def run(self) -> Dict[str, Any]:
        try:
            logger.info("=" * 50)
            logger.info("PIPELINE START")
            logger.info("=" * 50)
            
            # Step 1: Load and Sample
            logger.info("[Step 1/8] Loading data...")
            df = self.loader.load_data()
            if df is None or df.empty:
                raise DataLoadException("Dataset is empty or could not be loaded.")
            
            if self.dev_mode and len(df) > 5000:
                logger.warning("⚙️ DEV MODE ACTIVE: Downsampling to 5000 rows.")
                df = df.sample(n=5000, random_state=42)
                
            metadata = self.loader.basic_info(df)
            logger.info(f"✓ Data loaded: {metadata['num_rows']} rows")

            # Step 2: The Single Source of Truth Split (Zero-Leakage)
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            
            # Split BEFORE any diagnostics or transformations
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.preprocessing.test_size, 
                random_state=self.config.preprocessing.random_state,
                stratify=y if self.task_type == "classification" else None
            )
            
            # Step 3: Run Checks on TRAINING data only
            logger.info("[Step 2/8] Running diagnostics on training set...")
            train_df = pd.concat([X_train, y_train], axis=1)
            checks_output = self.checker.run_all_checks(train_df, self.target_column, self.task_type)
            logger.info(f"✓ Found {len(checks_output.get('issues', []))} issues")
            
            # Step 4: Generate Recommendations
            logger.info("[Step 3/8] Generating recommendations...")
            recommendations = self.recommender.generate(checks_output)
            
            # Step 5: Training (Internal Scaling & Encoding)
            logger.info("[Step 4/8] Training robust model pipeline...")
            feature_names = X_train.columns.tolist()
            self.model.train(X_train, y_train)
            logger.info(f"✓ Model trained on {X_train.shape[1]} features")
            
            # Step 6: Prediction
            logger.info("[Step 5/8] Making predictions on holdout set...")
            y_pred = self.model.predict(X_test)
            
            # Step 7: Comprehensive Evaluation
            logger.info("[Step 6/8] Evaluating model performance...")
            # Holdout Metrics
            metrics = self.model.evaluate(y_test, y_pred)
            
            # K-Fold Cross Validation
            cv_results = self.model.cross_validate(X_train, y_train)
            metrics.update(cv_results)
            logger.info(f"✓ Metrics: {list(metrics.keys())}")
            
            # Step 8: Feature Importance
            logger.info("[Step 7/8] Computing feature importance...")
            # Note: Ensure your Model class has the feature_importance method added
            try:
                feature_importance = self.model.feature_importance(feature_names)
            except AttributeError:
                logger.warning("Feature importance method not found in Model class.")
                feature_importance = {}

            logger.info("=" * 50)
            logger.info("PIPELINE SUCCESS")
            logger.info("=" * 50)
            
            return {
                "status": "success",
                "metadata": metadata,
                "checks": checks_output,
                "recommendations": recommendations,
                "model_metrics": metrics,
                "feature_importance": feature_importance,
                "error": None
            }
        
        except DataLoadException as e:
            logger.error(f"Data Load Failure: {str(e)}")
            return self._failure_response(f"Data Error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected Pipeline Failure: {str(e)}", exc_info=True)
            return self._failure_response(f"System Error: {str(e)}")

    def _failure_response(self, error_msg: str) -> Dict[str, Any]:
        return {
            "status": "failure",
            "error": error_msg,
            "metadata": None,
            "checks": None,
            "recommendations": None,
            "model_metrics": None,
            "feature_importance": None
        }
    

def _apply_auto_fixes(self, df: pd.DataFrame, diagnostics: Dict[str, Any]) -> pd.DataFrame:
    """Automatically removes problematic features identified by DataChecks."""
    cols_to_drop = set()

    # 1. Drop Target Leakage
    for leak in diagnostics.get("target_leakage", []):
        cols_to_drop.add(leak["column"])
        self.logger.warning(f"AUTO-FIX: Dropping {leak['column']} due to target leakage.")

    # 2. Drop Constant Features
    for col in diagnostics.get("constant_features", []):
        cols_to_drop.add(col)
        self.logger.warning(f"AUTO-FIX: Dropping {col} - feature is constant.")

    # 3. Handle High Correlation (Drop the second column in the pair)
    for col1, col2, corr in diagnostics.get("high_correlation", []):
        if col2 not in cols_to_drop:
            cols_to_drop.add(col2)
            self.logger.warning(f"AUTO-FIX: Dropping {col2} - highly correlated with {col1}.")

    return df.drop(columns=list(cols_to_drop))