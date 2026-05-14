from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Dict, Any
import numpy as np

from app.pipeline.data_loader import DataLoader
from app.pipeline.preprocessing import Preprocessor
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
    Orchestrates full ML pipeline:
    1. Load data
    2. Run checks (BEFORE preprocessing)
    3. Generate recommendations
    4. Preprocess (with NO leakage)
    5. Train model
    6. Evaluate
    7. Return structured analysis
    """
    
    def __init__(
        self,
        file_path: str,
        target_column: str,
        task_type: str,
        config: PipelineConfig = None,
        dev_mode: bool = False
    ):
        """
        Initialize pipeline.
        
        Args:
            file_path: Path to CSV file
            target_column: Name of target variable column
            task_type: 'classification' or 'regression'
            config: PipelineConfig object (uses defaults if None)
        """
        self.file_path = file_path
        self.target_column = target_column
        self.task_type = task_type
        self.config = config or PipelineConfig()
        self.dev_mode = dev_mode
        # Initialize components
        self.loader = DataLoader(file_path)
        self.preprocessor = Preprocessor(
            target_column=target_column,
            test_size=self.config.preprocessing.test_size,
            random_state=self.config.preprocessing.random_state
        )
        self.model = Model(task_type=self.task_type, dev_mode=self.dev_mode)
        self.checker = DataChecks()
        self.recommender = RecommendationEngine()
        
        logger.info(f"PipelineRunner initialized: task_type={task_type}, target={target_column}")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute full pipeline.
        
        Returns:
            dict: {
                "status": "success" | "failure",
                "metadata": {...},
                "checks": {...},
                "recommendations": {...},
                "model_metrics": {...},
                "feature_importance": {...},
                "error": None | str
            }
        """
        try:
            logger.info("=" * 50)
            logger.info("PIPELINE START")
            logger.info("=" * 50)
            
           # Step 1: Load data
            logger.info("[Step 1/8] Loading data...")
            df = self._load_data()
            
            if self.dev_mode:
                if len(df) > 5000:
                    logger.warning("⚙️ DEV MODE ACTIVE: Downsampling to 5000 rows for rapid testing.")
                    df = df.sample(n=5000, random_state=42)
                else:
                    logger.info("⚙️ DEV MODE ACTIVE: Dataset is already under 5000 rows.")
                
            # Calculate metadata AFTER sampling
            metadata = self.loader.basic_info(df)
            
            logger.info(
                f"✓ Loaded {metadata['num_rows']} rows, {metadata['num_columns']} columns"
            )
            
           
            # Separate features and target first
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            
            # Split BEFORE diagnostics
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if self.task_type == "classification" else None)
            
            # Step 2: Run checks on TRAINING data only
            logger.info("[Step 2/8] Running data quality checks on training data...")
            
            # Recombine training data purely for the diagnostic engine
            train_df = pd.concat([X_train, y_train], axis=1)
            
            # Pass ONLY train_df to the checker
            checks_output = self.checker.run_all_checks(train_df, self.target_column, self.task_type)
            num_issues = len(checks_output.get("issues", []))
            logger.info(f"✓ Found {num_issues} issues")
            
            # Step 3: Generate recommendations
            logger.info("[Step 3/8] Generating recommendations...")
            recommendations = self.recommender.generate(checks_output)
            num_recs = len(recommendations.get("recommendations", []))
            logger.info(f"✓ Generated {num_recs} recommendations")
            
            # Step 4: Preprocess (NO LEAKAGE)
            logger.info("[Step 4/8] Preprocessing data...")
            self.model.train(X_train, y_train)
            X_train, X_test, y_train, y_test, feature_names = self._preprocess(df)
            logger.info(
                f"✓ Train: {X_train.shape}, Test: {X_test.shape}, Features: {len(feature_names)}"
            )
            
            # Step 5: Train model
            logger.info("[Step 5/8] Training model...")
            self._train_model(X_train, y_train)
            logger.info("✓ Model training complete")
            
            # Step 6: Make predictions
            logger.info("[Step 6/8] Making predictions...")
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)
            logger.info(f"✓ Predictions made on {len(y_pred)} test samples")
            
            # Step 7: Evaluate
            logger.info("[Step 7/8] Evaluating model...")
            
            # Standard holdout evaluation
            metrics = self.model.evaluate(y_test, y_pred, y_pred_proba)
            
            # K-Fold Cross Validation
            cv_metrics = self.model.cross_validate(X_train, y_train)
            metrics.update(cv_metrics)  # Merge CV metrics into main metrics
            
            logger.info(f"✓ Metrics computed: {list(metrics.keys())}")
            # Step 8: Feature importance
            logger.info("[Step 8/8] Computing feature importance...")
            feature_importance = self.model.feature_importance(feature_names)
            top_5 = list(feature_importance.items())[:5]
            logger.info(f"✓ Top features: {[f[0] for f in top_5]}")
            
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
            logger.error(f"DATA LOAD FAILED: {str(e)}")
            return self._failure_response(f"Data loading failed: {str(e)}")
        
        except PreprocessingException as e:
            logger.error(f"PREPROCESSING FAILED: {str(e)}")
            return self._failure_response(f"Preprocessing failed: {str(e)}")
        
        except ModelTrainingException as e:
            logger.error(f"MODEL TRAINING FAILED: {str(e)}")
            return self._failure_response(f"Model training failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"UNEXPECTED ERROR: {str(e)}", exc_info=True)
            return self._failure_response(f"Unexpected error: {str(e)}")
    
    def _load_data(self) -> Any:
        """Load data with error handling"""
        try:
            return self.loader.load_data()
        except DataLoadException:
            raise
        except Exception as e:
            raise DataLoadException(f"Failed to load data: {str(e)}")
    
    def _run_checks(self, df: Any) -> Dict[str, Any]:
        """Run all data checks"""
        try:
            # Added self.task_type to the arguments here
            return self.checker.run_all_checks(df, self.target_column, self.task_type)
        except Exception as e:
            logger.warning(f"Data checks failed (non-fatal): {str(e)}")
            return {"issues": [], "error": str(e)}
    def _preprocess(self, df: Any) -> tuple:
        """Preprocess data with error handling"""
        try:
            return self.preprocessor.preprocess(df)
        except ValueError as e:
            raise PreprocessingException(f"Preprocessing failed: {str(e)}")
        except Exception as e:
            raise PreprocessingException(f"Unexpected preprocessing error: {str(e)}")
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train model with error handling"""
        try:
            self.model.train(X_train, y_train)
        except Exception as e:
            raise ModelTrainingException(f"Model training failed: {str(e)}")
    
    def _failure_response(self, error_msg: str) -> Dict[str, Any]:
        """Return standardized failure response"""
        return {
            "status": "failure",
            "error": error_msg,
            "metadata": None,
            "checks": None,
            "recommendations": None,
            "model_metrics": None,
            "feature_importance": None
        }