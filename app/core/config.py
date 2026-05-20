
from dataclasses import dataclass
from typing import Optional

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing step"""
    test_size: float = 0.2
    random_state: int = 42
    handle_missing_numeric: str = "median"  # median, mean, drop
    handle_missing_categorical: str = "mode"  # mode, unknown, drop
    scaling_method: str = "standard"  # standard, minmax, robust
    one_hot_sparse: bool = False


@dataclass
class ModelConfig:
    """Configuration for model training"""
    task_type: str = "classification"  # classification, regression
    model_type: str = "random_forest"  # random_forest, gradient_boosting
    random_state: int = 42
    n_estimators: int = 100
    max_depth: Optional[int] = None
    n_jobs: int = -1  # Parallel processing


@dataclass
class DataCheckConfig:
    """Configuration for data validation thresholds"""
    missing_threshold: float = 50.0  # Drop columns > 50% missing
    constant_threshold: int = 1  # Features with nunique <= this
    correlation_threshold: float = 0.9  # High correlation threshold
    imbalance_threshold: float = 0.3  # Minority class % threshold
    duplicate_check: bool = True
    LEAKAGE_THRESHOLD = 0.90  # Flag any feature with >90% correlation to target
    CORRELATION_THRESHOLD = 0.85 # Flag multicollinearity between features


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None  # None = console only


@dataclass
class PipelineConfig:
    """Full pipeline configuration"""
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    model: ModelConfig = ModelConfig()
    checks: DataCheckConfig = DataCheckConfig()
    logging: LoggingConfig = LoggingConfig()
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Load from dictionary (useful for YAML/JSON configs)"""
        return cls(
            preprocessing=PreprocessingConfig(**config_dict.get("preprocessing", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            checks=DataCheckConfig(**config_dict.get("checks", {})),
            logging=LoggingConfig(**config_dict.get("logging", {}))
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "preprocessing": self.preprocessing.__dict__,
            "model": self.model.__dict__,
            "checks": self.checks.__dict__,
            "logging": self.logging.__dict__
        }


# Default singleton instance
DEFAULT_CONFIG = PipelineConfig()

DiagnosticConfig = DataCheckConfig
