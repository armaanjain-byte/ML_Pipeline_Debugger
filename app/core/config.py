from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing step"""

    test_size: float = 0.2
    random_state: int = 42

    handle_missing_numeric: str = "median"
    handle_missing_categorical: str = "mode"

    scaling_method: str = "standard"

    one_hot_sparse: bool = False


@dataclass
class ModelConfig:
    """Configuration for model training"""

    task_type: str = "classification"

    model_type: str = "random_forest"

    random_state: int = 42

    n_estimators: int = 100

    max_depth: Optional[int] = None

    n_jobs: int = -1


@dataclass
class DataCheckConfig:
    """Configuration for data validation thresholds"""

    missing_threshold: float = 50.0

    constant_threshold: int = 1

    correlation_threshold: float = 0.9

    imbalance_threshold: float = 0.3

    duplicate_check: bool = True

    LEAKAGE_THRESHOLD: float = 0.90

    CORRELATION_THRESHOLD: float = 0.85


@dataclass
class LoggingConfig:
    """Configuration for logging"""

    level: str = "INFO"

    format: str = (
        "%(asctime)s - %(name)s - "
        "%(levelname)s - %(message)s"
    )

    log_file: Optional[str] = None


@dataclass
class PipelineConfig:
    """Full pipeline configuration"""

    preprocessing: PreprocessingConfig = field(
        default_factory=PreprocessingConfig
    )

    model: ModelConfig = field(
        default_factory=ModelConfig
    )

    checks: DataCheckConfig = field(
        default_factory=DataCheckConfig
    )

    logging: LoggingConfig = field(
        default_factory=LoggingConfig
    )

    @classmethod
    def from_dict(
        cls,
        config_dict: dict
    ):
        """Load config from dictionary"""

        return cls(

            preprocessing=PreprocessingConfig(
                **config_dict.get(
                    "preprocessing",
                    {}
                )
            ),

            model=ModelConfig(
                **config_dict.get(
                    "model",
                    {}
                )
            ),

            checks=DataCheckConfig(
                **config_dict.get(
                    "checks",
                    {}
                )
            ),

            logging=LoggingConfig(
                **config_dict.get(
                    "logging",
                    {}
                )
            )
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary"""

        return {

            "preprocessing":
                self.preprocessing.__dict__,

            "model":
                self.model.__dict__,

            "checks":
                self.checks.__dict__,

            "logging":
                self.logging.__dict__
        }


# Alias for backward compatibility
DiagnosticConfig = DataCheckConfig


# Default singleton config
DEFAULT_CONFIG = PipelineConfig()