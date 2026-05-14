# app/core/exceptions.py

from typing import Any, Dict, Optional


class PipelineDiagnosticError(Exception):
    """Exception raised when the pipeline fails but has diagnostic data."""
    def __init__(self, message: str, diagnostics: Optional[Dict[str, Any]]= None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}




class PipelineException(Exception):
    """Base exception for all pipeline errors"""
    pass


class DataLoadException(PipelineException):
    """Raised when data loading fails"""
    pass


class DataValidationException(PipelineException):
    """Raised when data validation fails"""
    pass


class PreprocessingException(PipelineException):
    """Raised during preprocessing"""
    pass


class ModelTrainingException(PipelineException):
    """Raised during model training"""
    pass


class InvalidTaskTypeException(PipelineException):
    """Raised for invalid task type"""
    pass


class ConfigException(PipelineException):
    """Raised for configuration errors"""
    pass
