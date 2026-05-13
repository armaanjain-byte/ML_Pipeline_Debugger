"""
NEW FILE: app/core/exceptions.py

Custom exceptions for graceful error handling.
Allows pipeline to fail cleanly with structured messages.
"""

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