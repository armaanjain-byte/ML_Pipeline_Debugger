"""
NEW FILE: app/utils/logger.py

Structured logging for pipeline.
Singleton pattern for single logger instance.
"""

import logging
import sys
from typing import Optional


class PipelineLogger:
    """Singleton logger for pipeline"""
    
    _instance: Optional['PipelineLogger'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.logger = logging.getLogger("ml_pipeline")
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if self.logger.hasHandlers():
            self._initialized = True
            return
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self._initialized = True
    
    def get_logger(self):
        return self.logger


# Singleton instance
_logger_instance = PipelineLogger().get_logger()


def get_logger():
    """Get the pipeline logger"""
    return _logger_instance