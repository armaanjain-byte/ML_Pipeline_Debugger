
import pandas as pd
from typing import Dict, Any
from app.core.exceptions import DataLoadException


class DataLoader:
    """Load and validate CSV data with proper error handling"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load_data(self) -> pd.DataFrame:
        """
        Load CSV file with comprehensive validation.
        
        Returns:
            pd.DataFrame: Loaded dataset
        
        Raises:
            DataLoadException: If file cannot be read or is invalid
        """
        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise DataLoadException(f"File not found: {self.file_path}")
        except pd.errors.EmptyDataError:
            raise DataLoadException(f"CSV file is empty: {self.file_path}")
        except Exception as e:
            raise DataLoadException(f"Error reading CSV: {str(e)}")
        
        # Validation checks
        if df.empty:
            raise DataLoadException("Loaded dataset has no rows")
        
        if df.shape[1] == 0:
            raise DataLoadException("Loaded dataset has no columns")
        
        return df
    
    def basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Return structured dataset metadata.
        
        Returns:
            dict: Metadata about the dataset
        """
        try:
            memory_usage = float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        except Exception:
            memory_usage = 0.0
        
        return {
            "num_rows": int(df.shape[0]),
            "num_columns": int(df.shape[1]),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": memory_usage,
            "missing_values": df.isnull().sum().to_dict()
        }