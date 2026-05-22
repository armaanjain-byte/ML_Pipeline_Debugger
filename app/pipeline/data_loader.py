import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
from app.core.exceptions import DataLoadException

logger = logging.getLogger(__name__)

class DataLoader:
    """Load and validate CSV data with production-grade datatype inference and error handling."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.coercion_audit: List[Dict[str, str]] = []
    
    def load_data(self) -> pd.DataFrame:
        """
        Load CSV file with comprehensive validation and intelligent type coercion.
        """
        try:
            df = pd.read_csv(self.file_path, low_memory=False)
        except FileNotFoundError:
            raise DataLoadException(f"File not found: {self.file_path}")
        except pd.errors.EmptyDataError:
            raise DataLoadException(f"CSV file is empty: {self.file_path}")
        except Exception as e:
            raise DataLoadException(f"Error reading CSV: {str(e)}")
        
        if df.empty:
            raise DataLoadException("Loaded dataset has no rows.")
        if df.shape[1] == 0:
            raise DataLoadException("Loaded dataset has no columns.")
            
        # 1. Clean Column Names
        df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)
        
        # 2. Standardize Missing Values
        df = df.replace(r'^\s*$', np.nan, regex=True)
        df = df.replace(['?', 'NA', 'N/A', 'null', 'NULL', 'None'], np.nan)

        # 3. Intelligent Numeric Coercion
        for col in df.select_dtypes(include=['object']):
            num_valid_strings = df[col].notna().sum()
            if num_valid_strings == 0:
                continue
                
            # Strip common non-numeric characters safely
            cleaned_series = df[col].astype(str).str.replace(r'[$,%€£]', '', regex=True).str.strip()
            
            # Attempt to coerce
            coerced = pd.to_numeric(cleaned_series, errors='coerce')
            valid_coerced = coerced.notna().sum()
            
            # Coerce if >= 85% of populated strings are valid numbers
            if valid_coerced / num_valid_strings >= 0.85:
                df[col] = coerced
                self.coercion_audit.append({
                    "column": col,
                    "original_type": "object",
                    "inferred_type": str(coerced.dtype),
                    "reason": f"{valid_coerced}/{num_valid_strings} values parsed as numeric"
                })
                logger.info(f"Auto-coerced column '{col}' from object to numeric.")
        
        return df
    
    def basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Return deep, structured dataset metadata including a full feature audit table.
        """
        try:
            memory_usage = float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        except Exception:
            memory_usage = 0.0
            
        numeric_df = df.select_dtypes(include=[np.number])
        
        sparsity = float((df == 0).sum().sum() / (df.shape[0] * df.shape[1])) if df.size > 0 else 0.0
        inf_count = int(np.isinf(numeric_df).sum().sum()) if not numeric_df.empty else 0
        
        # Generate Preprocessing Feature Audit
        feature_audit = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing_count = int(df[col].isnull().sum())
            missing_pct = (missing_count / len(df)) * 100
            cardinality = df[col].nunique()
            
            coercion_note = next((item['inferred_type'] for item in self.coercion_audit if item['column'] == col), "Native")
            
            # Determine expected preprocessing strategy
            if "int" in dtype or "float" in dtype:
                strategy = "Impute Median + Robust Scale"
            elif cardinality < 15:
                strategy = "Impute Mode + OHE"
            else:
                strategy = "Target/Hash Encoding or Drop"
                
            feature_audit.append({
                "Feature": col,
                "Dtype": dtype,
                "Missing (%)": round(missing_pct, 2),
                "Missing (Count)": missing_count,
                "Cardinality": cardinality,
                "Type Origin": coercion_note,
                "Auto-Strategy": strategy
            })
        
        return {
            "num_rows": int(df.shape[0]),
            "num_columns": int(df.shape[1]),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": memory_usage,
            "sparsity_ratio": sparsity,
            "infinite_values": inf_count,
            "coercion_logs": self.coercion_audit,
            "feature_audit": feature_audit
        }