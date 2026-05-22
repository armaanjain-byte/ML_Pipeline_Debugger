# FILE: app/pipeline/data_loader.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.exceptions import (
    DataLoadException,
    DatasetEmptyException,
    InvalidDatasetFormatException,
    MissingTargetColumnException,
    UnsupportedFileTypeException,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

SUPPORTED_EXTENSIONS = {
    ".csv",
}

DEFAULT_ENCODING_CANDIDATES = [
    "utf-8",
    "utf-8-sig",
    "latin1",
    "cp1252",
]

DEFAULT_MISSING_TOKENS = {
    "",
    " ",
    "NA",
    "N/A",
    "NULL",
    "null",
    "None",
    "none",
    "?",
    "-",
}

MAX_PREVIEW_ROWS = 5

# ============================================================================
# DATA LOADER
# ============================================================================


class DataLoader:
    """
    Stable dataset loading and schema inspection layer.

    Goals:
    - deterministic CSV loading
    - encoding-safe ingestion
    - stable dtype normalization
    - reproducible schema behavior
    - dashboard-safe metadata extraction
    - compatibility-safe missing handling

    IMPORTANT:
    This implementation preserves original loading semantics
    while improving reliability and consistency.
    """

    def __init__(
        self,
        file_path: str | Path,
    ):
        self.file_path = Path(file_path)

        self.loaded_dataframe: Optional[
            pd.DataFrame
        ] = None

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def load_data(self) -> pd.DataFrame:
        """
        Stable dataset loading entrypoint.
        """

        self._validate_file()

        dataframe = self._read_dataset()

        dataframe = self._normalize_dataframe(
            dataframe
        )

        self._validate_dataframe(dataframe)

        self.loaded_dataframe = dataframe

        logger.info(
            "Dataset loaded successfully | rows=%s | columns=%s",
            len(dataframe),
            len(dataframe.columns),
        )

        return dataframe

    def basic_info(
        self,
        dataframe: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Stable metadata extraction.

        Used by:
        - pipeline runner
        - dashboard
        - diagnostics
        """

        feature_audit = (
            self._build_feature_audit(
                dataframe
            )
        )

        return {
            "num_rows": int(len(dataframe)),
            "num_columns": int(
                len(dataframe.columns)
            ),
            "columns": list(
                dataframe.columns
            ),
            "dtypes": {
                column: str(dtype)
                for column, dtype
                in dataframe.dtypes.items()
            },
            "memory_usage_mb": round(
                float(
                    dataframe.memory_usage(
                        deep=True
                    ).sum()
                    / (1024**2)
                ),
                2,
            ),
            "duplicate_rows": int(
                dataframe.duplicated().sum()
            ),
            "missing_cells": int(
                dataframe.isna().sum().sum()
            ),
            "feature_audit": feature_audit,
            "preview": dataframe.head(
                MAX_PREVIEW_ROWS
            ).to_dict(orient="records"),
        }

    def validate_target_column(
        self,
        target_column: str,
    ) -> None:
        """
        Stable target validation.
        """

        if self.loaded_dataframe is None:
            raise DataLoadException(
                "No dataset loaded."
            )

        if (
            target_column
            not in self.loaded_dataframe.columns
        ):
            raise MissingTargetColumnException(
                (
                    f"Target column "
                    f"'{target_column}' "
                    f"not found."
                ),
                context={
                    "available_columns":
                    list(
                        self.loaded_dataframe.columns
                    )
                },
            )

    # ==========================================================
    # FILE VALIDATION
    # ==========================================================

    def _validate_file(self) -> None:
        """
        Prevents invalid file contracts.
        """

        if not self.file_path.exists():
            raise DataLoadException(
                f"File does not exist: "
                f"{self.file_path}"
            )

        if not self.file_path.is_file():
            raise DataLoadException(
                f"Invalid file path: "
                f"{self.file_path}"
            )

        extension = (
            self.file_path.suffix.lower()
        )

        if (
            extension
            not in SUPPORTED_EXTENSIONS
        ):
            raise UnsupportedFileTypeException(
                (
                    f"Unsupported file type: "
                    f"{extension}"
                ),
                context={
                    "supported_extensions":
                    sorted(
                        SUPPORTED_EXTENSIONS
                    )
                },
            )

    # ==========================================================
    # FILE READING
    # ==========================================================

    def _read_dataset(
        self,
    ) -> pd.DataFrame:
        """
        Encoding-safe CSV ingestion.
        """

        last_error = None

        for encoding in (
            DEFAULT_ENCODING_CANDIDATES
        ):
            try:
                dataframe = pd.read_csv(
                    self.file_path,
                    encoding=encoding,
                    na_values=list(
                        DEFAULT_MISSING_TOKENS
                    ),
                    keep_default_na=True,
                    low_memory=False,
                )

                logger.info(
                    "Dataset read successfully "
                    "using encoding=%s",
                    encoding,
                )

                return dataframe

            except Exception as error:
                last_error = error

                logger.warning(
                    "Failed loading dataset "
                    "with encoding=%s | %s",
                    encoding,
                    str(error),
                )

        raise InvalidDatasetFormatException(
            (
                "Failed to parse dataset. "
                "Could not decode CSV safely."
            ),
            context={
                "file_path":
                str(self.file_path),
                "last_error":
                str(last_error),
            },
        )

    # ==========================================================
    # NORMALIZATION
    # ==========================================================

    def _normalize_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Stable dataframe normalization.

        Goals:
        - deterministic columns
        - normalized missing values
        - safer string handling
        - stable schema inference
        """

        dataframe = dataframe.copy()

        # ======================================================
        # COLUMN NORMALIZATION
        # ======================================================

        dataframe.columns = [
            str(column).strip()
            for column
            in dataframe.columns
        ]

        # ======================================================
        # DUPLICATE COLUMN HANDLING
        # ======================================================

        dataframe = (
            self._resolve_duplicate_columns(
                dataframe
            )
        )

        # ======================================================
        # STRING NORMALIZATION
        # ======================================================

        object_columns = (
            dataframe.select_dtypes(
                include=["object"]
            ).columns
        )

        for column in object_columns:
            try:
                dataframe[column] = (
                    dataframe[column]
                    .astype(str)
                    .str.strip()
                )

                dataframe.loc[
                    dataframe[column].isin(
                        DEFAULT_MISSING_TOKENS
                    ),
                    column,
                ] = np.nan

            except Exception:
                logger.warning(
                    "String normalization failed "
                    "for column '%s'",
                    column,
                )

        return dataframe

    def _resolve_duplicate_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Deterministic duplicate-column handling.
        """

        seen_columns = {}

        resolved_columns = []

        for column in dataframe.columns:
            if column not in seen_columns:
                seen_columns[column] = 0

                resolved_columns.append(
                    column
                )

            else:
                seen_columns[column] += 1

                new_column_name = (
                    f"{column}__duplicate_"
                    f"{seen_columns[column]}"
                )

                resolved_columns.append(
                    new_column_name
                )

        dataframe.columns = resolved_columns

        return dataframe

    # ==========================================================
    # VALIDATION
    # ==========================================================

    def _validate_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Stable dataframe validation.
        """

        if dataframe.empty:
            raise DatasetEmptyException(
                "Dataset contains no rows."
            )

        if len(dataframe.columns) == 0:
            raise DatasetEmptyException(
                "Dataset contains no columns."
            )

    # ==========================================================
    # FEATURE AUDIT
    # ==========================================================

    def _build_feature_audit(
        self,
        dataframe: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Dashboard-safe feature metadata.
        """

        audit_rows = []

        for column in dataframe.columns:
            series = dataframe[column]

            missing_pct = (
                series.isna().mean() * 100
            )

            unique_count = (
                series.nunique(
                    dropna=True
                )
            )

            row = {
                "feature": column,
                "dtype": str(series.dtype),
                "missing_pct": round(
                    float(missing_pct),
                    2,
                ),
                "unique_values": int(
                    unique_count
                ),
                "is_numeric": bool(
                    pd.api.types.is_numeric_dtype(
                        series
                    )
                ),
                "is_categorical": bool(
                    (
                        pd.api.types.is_object_dtype(
                            series
                        )
                    )
                    or (
                        pd.api.types.is_categorical_dtype(
                            series
                        )
                    )
                ),
            }

            audit_rows.append(row)

        audit_rows.sort(
            key=lambda row: (
                -row["missing_pct"],
                row["feature"],
            )
        )

        return audit_rows
