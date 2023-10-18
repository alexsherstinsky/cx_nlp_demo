from __future__ import annotations

import pandas as pd
import logging
import re
from typing import TypeVar, Pattern

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


T = TypeVar("T")


def _is_int_or_parseable_as_int(value: T) -> bool:
    if isinstance(value, int):
        return True

    try:
        _ = int(value)
        return True
    except (ValueError, TypeError):
        return False


class DataCleaner:
    """
    The DataCleaner class contains methods for cleaning/conditioning the original dataframe using heuristics.

    Args:
        dataframe: Pandas DataFrame containing raw data
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        self._dataframe: pd.DataFrame = dataframe.copy()

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    def remove_nulls(self) -> None:
        self._dataframe.dropna(inplace=True)

    def remove_non_printable_characters(
        self,
        column_names: list[str] | None = None,
        pattern: Pattern = re.compile(r"[^\x00-\x7f]|[^\w\s]"),
        replacement_character: str = " ",
    ) -> None:
        """
        Removes characters that do not appear "natural" when handled with print() method (default pattern is provided).
        If the column_names list is omitted, then all columns of the dataframe undergo non-printable character removal.

        Args:
            column_names: Optional list of columns to process.
            pattern: Regular Expressions pattern of non-printable characters.
            replacement_character: Character to replace the non-printable characters with (default is one blank space).
        """
        if column_names:
            column_name: str
            for column_name in column_names:
                self._dataframe[column_name] = self._dataframe[column_name].apply(
                    lambda x: re.sub(pattern, replacement_character, x)
                )
        else:
            self._dataframe.replace(
                to_replace=pattern,
                value=replacement_character,
                regex=True,
                inplace=True,
            )

    def retain_numeric_rows_for_column(
        self, column_name: str, convert_to_int: bool = True
    ) -> None:
        """
        Filters the dataframe to retain only those rows, in which the value of the column_name can be parsed as numeric.

        Args:
            column_name: column name of interest as the one whose values can be parsed as numeric.
            convert_to_int: Directive for converting values of column_name in resulting rows to integer type formally.
        """
        self._dataframe = self._dataframe[
            self._dataframe[column_name].apply(
                lambda x: _is_int_or_parseable_as_int(value=x)
            )
        ]
        if convert_to_int:
            self._dataframe[column_name] = self._dataframe[column_name].astype("int")

    def create_standard_text_and_label_columns(
        self, source_text_column_name: str, source_label_column_name: str
    ) -> None:
        """
        Standard labels used by HuggingFace models are "text" and "label" for classifier operations.  While these
        attribute names can be made customizable (HuggingFace API permit this), using standard names aides readability.

        Args:
            source_text_column_name: Column name in source dataset that should serve as the "text" column.
            source_label_column_name: Column name in source dataset that should serve as the "label" column.
        """
        self._dataframe["text"] = self._dataframe[source_text_column_name]
        self._dataframe["label"] = self._dataframe[source_label_column_name]

    def convert_label_column_to_binary(self, threshold: int) -> None:
        """
        For binary classification, convert the integer-valued "label" column entries to 0 and 1 based on threshold.

        Args:
            threshold: Threshold value for producing binary labels if original label values are multi-valued.
        """
        self._dataframe["label"] = self._dataframe["label"].apply(
            lambda x: 1 if x > threshold else 0
        )
