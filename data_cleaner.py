from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Pattern

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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

    def remove_non_printable_characters(self, column_names: list[str] | None, pattern: Pattern = r"[^\x00-\x7f]|[^\w\s]", replacement_character: str = " ") -> None:
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
                self._dataframe[column_name] = self._dataframe[column_name].apply(lambda x: re.sub(pattern, replacement_character, x))
        else:
            self._dataframe.replace(to_replace=pattern, value=replacement_character, regex=True, inplace=True)

    def retain_numeric_rows_for_column(self, column_name: str, convert_to_int: bool = True) -> None:
        self._dataframe = self._dataframe[self._dataframe[column_name].apply(lambda x: x.isnumeric())]
        if convert_to_int:
            self._dataframe[column_name] = self._dataframe[column_name].astype("int")

    def create_standard_text_and_label_columns(self, source_text_column_name: str, source_label_column_name: str) -> None:
        self._dataframe["text"] = self._dataframe[source_text_column_name]
        self._dataframe["label"] = self._dataframe[source_label_column_name]

    def convert_label_column_to_binary(self, threshold: int = 3) -> None:
        self._dataframe["label"] = self._dataframe["label"].apply(lambda x: 1 if x > threshold else 0)

