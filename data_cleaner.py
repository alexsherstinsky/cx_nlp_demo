from __future__ import annotations

import pandas as pd

import re


class DataCleaner:
    def __init__(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        The DataCleaner

        Args:
            dataframe: Pandas DataFrame containing raw data
        """
        self._dataframe: pd.DataFrame = dataframe.copy()

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    def remove_nulls(self) -> None:
        self._dataframe.dropna(inplace=True)

    def remove_non_printable_characters(self, column_names: list[str] | None) -> None:
        if column_names:
            column_name: str
            for column_name in column_names:
                self._dataframe[column_name] = self._dataframe[column_name].apply(lambda x: re.sub(r"[^\x00-\x7f]|[^\w\s]", " ", x))
        else:
            self._dataframe.replace(to_replace=r"[^\x00-\x7f]|[^\w\s]", value=" ", regex=True, inplace=True)

    def retain_numeric_rows_for_column(self, column_name: str, convert_to_int: bool = True) -> None:
        self._dataframe = self._dataframe[self._dataframe[column_name].apply(lambda x: x.isnumeric())]
        if convert_to_int:
            self._dataframe[column_name] = self._dataframe[column_name].astype("int")

    def create_standard_text_and_label_columns(self, source_text_column_name: str, source_label_column_name: str) -> None:
        self._dataframe["text"] = self._dataframe[source_text_column_name]
        self._dataframe["label"] = self._dataframe[source_label_column_name]

