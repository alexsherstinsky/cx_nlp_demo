import pandas as pd

import pytest
# TODO: <Alex>ALEX</Alex>
# from unittest import mock
# TODO: <Alex>ALEX</Alex>

from data_cleaner import DataCleaner


@pytest.fixture()
def df_pandas_with_nulls():
    test_df: pd.DataFrame = pd.DataFrame(data={"col1": [1, None], "col2": [2, 3]})
    return test_df


@pytest.fixture()
def df_pandas_with_non_printable_characters() -> pd.DataFrame:
    test_df = pd.DataFrame(data={"col1": ["Déjà vu", "test123"], "col2": ["Ò|zz", "test 456"]})
    return test_df


def test_remove_non_printable_characters(df_pandas_with_non_printable_characters: pd.DataFrame):
    data_cleaner = DataCleaner(dataframe=df_pandas_with_non_printable_characters)
    data_cleaner.remove_non_printable_characters(column_names=["col1", "col2"])

    """
    This assertion shows that 
    """
    expected_df = pd.DataFrame(data={"col1": ["D j  vu", "test123"], "col2": ["  zz", "test 456"]})
    pd.testing.assert_frame_equal(data_cleaner.dataframe, expected_df)
