import pandas as pd
import pytest

from data_cleaner import DataCleaner


@pytest.mark.unit
def test_remove_non_printable_characters(
    df_pandas_with_non_printable_characters: pd.DataFrame,
):
    data_cleaner = DataCleaner(dataframe=df_pandas_with_non_printable_characters)
    data_cleaner.remove_non_printable_characters(column_names=["col1", "col2"])

    """
    This assertion shows that
    """
    expected_df = pd.DataFrame(
        data={"col1": ["D j  vu", "test123"], "col2": ["  zz", "test 456"]}
    )
    pd.testing.assert_frame_equal(data_cleaner.dataframe, expected_df)
