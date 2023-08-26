import pandas as pd
import pytest

from data_cleaner import DataCleaner


@pytest.mark.unit
def test_remove_nulls(
    df_pandas_with_nulls: pd.DataFrame,
):
    data_cleaner = DataCleaner(dataframe=df_pandas_with_nulls)
    data_cleaner.remove_nulls()

    """
    This assertion shows that None values are no longer present in processed dataframe.
    """
    expected_df: pd.DataFrame = pd.DataFrame(data={"col1": [1.0], "col2": [2]})
    pd.testing.assert_frame_equal(data_cleaner.dataframe, expected_df)


@pytest.mark.unit
def test_remove_non_printable_characters(
    df_pandas_with_non_printable_characters: pd.DataFrame,
):
    data_cleaner = DataCleaner(dataframe=df_pandas_with_non_printable_characters)
    data_cleaner.remove_non_printable_characters(column_names=["col1", "col2"])

    """
    This assertion shows that unprintable characters are no longer present in processed dataframe.
    """
    expected_df: pd.DataFrame = pd.DataFrame(
        data={"col1": ["D j  vu", "test123"], "col2": ["  zz", "test 456"]}
    )
    pd.testing.assert_frame_equal(data_cleaner.dataframe, expected_df)


@pytest.mark.parametrize(
    "df_test_name,convert_to_int,expected_df",
    [
        pytest.param(
            "df_pandas_with_integer_labels",
            False,
            pd.DataFrame(
                data={
                    "text": ["abc", "xyz"],
                    "label": [2, 4],
                }
            ),
            id="all_rows_retained_integers",
        ),
        pytest.param(
            "df_pandas_with_integer_labels",
            True,
            pd.DataFrame(
                data={
                    "text": ["abc", "xyz"],
                    "label": [2, 4],
                }
            ),
            id="all_rows_retained_integers_noop_conversion",
        ),
        pytest.param(
            "df_pandas_with_numeric_parseable_labels",
            False,
            pd.DataFrame(
                data={
                    "text": ["abc", "xyz"],
                    "label": ["0", 3],
                }
            ),
            id="all_rows_retained_parseable",
        ),
        pytest.param(
            "df_pandas_with_numeric_parseable_labels",
            True,
            pd.DataFrame(
                data={
                    "text": ["abc", "xyz"],
                    "label": [0, 3],
                }
            ),
            id="all_rows_retained_parseable_with_conversion",
        ),
        pytest.param(
            "df_pandas_with_numeric_nonparseable_labels",
            False,
            pd.DataFrame(
                data={
                    "text": ["abc"],
                    "label": ["0"],
                }
            ),
            id="not_all_rows_retained_nonparseable",
        ),
        pytest.param(
            "df_pandas_with_numeric_nonparseable_labels",
            True,
            pd.DataFrame(
                data={
                    "text": ["abc"],
                    "label": [0],
                }
            ),
            id="not_all_rows_retained_nonparseable_noop_conversion",
        ),
    ],
)
@pytest.mark.unit
def test_retain_numeric_rows_for_column(
    df_test_name: str,
    convert_to_int: bool,
    expected_df: pd.DataFrame,
    request: pytest.FixtureRequest,
):
    """
    In this test, each parametrized fixture is specified by name and retrieved through the request object.  Then for
    every parametrized test case, the corresponding dataframe is processed for both values of the convert_to_int flag.
    This assertion confirms the correct number of rows with expected column values are retained in processed dataframe.
    """
    df_test: pd.DataFrame = request.getfixturevalue(df_test_name)
    data_cleaner = DataCleaner(dataframe=df_test)
    data_cleaner.retain_numeric_rows_for_column(
        column_name="label", convert_to_int=convert_to_int
    )
    pd.testing.assert_frame_equal(data_cleaner.dataframe, expected_df)


@pytest.mark.unit
def test_create_standard_text_and_label_columns(
    df_pandas_with_text_and_integer_columns: pd.DataFrame,
):
    data_cleaner = DataCleaner(dataframe=df_pandas_with_text_and_integer_columns)
    data_cleaner.create_standard_text_and_label_columns(
        source_text_column_name="col1", source_label_column_name="col2"
    )

    """
    This assertion shows that the new "text" and "label" columns are present in processed dataframe.
    """
    expected_df: pd.DataFrame = pd.DataFrame(
        data={
            "col1": ["abc", "xyz"],
            "col2": [2, 4],
            "text": ["abc", "xyz"],
            "label": [2, 4],
        }
    )
    pd.testing.assert_frame_equal(data_cleaner.dataframe, expected_df)
