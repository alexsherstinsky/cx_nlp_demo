import pandas as pd
import pytest
from datasets import Dataset


@pytest.fixture()
def df_pandas_with_nulls():
    test_df: pd.DataFrame = pd.DataFrame(data={"col1": [1, None], "col2": [2, 3]})
    return test_df


@pytest.fixture()
def df_pandas_with_non_printable_characters() -> pd.DataFrame:
    test_df: pd.DataFrame = pd.DataFrame(
        data={"col1": ["Déjà vu", "test123"], "col2": ["Ò|zz", "test 456"]}
    )
    return test_df


@pytest.fixture()
def df_pandas_with_integer_labels() -> pd.DataFrame:
    test_df: pd.DataFrame = pd.DataFrame(
        data={
            "text": ["abc", "xyz"],
            "label": [0, 1],
        }
    )
    return test_df


@pytest.fixture()
def df_pandas_with_numeric_parseable_labels() -> pd.DataFrame:
    test_df: pd.DataFrame = pd.DataFrame(
        data={
            "text": ["abc", "xyz"],
            "label": ["0", 1],
        }
    )
    return test_df


@pytest.fixture()
def df_pandas_with_numeric_nonparseable_labels() -> pd.DataFrame:
    test_df: pd.DataFrame = pd.DataFrame(
        data={
            "text": ["abc", "xyz"],
            "label": ["0", "positive"],
        }
    )
    return test_df


@pytest.fixture()
def dummy_dataset_with_integer_labels(
    df_pandas_with_integer_labels: pd.DataFrame,
) -> Dataset:
    return Dataset.from_pandas(df_pandas_with_integer_labels, preserve_index=False)


@pytest.fixture()
def dummy_dataset_with_numeric_parseable_labels(
    df_pandas_with_numeric_parseable_labels: pd.DataFrame,
) -> Dataset:
    return Dataset.from_pandas(
        df_pandas_with_numeric_parseable_labels, preserve_index=False
    )


@pytest.fixture()
def dummy_dataset_with_numeric_nonparseable_labels(
    df_pandas_with_numeric_nonparseable_labels: pd.DataFrame,
) -> Dataset:
    return Dataset.from_pandas(
        df_pandas_with_numeric_nonparseable_labels, preserve_index=False
    )
