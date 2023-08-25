import pandas as pd
import pytest
from datasets import Dataset


@pytest.fixture()
def df_pandas_with_nulls():
    test_df: pd.DataFrame = pd.DataFrame(data={"col1": [1, None], "col2": [2, 3]})
    return test_df


@pytest.fixture()
def df_pandas_with_non_printable_characters() -> pd.DataFrame:
    test_df = pd.DataFrame(
        data={"col1": ["Déjà vu", "test123"], "col2": ["Ò|zz", "test 456"]}
    )
    return test_df


@pytest.fixture()
def dummy_dataset() -> Dataset:
    df_text_label: pd.DataFrame = pd.DataFrame(
        {
            "text": ["abc", "xyz"],
            "label": [0, 1],
        }
    )
    return Dataset.from_pandas(df_text_label, preserve_index=False)
