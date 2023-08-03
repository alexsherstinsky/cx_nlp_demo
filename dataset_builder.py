from __future__ import annotations

import pandas as pd

import logging

from datasets import DatasetDict, Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DatasetBuilder:
    DEFAULT_RANDOM_STATE: int = 200

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        The DatasetBuilder

        Args:
            dataframe: Pandas DataFrame containing raw data
        """
        self._dataframe: pd.DataFrame = dataframe.copy()

        self._evaluation_dataframe: pd.DataFrame = self._get_evaluation_dataframe()
        self._datasets_train_test: DatasetDict = self._build_train_test_datasets()

    @property
    def evaluation_dataframe(self) -> pd.DataFrame:
        return self._evaluation_dataframe

    @property
    def datasets_train_test(self) -> DatasetDict:
        return self._datasets_train_test

    def _get_evaluation_dataframe(self, num_samples: int = 10, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
        # Hold specified number of random samples out for validating the model performance.
        df_eval: pd.DataFrame = self._dataframe.sample(n=num_samples, random_state=random_state)
        self._dataframe = self._dataframe.drop(df_eval.index)

        return df_eval

    def _build_train_test_datasets(self, num_train_samples: int = 8, num_test_samples: int = 10, random_state: int = DEFAULT_RANDOM_STATE) -> DatasetDict:
        # Separate negatives and positives into their own dataframes.
        df_negatives: pd.DataFrame = self._dataframe[self._dataframe["label"] == 0]
        df_positives: pd.DataFrame = self._dataframe.drop(df_negatives.index)

        df_train_negatives: pd.DataFrame = df_negatives.sample(n=num_train_samples, random_state=random_state)
        df_negatives: pd.DataFrame = df_negatives.drop(df_train_negatives.index)

        df_train_positives: pd.DataFrame = df_positives.sample(n=num_train_samples, random_state=random_state)
        df_positives: pd.DataFrame = df_positives.drop(df_train_positives.index)

        df_train: pd.DataFrame = pd.concat([df_train_negatives, df_train_positives])

        df_test_negatives: pd.DataFrame = df_negatives.sample(n=num_test_samples, random_state=random_state)
        # TODO: <Alex>ALEX</Alex>
        # df_negatives: pd.DataFrame = df_negatives.drop(df_test_negatives.index)
        # TODO: <Alex>ALEX</Alex>

        df_test_positives: pd.DataFrame = df_positives.sample(n=num_test_samples, random_state=random_state)
        # TODO: <Alex>ALEX</Alex>
        # df_positives: pd.DataFrame = df_positives.drop(df_test_positives.index)
        # TODO: <Alex>ALEX</Alex>

        df_test: pd.DataFrame = pd.concat([df_test_negatives, df_test_positives])

        # Create train/test DatasetDict
        datasets_train_test: DatasetDict = DatasetDict(
            {
                "train": Dataset.from_pandas(df_train, preserve_index=False),
                "test": Dataset.from_pandas(df_test, preserve_index=False)
            }
        )

        return datasets_train_test
