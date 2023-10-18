from __future__ import annotations

import logging

import pandas as pd

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataPartitioner:
    """
    The DataPartitioner class contains methods for splitting the original dataframe into the non-overlapping
    dataframes: train, test, and evaluation.

    Args:
        dataframe: Pandas DataFrame containing raw data
    """

    DEFAULT_RANDOM_STATE: int = 200

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        self._dataframe: pd.DataFrame = dataframe.copy()

    def build_dataset_dict(
        self,
        num_train_samples: int = 8,
        num_test_samples: int = 10,
        num_evaluation_samples: int = 10,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> DatasetDict:
        """
        Construct a dictionary containing train, test, and evaluation datasets from the underlying dataframe source.

        This method prepares and organizes the dataset for model training, testing, and evaluation by following a
        specific order of operations. It creates evaluation, train, and test dataframe, converts them into Dataset
        objects, and packs them into a DatasetDict object.

        Note:
            The order of operations is important, because rows are deleted from the original dataframe upon every
            operation.  Hence, the evaluation dataframe is obtained first (to minimize the risk of information leakage),
            followed by obtaining the train and test dataframes (together).

        Args:
            num_train_samples (int, optional): The number of samples to include in the train dataset. Default is 8.
            num_test_samples (int, optional): The number of samples to include in the test dataset. Default is 10.
            num_evaluation_samples (int, optional): The number of samples to include in the evaluation dataset. Default is 10.
            random_state (int, optional): Seed for random number generation. Default is DEFAULT_RANDOM_STATE.

        Returns:
            DatasetDict: A dictionary containing "train", "test', and "eval" datasets.
        """

        df_eval: pd.DataFrame = self._get_evaluation_dataframe(
            num_samples=num_evaluation_samples, random_state=random_state
        )
        df_train: pd.DataFrame
        df_test: pd.DataFrame
        (
            df_train,
            df_test,
        ) = self._get_train_and_test_dataframes(
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            random_state=random_state,
        )
        train_ds: Dataset = Dataset.from_pandas(df_train, preserve_index=False)
        test_ds: Dataset = Dataset.from_pandas(df_test, preserve_index=False)
        eval_ds: Dataset = Dataset.from_pandas(df_eval, preserve_index=False)
        return DatasetDict(
            {
                "train": train_ds,
                "test": test_ds,
                "eval": eval_ds,
            }
        )

    def _get_evaluation_dataframe(
        self, num_samples: int = 10, random_state: int = DEFAULT_RANDOM_STATE
    ) -> pd.DataFrame:
        # Hold specified number of random samples out for validating the model performance.
        df_eval: pd.DataFrame = self._dataframe.sample(
            n=num_samples, random_state=random_state
        )
        self._dataframe = self._dataframe.drop(df_eval.index)

        return df_eval

    def _get_train_and_test_dataframes(
        self,
        num_train_samples: int = 8,
        num_test_samples: int = 10,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Split negatives and positives into separate dataframes (assumes binary labels with 0 and 1 integer values).
        df_negatives: pd.DataFrame = self._dataframe[self._dataframe["label"] == 0]
        df_positives: pd.DataFrame = self._dataframe.drop(df_negatives.index)

        df_train_negatives: pd.DataFrame = df_negatives.sample(
            n=num_train_samples, random_state=random_state
        )
        df_negatives = df_negatives.drop(df_train_negatives.index)

        df_train_positives: pd.DataFrame = df_positives.sample(
            n=num_train_samples, random_state=random_state
        )
        df_positives = df_positives.drop(df_train_positives.index)

        df_train: pd.DataFrame = pd.concat([df_train_negatives, df_train_positives])

        df_test_negatives = df_negatives.sample(
            n=num_test_samples, random_state=random_state
        )

        df_test_positives = df_positives.sample(
            n=num_test_samples, random_state=random_state
        )

        df_test: pd.DataFrame = pd.concat([df_test_negatives, df_test_positives])

        return df_train, df_test
