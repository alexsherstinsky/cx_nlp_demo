from __future__ import annotations

import dataclasses
import logging
import pathlib

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data_cleaner import DataCleaner
from data_partitioner import DataPartitioner
from set_fit_model_provider import SetFitModelProvider

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclasses.dataclass(frozen=True)
class PerformanceMetrics:
    """
    Performance metrics (precision, recall, f1_score, accuracy) for named predictions vs. ground truth evaluation.
    """

    name: str
    precision: float
    recall: float
    f1_score: float
    accuracy: float


class FlipkartDemo:
    @staticmethod
    def run_flipkart_demo(
        csv_file_path: pathlib.Path, delimiter: str = ",", encoding: str = "latin1"
    ) -> pd.DataFrame:
        """
        Runs main logic of the experiment:
            1) Read in https://www.kaggle.com/datasets/mansithummar67/flipkart-product-review-dataset (in CSV format).
            2) Clean the dataset.
            3) Prepare a classification task by converting the "Rate" column to binary (0, 1) -> (negative, positive) class labels.
            4) Partition the dataset into training (default 8 samples), test (default 8 samples), and evaluation (default 10 samples) dataframes.
            5) Train SetFit classifier.
            6) Inference:
                a) Use SetFit model to predict;
                b) Use OpenAI Conversation API model to predict (these predictions will serve as alternative labels).
            7) Compute performance metrics (precision, recall, f1_score, accuracy) across different predictions vs. ground truth.
            8) Log the experiment to weights and biases (TODO).

        Args:
            csv_file_path: path to CSV file containing raw data
            delimiter: CSV column delimiter
            encoding: CSV data encoding format

        Returns:
            Evaluation dataframe with predictions from running inference on "text" column by several classifiers.
        """
        df_original: pd.DataFrame = pd.read_csv(
            csv_file_path, delimiter=delimiter, encoding=encoding
        )

        # Clean original dataset.
        data_cleaner: DataCleaner = DataCleaner(dataframe=df_original)
        data_cleaner.remove_nulls()
        data_cleaner.remove_non_printable_characters(
            column_names=["ProductName", "Summary"]
        )
        data_cleaner.retain_numeric_rows_for_column(column_name="Rate")

        # Use "Summary" column as "text" and "Rate" column as "label" (these standard attribute names can be made customizable in the future).
        data_cleaner.create_standard_text_and_label_columns(
            source_text_column_name="Summary", source_label_column_name="Rate"
        )
        # For binary classification purposes, convert 5-star "Rate" ("label") column entries to 0 and 1 (default threshold is 3; this can be experimented with in the future).
        data_cleaner.convert_label_column_to_binary(threshold=3)

        # Partition the cleaned dataframe into train, test, and evaluation datasets.
        dataset_builder: DataPartitioner = DataPartitioner(
            dataframe=data_cleaner.dataframe
        )

        # Obtain train and test datasets (non-overlapping parts of the overall cleaned dataframe).
        df_train: pd.DataFrame
        df_test: pd.DataFrame
        df_train, df_test = dataset_builder.train_and_test_dataframes

        # Train the SetFit model, unless that has already been done and the model file exists on the local filesystem.
        model_name: str = "my-test-setfit-model"

        set_fit_model_provider: SetFitModelProvider = SetFitModelProvider(
            model_name=model_name,
            df_train=df_train,
            df_test=df_test,
            selection_range=range(8 * 2),
        )

        try:
            # The SetFit model has already been trained previously; hence, load it.
            set_fit_model_provider.load_model()
        except FileNotFoundError:
            # The SetFit model has not yet been trained; hence, train it and persist it to the local filesystem.
            metrics: dict[str, float] = set_fit_model_provider.train()
            logger.info(f'Evaluating trained model "{model_name}": {metrics}.')
            set_fit_model_provider.persist_model()

        # Obtain evaluation dataset (non-overlapping with train and test datasets part of the overall cleaned dataframe).
        df_evaluation: pd.DataFrame = dataset_builder.evaluation_dataframe

        # Use the SetFit model to predict.
        df_evaluation["setfit"] = df_evaluation["text"].apply(
            lambda x: int(set_fit_model_provider.predict(x))
        )

        return df_evaluation

    @staticmethod
    def get_performance_metrics(
        name: str, df_evaluation: pd.DataFrame, truth_label: str, prediction_label: str
    ) -> PerformanceMetrics:
        """
        Computes performance metrics (precision, recall, f1_score, accuracy) for predictions vs. ground truth.

        Args:
            name: Indication of what is being evaluated (i.e., which classifier against what is taken as ground truth).
            df_evaluation: Evaluation dataframe with predictions from running inference on "text" column by several classifiers.
            truth_label: Name of column containing ground truth labels.
            prediction_label: Name of column containing predictions for the named evaluation.

        Returns:
            PerformanceMetrics object containing performance metrics (precision, recall, f1_score, accuracy) for named predictions vs. ground truth evaluation.
        """
        precision: float = precision_score(
            y_true=df_evaluation[truth_label].tolist(),
            y_pred=df_evaluation[prediction_label].tolist(),
        )
        recall: float = recall_score(
            y_true=df_evaluation[truth_label].tolist(),
            y_pred=df_evaluation[prediction_label].tolist(),
        )
        f1: float = f1_score(
            y_true=df_evaluation[truth_label].tolist(),
            y_pred=df_evaluation[prediction_label].tolist(),
        )
        accuracy: float = accuracy_score(
            y_true=df_evaluation[truth_label].tolist(),
            y_pred=df_evaluation[prediction_label].tolist(),
        )
        return PerformanceMetrics(
            name=name,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
        )


if __name__ == "__main__":
    import sys

    # TODO: <Alex>ALEX-Cleanup</Alex>
    # Obtain data file (stored in CSV format) from command line.
    csv_file_path_str: str = sys.argv[1] if len(sys.argv) >= 2 else None
    # TODO: <Alex>ALEX</Alex>
    flipkart_demo = FlipkartDemo()

    df_eval: pd.DataFrame = flipkart_demo.run_flipkart_demo(
        csv_file_path=pathlib.Path(csv_file_path_str)
    )

    performance_metrics: PerformanceMetrics

    performance_metrics = FlipkartDemo.get_performance_metrics(
        name="setfit_to_rating",
        df_evaluation=df_eval,
        truth_label="label",
        prediction_label="setfit",
    )
    print(
        f"\n[ALEX_TEST] [MAIN] PERFORMANCE_METRICS_SETFIT_TO_RATING:\n{performance_metrics} ; TYPE: {str(type(performance_metrics))}"
    )
