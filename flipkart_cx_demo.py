from __future__ import annotations

import dataclasses

import pandas as pd

import logging

from data_cleaner import DataCleaner
from data_partitioner import DataPartitioner
from set_fit_model_provider import SetFitModelProvider
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclasses.dataclass(frozen=True)
class PerformanceMetrics:
    name: str
    precision: float
    recall: float
    f1_score: float
    accuracy: float


class FlipkartDemo:
    @staticmethod
    def run_flipkart_demo(csv_file_path: str, delimiter: str = ",", encoding: str = "latin1") -> pd.DataFrame:
        df_original: pd.DataFrame = pd.read_csv(csv_file_path, delimiter=delimiter, encoding=encoding)

        data_cleaner: DataCleaner = DataCleaner(dataframe=df_original)
        data_cleaner.remove_nulls()
        data_cleaner.remove_non_printable_characters(column_names=["ProductName", "Summary"])
        data_cleaner.retain_numeric_rows_for_column(column_name="Rate")
        data_cleaner.create_standard_text_and_label_columns(source_text_column_name="Summary", source_label_column_name="Rate")
        data_cleaner.convert_label_column_to_binary()

        dataset_builder: DataPartitioner = DataPartitioner(dataframe=data_cleaner.dataframe)

        df_eval: pd.DataFrame = dataset_builder.evaluation_dataframe

        df_train: pd.DataFrame
        df_test: pd.DataFrame
        df_train, df_test = dataset_builder.train_and_test_dataframes

        model_name: str = "my-test-setfit-model"

        set_fit_model_provider: SetFitModelProvider = SetFitModelProvider(
            model_name=model_name,
            df_train=df_train,
            df_test=df_test,
            selection_range=range(8 * 2),
        )

        try:
            set_fit_model_provider.load_model()
        except FileNotFoundError as e:
            metrics: dict[str, float] = set_fit_model_provider.train()
            logger.info(f'Evaluating trained model "{model_name}": {metrics}.')
            set_fit_model_provider.persist_model()

        # Use the SetFit model to predict
        df_eval["setfit"] = df_eval["text"].apply(lambda x: int(set_fit_model_provider.predict(x)))

        return df_eval

    @staticmethod
    def get_performance_metrics(name: str, df_evaluation: pd.DataFrame, truth_label: str, prediction_label: str) -> PerformanceMetrics:
        precision: float = precision_score(y_true=df_evaluation[truth_label].tolist(), y_pred=df_evaluation[prediction_label].tolist())
        recall: float = recall_score(y_true=df_evaluation[truth_label].tolist(), y_pred=df_evaluation[prediction_label].tolist())
        f1: float = f1_score(y_true=df_evaluation[truth_label].tolist(), y_pred=df_evaluation[prediction_label].tolist())
        accuracy: float = accuracy_score(y_true=df_evaluation[truth_label].tolist(), y_pred=df_evaluation[prediction_label].tolist())
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
    csv_file_path: str = sys.argv[1] if len(sys.argv) >= 2 else None
    # TODO: <Alex>ALEX</Alex>
    flipkart_demo = FlipkartDemo()

    df_eval: pd.DataFrame = flipkart_demo.run_flipkart_demo(csv_file_path=csv_file_path)

    performance_metrics: PerformanceMetrics

    performance_metrics = FlipkartDemo.get_performance_metrics(
        name="setfit_to_rating",
        df_evaluation=df_eval,
        truth_label="label", prediction_label="setfit",
    )
    print(f'\n[ALEX_TEST] [MAIN] PERFORMANCE_METRICS_SETFIT_TO_RATING:\n{performance_metrics} ; TYPE: {str(type(performance_metrics))}')
