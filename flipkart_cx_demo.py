from __future__ import annotations

import pandas as pd

import logging

from data_cleaner import DataCleaner
from data_partitioner import DataPartitioner
from set_fit_model_provider import SetFitModelProvider

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# TODO: <Alex>ALEX-Cleanup</Alex>
# Add return typehint when done
# TODO: <Alex>ALEX</Alex>
def run_flipkart_demo(csv_file_path: str, delimiter: str = ",", encoding: str = "latin1"):
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

    set_fit_model_provider: SetFitModelProvider = SetFitModelProvider(
        model_name="my-test-setfit-model",
        df_train=df_train,
        df_test=df_test,
        selection_range=range(8 * 2),
    )

    metrics: dict[str, float] | None = None
    try:
        set_fit_model_provider.load_model()
    except FileNotFoundError as e:
        metrics = set_fit_model_provider.train()
        set_fit_model_provider.persist_model()

    # Use the SetFit model to predict
    # TODO: <Alex>ALEX</Alex>
    # set_fit_model_provider.load_model()
    # TODO: <Alex>ALEX</Alex>
    df_eval["setfit"] = df_eval["text"].apply(lambda x: int(set_fit_model_provider.predict(x)))

    # TODO: <Alex>ALEX</Alex>
    # return data_cleaner.dataframe
    # TODO: <Alex>ALEX</Alex>
    # TODO: <Alex>ALEX</Alex>
    return df_eval, df_train, df_test, metrics
    # TODO: <Alex>ALEX</Alex>


if __name__ == "__main__":
    import sys

    # TODO: <Alex>ALEX-Cleanup</Alex>
    csv_file_path: str = sys.argv[1] if len(sys.argv) >= 2 else None
    # TODO: <Alex>ALEX</Alex>
    # TODO: <Alex>ALEX</Alex>
    # run_flipkart_demo(target_dir)
    # TODO: <Alex>ALEX</Alex>
    # TODO: <Alex>ALEX</Alex>
    res = run_flipkart_demo(csv_file_path=csv_file_path)
    # print(f'\n[ALEX_TEST] [MAIN] DF.SHAPE:\n{res.shape} ; TYPE: {str(type(res.shape))}')
    # print(f'\n[ALEX_TEST] [MAIN] DF.HEAD:\n{res.head()} ; TYPE: {str(type(res.head()))}')
    # print(f'\n[ALEX_TEST] [MAIN] DF.TAIL:\n{res.tail()} ; TYPE: {str(type(res.tail()))}')
    # TODO: <Alex>ALEX</Alex>
    # TODO: <Alex>ALEX</Alex>
    # df_eval, df_train, df_test, metrics = res
    # print(f'\n[ALEX_TEST] [MAIN] DF_EVAL.SHAPE:\n{df_eval.shape} ; TYPE: {str(type(df_eval.shape))}')
    # print(f'\n[ALEX_TEST] [MAIN] DF_EVAL.HEAD:\n{df_eval.head()} ; TYPE: {str(type(df_eval.head()))}')
    # # print(f'\n[ALEX_TEST] [MAIN] DF_EVAL.TAIL:\n{df_eval.tail()} ; TYPE: {str(type(df_eval.tail()))}')
    # print(f'\n[ALEX_TEST] [MAIN] DF_TRAIN.SHAPE:\n{df_train.shape} ; TYPE: {str(type(df_train.shape))}')
    # print(f'\n[ALEX_TEST] [MAIN] DF_TRAIN.HEAD:\n{df_train.head()} ; TYPE: {str(type(df_train.head()))}')
    # # print(f'\n[ALEX_TEST] [MAIN] DF_TRAIN.TAIL:\n{df_train.tail()} ; TYPE: {str(type(df_train.tail()))}')
    # print(f'\n[ALEX_TEST] [MAIN] DF_TEST.SHAPE:\n{df_test.shape} ; TYPE: {str(type(df_test.shape))}')
    # print(f'\n[ALEX_TEST] [MAIN] DF_TEST.HEAD:\n{df_test.head()} ; TYPE: {str(type(df_test.head()))}')
    # # print(f'\n[ALEX_TEST] [MAIN] DF_TEST.TAIL:\n{df_test.tail()} ; TYPE: {str(type(df_test.tail()))}')
    # print(f'\n[ALEX_TEST] [MAIN] DF_TEST.SETFIT_METRICS:\n{metrics} ; TYPE: {str(type(metrics))}')
    # TODO: <Alex>ALEX</Alex>
