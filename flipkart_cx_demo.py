import pandas as pd

import logging

from datasets import DatasetDict

from data_cleaner import DataCleaner
from dataset_builder import DatasetBuilder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    dataset_builder: DatasetBuilder = DatasetBuilder(dataframe=data_cleaner.dataframe)

    df_eval: pd.DataFrame = dataset_builder.evaluation_dataframe
    datasets_train_test: DatasetDict = dataset_builder.datasets_train_test

    # TODO: <Alex>ALEX</Alex>
    # return data_cleaner.dataframe
    # TODO: <Alex>ALEX</Alex>
    # TODO: <Alex>ALEX</Alex>
    return df_eval, datasets_train_test
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
    # print(f'\n[ALEX_TEST] [MAIN] DF.HEAD:\n{res.head} ; TYPE: {str(type(res.head))}')
    # print(f'\n[ALEX_TEST] [MAIN] DF.TAIL:\n{res.tail} ; TYPE: {str(type(res.tail))}')
    # TODO: <Alex>ALEX</Alex>
    # TODO: <Alex>ALEX</Alex>
    df_eval, datasets_train_test = res
    print(f'\n[ALEX_TEST] [MAIN] DF_EVAL.SHAPE:\n{df_eval.shape} ; TYPE: {str(type(df_eval.shape))}')
    print(f'\n[ALEX_TEST] [MAIN] DF_EVAL.HEAD:\n{df_eval.head} ; TYPE: {str(type(df_eval.head))}')
    # print(f'\n[ALEX_TEST] [MAIN] DF_EVAL.TAIL:\n{df_eval.tail} ; TYPE: {str(type(df_eval.tail))}')
    print(f'\n[ALEX_TEST] [MAIN] DATASETS_TRAIN_TEST:\n{datasets_train_test} ; TYPE: {str(type(datasets_train_test))}')
    # TODO: <Alex>ALEX</Alex>
