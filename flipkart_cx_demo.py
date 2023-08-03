import pandas as pd

import logging

from data_cleaner import DataCleaner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_flipkart_demo(csv_file_path: str, delimiter: str = ",", encoding: str = "latin1") -> None:
    df_original: pd.DataFrame = pd.read_csv(csv_file_path, delimiter=delimiter, encoding=encoding)

    data_cleaner: DataCleaner = DataCleaner(dataframe=df_original)
    data_cleaner.remove_nulls()
    data_cleaner.remove_non_printable_characters(column_names=["ProductName", "Summary"])
    data_cleaner.retain_numeric_rows_for_column(column_name="Rate")
    data_cleaner.create_standard_text_and_label_columns(source_text_column_name="Summary", source_label_column_name="Rate")
    data_cleaner.convert_label_column_to_binary()


if __name__ == "__main__":
    import sys

    target_dir = sys.argv[1] if len(sys.argv) >= 2 else "."  # noqa: PLR2004
    run_flipkart_demo(target_dir)
