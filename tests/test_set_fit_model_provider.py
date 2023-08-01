import pytest
from unittest import mock

import pandas as pd

from set_fit_model_provider import SetFitModelProvider


class PandasDataFrameStub(pd.DataFrame):
    def __init__(self) -> None:
        super().__init__(data={"text": ["abc", "xyz"], "label": [0, 1]})


@pytest.mark.unit
@mock.patch("setfit.SetFitTrainer.evaluate")
@mock.patch("setfit.SetFitTrainer.train")
def test_set_fit_trainer_train_and_evaluate_methods_are_called(mock_set_fit_trainer_train: mock.MagicMock, mock_set_fit_trainer_evaluate: mock.MagicMock):
    set_fit_model_provider = SetFitModelProvider(
        model_name="my_set_fit_model",
        df_train=PandasDataFrameStub(),
        df_test=PandasDataFrameStub(),
    )
    _ = set_fit_model_provider.train()
    assert mock_set_fit_trainer_train.called_once
    assert mock_set_fit_trainer_evaluate.called_once
