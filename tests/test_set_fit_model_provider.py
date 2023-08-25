from unittest import mock

import pytest
from datasets import Dataset

from set_fit_model_provider import SetFitModelProvider


@pytest.mark.unit
@mock.patch("setfit.SetFitTrainer.evaluate")
@mock.patch("setfit.SetFitTrainer.train")
def test_set_fit_trainer_train_and_evaluate_methods_are_called(
    mock_set_fit_trainer_train: mock.MagicMock,
    mock_set_fit_trainer_evaluate: mock.MagicMock,
    dummy_dataset_with_integer_labels: Dataset,
):
    set_fit_model_provider = SetFitModelProvider(
        model_name="my_set_fit_model",
        train_ds=dummy_dataset_with_integer_labels,
        test_ds=dummy_dataset_with_integer_labels,
    )
    _ = set_fit_model_provider.train()
    assert mock_set_fit_trainer_train.called_once
    assert mock_set_fit_trainer_evaluate.called_once
