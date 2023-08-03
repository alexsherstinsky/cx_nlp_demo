from __future__ import annotations

import joblib
import pandas as pd

from datasets import DatasetDict, Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SetFitModelProvider:
    def __init__(
        self,
        model_name: str,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        selection_range: range | None = None,
        pretrained_model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        shuffle_seed: int = 42,
    ) -> None:
        self._model_name: str = model_name
        self._df_train: pd.DataFrame = df_train
        self._df_test: pd.DataFrame = df_test
        self._selection_range: range | None = selection_range
        self._pretrained_model_name: str = pretrained_model_name
        self._shuffle_seed: int = shuffle_seed

        self._model: SetFitModel = SetFitModel.from_pretrained(pretrained_model_name)
        self._trainer: SetFitTrainer | None = None

    def train(self) -> dict[str, float]:
        train_ds: Dataset
        test_ds: Dataset
        train_ds, test_ds = self._get_train_and_test_datasets()

        trainer = SetFitTrainer(
            model=self._model,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            loss_class=CosineSimilarityLoss,
            batch_size=16,
            num_iterations=20, # Number of text pairs to generate for contrastive learning
            num_epochs=1, # Number of epochs to use for contrastive learning
        )
        logger.info(f'Training model "{self._model_name}".')
        trainer.train()

        self._trainer = trainer

        metrics: dict[str, float] = self._trainer.evaluate()

        return metrics

    def persist_model(self) -> None:
        if self._trainer is None:
            raise ValueError(f'Unable to persist model "{self._model_name}", since it has not yet been trained.')

        joblib.dump(self._trainer, f"{self._model_name}.joblib")

    def load_model(self, reload: bool = False) -> None:
        do_load: bool
        if self._trainer is None:
            logger.info(f'Loading model "{self._model_name}" from most recently saved version.')
            do_load = True
        elif reload:
            logger.info(f'Reloading model "{self._model_name}" from most recently saved version.')
            do_load = True
        else:
            do_load = False

        if do_load:
            self._trainer = joblib.load(f"{self._model_name}.joblib")

    def predict(self, text: str) -> int:
        if self._trainer is None:
            raise ValueError(f'Unable to use model "{self._model_name}" for inference, since it has not yet been trained.')

        return self._trainer.model.predict([text])[0]

    def _get_train_and_test_datasets(self) -> tuple[Dataset, Dataset]:
        datasets_train_test: DatasetDict = DatasetDict(
            {
                "train": Dataset.from_pandas(self._df_train, preserve_index=False),
                "test": Dataset.from_pandas(self._df_test, preserve_index=False)
            }
        )
        train_ds: Dataset = datasets_train_test["train"].shuffle(seed=self._shuffle_seed)
        test_ds: Dataset = datasets_train_test["test"].shuffle(seed=self._shuffle_seed)

        if self._selection_range is not None:
            train_ds = train_ds.select(self._selection_range)
            test_ds = test_ds.select(self._selection_range)

        return train_ds, test_ds
