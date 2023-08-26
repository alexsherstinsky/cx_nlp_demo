from __future__ import annotations

import logging

import joblib
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SetFitModelProvider:
    """
    The SetFitModelProvider class is a wrapper around the HuggingFace pretrained SetFitModel and SetFitTrainer classes.
    It initializes the SetFitModel with a pretrained language model and fine-tunes it using few-shot learning strategy.
    For the trained (fine-tuned) model, this class also provides serialization/deserialization and inference methods.
    """

    DEFAULT_SHUFFLE_SEED: int = 42

    def __init__(
        self,
        model_name: str,
        train_ds: Dataset,
        test_ds: Dataset,
        pretrained_model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
        selection_range: range | None = None,
    ) -> None:
        self._model_name: str = model_name
        self._pretrained_model_name: str = pretrained_model_name

        self._train_ds: Dataset
        self._test_ds: Dataset
        self._train_ds, self._test_ds = self._randomize_train_and_test_datasets(
            train_ds=train_ds,
            test_ds=test_ds,
            shuffle_seed=shuffle_seed,
            selection_range=selection_range,
        )

        model: SetFitModel = SetFitModel.from_pretrained(pretrained_model_name)
        # TODO (8/18/2023): <Alex>The arguments loss_class, batch_size, num_iterations, and num_epochs can be made customizable in the future.</Alex>
        self._trainer: SetFitTrainer = SetFitTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            loss_class=CosineSimilarityLoss,
            batch_size=16,
            num_iterations=20,  # Number of text pairs to generate for contrastive learning
            num_epochs=1,  # Number of epochs to use for contrastive learning
        )

        self._model_training_complete: bool = False
        self._model_loaded: bool = False

    def train(self) -> dict[str, float]:
        """
        Trains the initialized SetFitModel using the initialized SetFitTrainer on the supplied train and test datasets.

        Returns:
            Dictionary of metrics values keyed by metric name obtained through running evaluation on the trained model.
        """
        logger.info(f'Training model "{self._model_name}".')
        self._trainer.train()

        metrics: dict[str, float] = self._trainer.evaluate()

        self._model_training_complete = True

        return metrics

    def persist_model(self) -> None:
        """
        Save (serialize) the trained model to the local filesystem using the model name as the base for the file name.
        """

        if not self._model_training_complete:
            raise ValueError(
                f'Unable to persist model "{self._model_name}", since it has not yet been trained.'
            )

        joblib.dump(self._trainer, f"{self._model_name}.joblib")

    def load_model(self, reload: bool = False) -> None:
        """
        Load (deserialize) the model from the local filesystem using the model name as the base for the file name.

        Args:
            reload: Directive to attempt to load the model from the filesystem, even if it already exists in memory.
        """

        do_load: bool
        if not self._model_training_complete:
            logger.info(
                f'Loading model "{self._model_name}" from most recently saved version.'
            )
            do_load = True
        elif reload:
            logger.info(
                f'Reloading model "{self._model_name}" from most recently saved version.'
            )
            do_load = True
        else:
            do_load = False

        if do_load:
            self._trainer = joblib.load(f"{self._model_name}.joblib")
            self._model_loaded = True

    def predict(self, text: str) -> int:
        """
        Runs inference (prediction) on the input text using the trained model.

        Args:
            text: Input text for the trained model to classify.

        Returns:
            Binary integer-valued output (0: "negative" or 1: "positive").
        """

        if not (self._model_training_complete or self._model_loaded):
            raise ValueError(
                f'Unable to use model "{self._model_name}" for inference, since it has not yet been trained or loaded.'
            )

        return self._trainer.model.predict([text])[0]

    @staticmethod
    def _randomize_train_and_test_datasets(
        train_ds: Dataset,
        test_ds: Dataset,
        shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
        selection_range: range | None = None,
    ) -> tuple[Dataset, Dataset]:
        train_ds = train_ds.shuffle(seed=shuffle_seed)
        test_ds = test_ds.shuffle(seed=shuffle_seed)

        if selection_range is not None:
            train_ds = train_ds.select(selection_range)
            test_ds = test_ds.select(selection_range)

        return train_ds, test_ds
