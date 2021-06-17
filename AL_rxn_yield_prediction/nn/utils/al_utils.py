import torch
import numpy as np
from typing import Union, Optional, List
import pytorch_lightning as pl
import torch.nn.functional as F
import pickle


class ActiveLearningLoop:
    def __init__(
        self,
        data_module: pl.LightningDataModule,
        query_strategy,
        query_size: int,
        max_n_labelled: int,
        random_state=None,
    ) -> None:
        self.data_module = data_module
        self.dataset = data_module.train
        self.query_strategy = query_strategy
        self.query_size = query_size
        self.random_state = random_state
        self.max_n_labelled = max_n_labelled
        self.model = None
        self.init_model_state = None

    def set_model_and_init_state(
        self, model: pl.LightningModule, init_model_state
    ) -> None:
        self.model = model
        self.init_model_state = init_model_state

    def step(self, logger: pl.loggers, query_size: int = None) -> bool:

        assert self.model is not None, "No model chosen for step"
        assert (
            self.init_model_state is not None
        ), "No initial model state chosen for step"

        x = self.data_module.train_and_val_dataset().input.cuda()
        self.model.cuda().eval()
        logits = self.model(x)
        probabilities = F.softmax(logits, dim=1)

        probabilities_sucessful = probabilities[:, 1].detach().cpu().numpy()
        with open(
            f"{logger.log_dir}/probabilities_sucessful_train_and_val.pkl", "wb"
        ) as f:
            pickle.dump(probabilities_sucessful, f)

        if (self.dataset.n_unlabelled <= 0) or (
            self.dataset.n_labelled >= self.max_n_labelled
        ):
            return False

        logger.log_metrics({"n_labelled": self.dataset.n_labelled})

        logger.save()

        if query_size is None:
            query_size = self.query_size

        self.query_strategy.query_labels(
            self.model,
            self.data_module,
            self.dataset,
            logger,
            query_size,
        )

        return True

class ActiveLearningDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, labelled=None):
        self._dataset = dataset

        if labelled is not None:
            if isinstance(labelled, torch.Tensor):
                labelled = labelled.numpy()  # May need to do detach().cpu().numpy()
            self._labelled = labelled.astype(np.bool)
        else:
            self._labelled = np.zeros(len(self._dataset), dtype=np.bool)

    def __len__(self) -> int:
        return self._labelled.sum()

    def __getitem__(self, index: int):
        return self._dataset[self._labelled_to_oracle_index(index)]

    def _pool_to_oracle_index(self, index: Union[int, List[int]]) -> List[int]:
        if isinstance(index, np.int64) or isinstance(index, int):
            index = [index]

        oracle_idx = (~self._labelled).nonzero()[0]

        return [int(oracle_idx[idx].squeeze().item()) for idx in index]

    def _labelled_to_oracle_index(self, index: int):
        return self._labelled.nonzero()[0][index].squeeze().item()

    def _oracle_to_pool_index(self, index: Union[int, List[int]]) -> List[int]:
        assert self._labelled[index] == 0, f"Trying to label a labeled point: {index}"

        if isinstance(index, np.int64) or isinstance(index, int):
            index = [index]

        return [len((~self._labelled[:idx]).nonzero()[0]) for idx in index]

    def label(
        self,
        index: Union[List[int], int],
        target: Optional[Union[List[int], int]] = None,
    ):
        # Label points with index

        if isinstance(index, int):
            index = [index]

        if isinstance(target, int):
            target = [target]

        assert (
            len(index) <= self.n_unlabelled
        ), "Query size is larger than the number of unlabeled data points."

        indexes = self._pool_to_oracle_index(index)

        # Maybe can skip loop
        for i, idx in enumerate(indexes):
            assert self._labelled[idx] == 0, f"Trying to label a labeled point: {idx}"
            self._labelled[idx] = 1  # Should it be True?
            if target is not None:
                self._dataset.target[idx] = target[i]

        return self._dataset.yields[indexes], self._dataset.input_as_array[indexes]

    def unset_labels(self) -> None:
        self._labelled = np.zeros(len(self._dataset), dtype=np.bool)

    def label_randomly(self, query_size: int, random_state=None):
        rng = np.random.default_rng(seed=random_state)

        index = rng.choice(self.n_unlabelled, query_size, replace=False)

        added_yield, added_input = self.label(index)

        return added_yield, added_input, index


    def label_by_idx(self, index: Union[List[int], int]) -> None:
        if isinstance(index, np.int64) or isinstance(index, int):
            index = [index]

        for idx in index:
            assert (
                self._labelled[idx] == 0
            ), f"Trying to add a point by index that is already labeled: {idx}"
            self._labelled[idx] = 1

    @property
    def n_unlabelled(self):
        """The number of unlabelled data points."""
        return (~self._labelled).sum()

    @property
    def n_labelled(self):
        """The number of labelled data points."""
        return self._labelled.sum()

    def get_pool(self) -> torch.Tensor:

        pool_index = (~self._labelled).nonzero()[0].squeeze()

        pool_index = [idx for idx in pool_index]  # TODO is this necessary?

        pool = self._dataset.input[pool_index]

        return pool

    def get_all_train(self) -> torch.Tensor:

        return self._dataset.input

    def get_all_train_as_array(self):

        return self._dataset.input_as_array