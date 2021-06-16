from abc import ABC
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from .al_utils import ActiveLearningDataset
import pickle
import numpy as np


class QueryStrategy(ABC):
    "Abstract class"
    def query_labels(
        self,
        model: pl.LightningModule,
        data_module: pl.LightningDataModule,
        dataset: ActiveLearningDataset,
        logger: pl.loggers,
        query_size: int,
    ) -> None:
        pass


class Random(QueryStrategy):
    def query_labels(
        self,
        model: pl.LightningModule,
        data_module: pl.LightningDataModule,
        dataset: ActiveLearningDataset,
        logger: pl.loggers,
        query_size: int,
    ) -> None:
        print("Doing random sampling...")
        added_yield, added_input, added_index = dataset.label_randomly(query_size)

        with open(f"{logger.log_dir}/added_yield.pkl", "wb") as f:
            pickle.dump(added_yield, f)

        with open(f"{logger.log_dir}/added_input.pkl", "wb") as f:
            pickle.dump(added_input, f)
        print("Done.")


class Margin(QueryStrategy):
    def query_labels(
        self,
        model: pl.LightningModule,
        data_module: pl.LightningDataModule,
        dataset: ActiveLearningDataset,
        logger: pl.loggers,
        query_size: int,
    ) -> None:

        x = dataset.get_pool().cuda()
        model.eval()
        logits = model(x)
        probabilities = F.softmax(logits, dim=1)

        toptwo = torch.topk(probabilities, 2, dim=1)[0]

        differences = toptwo[:, 0] - toptwo[:, 1]
        margins = torch.abs(differences).cpu().detach().numpy()

        margins_sorted_index = np.argsort(margins, axis=0)

        query_index = margins_sorted_index[:query_size]

        added_yield, added_input = dataset.label(query_index)

        with open(f"{logger.log_dir}/added_yield.pkl", "wb") as f:
            pickle.dump(added_yield, f)

        with open(f"{logger.log_dir}/added_margin.pkl", "wb") as f:
            pickle.dump(margins[query_index], f)

        with open(f"{logger.log_dir}/added_input.pkl", "wb") as f:
            pickle.dump(added_input, f)
