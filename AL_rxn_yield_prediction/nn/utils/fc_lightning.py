import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


class ResetCallback(pl.callbacks.Callback):
    """Callback to reset the weights between active learning steps.
    Args:
        weights (dict): State dict of the model.
    Notes:
        The weight should be deep copied beforehand.
    """

    def __init__(self, weights):
        self.weights = weights

    def on_train_start(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        """Will reset the module to its initial weights."""
        module.load_state_dict(self.weights)


class FullyConnected(pl.LightningModule):
    """Feed-forward fully connect neural network."""

    def __init__(self, layer_size, dropout, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()

        self.hparams.n_layers = len(self.hparams.layer_size)

        print(self.hparams.layer_size)

        modules = []
        modules.append(
            nn.Linear(self.hparams.layer_size[0], self.hparams.layer_size[1])
        )
        for i in range(1, self.hparams.n_layers - 1):
            modules.append(nn.Dropout(p=self.hparams.dropout, inplace=True))
            modules.append(nn.LeakyReLU(inplace=True))
            modules.append(
                nn.Linear(self.hparams.layer_size[i], self.hparams.layer_size[i + 1])
            )

        self.fc = nn.Sequential(*modules)

        print(self.fc)

    def forward(self, x):
        logits = self.fc(x)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        probabilities = F.softmax(logits, dim=1)

        labels_pred = torch.argmax(probabilities, dim=1)
        n_correct_pred = torch.sum(y == labels_pred).item()

        logs = {}

        return {
            "loss": loss,
            "n_correct_pred": n_correct_pred,
            "n_pred": len(y),
            "log": logs,
        }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        train_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(
            x["n_pred"] for x in outputs
        )
        logs = {
            "loss": avg_loss,
            "train_acc": train_acc,
            "train_loss": avg_loss,
            "step": self.current_epoch,
        }
        self.log_dict(logs)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        val_loss = F.cross_entropy(logits, y)

        val_avg_pr = pl.metrics.functional.classification.average_precision(
            probabilities[:, 1],
            y,
        )
        val_auc_roc = pl.metrics.functional.classification.auroc(
            probabilities[:, 1],
            y,
        )

        labels_pred = torch.argmax(probabilities, dim=1)
        n_correct_pred = torch.sum(y == labels_pred).item()

        return {
            "val_loss": val_loss,
            "val_avg_pr": val_avg_pr,
            "val_auc_roc": val_auc_roc,
            "n_correct_pred": n_correct_pred,
            "n_pred": len(y),
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_avg_pr = torch.stack([x["val_avg_pr"] for x in outputs]).mean()
        avg_auc_roc = torch.stack([x["val_auc_roc"] for x in outputs]).mean()

        val_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(
            x["n_pred"] for x in outputs
        )
        logs = {
            "val_loss": avg_loss,
            "val_acc": val_acc,
            "step": self.current_epoch,
            "val_avg_pr": avg_avg_pr,
            "val_auc_roc": avg_auc_roc,
        }
        self.log_dict(logs)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = x.view(x.size(0), -1)
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)

        probabilities_sucessful = probabilities[:, 1].detach().cpu().numpy()
        with open(
            f"{self.logger.log_dir}/probabilities_sucessful_test_{batch_idx}.pkl", "wb"
        ) as f:
            pickle.dump(probabilities_sucessful, f)

        test_loss = F.cross_entropy(logits, y)

        test_avg_pr = pl.metrics.functional.classification.average_precision(
            probabilities[:, 1],
            y,
        )
        test_auc_roc = pl.metrics.functional.classification.auroc(
            probabilities[:, 1],
            y,
        )

        labels_pred = torch.argmax(probabilities, dim=1)
        n_correct_pred = torch.sum(y == labels_pred).item()

        return {
            "test_loss": test_loss,
            "test_avg_pr": test_avg_pr,
            "test_auc_roc": test_auc_roc,
            "n_correct_pred": n_correct_pred,
            "n_pred": len(y),
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(
            x["n_pred"] for x in outputs
        )
        avg_avg_pr = torch.stack([x["test_avg_pr"] for x in outputs]).mean()
        avg_auc_roc = torch.stack([x["test_auc_roc"] for x in outputs]).mean()
        logs = {
            "test_loss": avg_loss,
            "test_acc": test_acc,
            "step": self.current_epoch,
            "test_avg_pr": avg_avg_pr,
            "test_auc_roc": avg_auc_roc,
        }
        self.log_dict(logs)