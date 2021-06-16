import argparse
import numpy as np
import os
import shutil
import torch
import pickle
import pytorch_lightning as pl
import sklearn.model_selection
import copy
import threading

from utils.fc_lightning import FullyConnected, ResetCallback
from utils.al_utils import ActiveLearningDataset, ActiveLearningLoop
from utils.query_strategies import Random, Margin 




class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, input, target, yields=None):
        self.input = torch.as_tensor(input, dtype=torch.float)
        self.target = torch.as_tensor(target, dtype=torch.long)
        self.input_as_array = input
        self.yields = yields

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]

class MyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(
        self,
        test_size: float,
        val_size: float,
        batch_size: int,
        random_state=42,
    ) -> None:
        train_input_all = np.load(file=args.input_path_train)
        train_target_all = np.load(file=args.target_path_train)
        train_yields_all = np.load(file=args.yields_path)

        if args.start_idx_path is not None:
            self.start_idx = list(np.load(file=args.start_idx_path))

        self.batch_size = batch_size

        if (args.input_path_test is None) and (args.target_path_test is None):
            print("Creating train/test split from training data", flush=True)
            (
                train_input_all,
                test_input,
                train_target_all,
                test_target,
                train_yields_all,
                test_yields,
            ) = sklearn.model_selection.train_test_split(
                train_input_all,
                train_target_all,
                train_yields_all,
                test_size=test_size,
                random_state=random_state,
                stratify=train_target_all,
            )
        else:
            print("Loading test data")
            test_input = np.load(file=args.input_path_test)
            test_target = np.load(file=args.target_path_test)

        self.test = CreateDataset(test_input, test_target)

        self.train_and_validation = CreateDataset(
            train_input_all, train_target_all, train_yields_all
        )

        if val_size > 0:
            (
                train_input,
                val_input,
                train_target,
                val_target,
                train_yields,
                val_yields,
            ) = sklearn.model_selection.train_test_split(
                train_input_all,
                train_target_all,
                train_yields_all,
                test_size=val_size,
                random_state=random_state,
                stratify=train_target_all,
            )

            self.val = CreateDataset(val_input, val_target)
            train = CreateDataset(train_input, train_target, train_yields)

        else:
            self.val = None
            train = self.train_and_validation

        self.train = ActiveLearningDataset(train)

    def setup(self) -> None:
        # called on every GPU
        self.test_loader = torch.utils.data.DataLoader(
            self.test,
            batch_size=len(self.test),
            shuffle=False,
            pin_memory=True,
            num_workers=0,
        )
        if self.val is not None:
            self.val_loader = torch.utils.data.DataLoader(
                self.val, batch_size=len(self.val), pin_memory=True, num_workers=0
            )
        else:
            self.val_loader = None

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train, batch_size=self.batch_size, pin_memory=True, num_workers=0
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.val_loader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.test_loader

    def train_and_val_dataset(self):
        return self.train_and_validation


def main():

    random_state = args.random_state
    if random_state == None:
        random_state = [None for _ in range(args.n_predictions)]
    elif len(random_state) == args.n_predictions:
        pass
    elif len(random_state) == 1 and args.n_predictions > 1:
        random_state = [random_state for _ in range(args.n_predictions)]
    else:
        raise ValueError("Not enough random states given to argument parser")

    init_train_size = (
        args.query_size if args.initial_size is None else args.initial_size
    )

    layer_size = args.layer_size

    my_dm = MyDataModule()
    my_dm.prepare_data(
        test_size=0.2, val_size=args.val_size, batch_size=args.batch_size
    )
    my_dm.setup()

    input_size = my_dm.test.input.size()[1]
    layer_size.insert(0, input_size)

    if args.query_strategy == "random":
        query_strategy = Random()
    elif args.query_strategy == "margin":
        query_strategy = Margin()
    else:
        raise NotImplementedError("Selected query strategy is not implemented.")

    loop = ActiveLearningLoop(my_dm, query_strategy, args.query_size, args.labelled)

    ##################################################
    ############### ACTIVE LEARNING ##################
    ##################################################

    if args.start_prediction is None:
        start_prediction = 0
    else:
        start_prediction = args.start_prediction

    for i_prediction in range(start_prediction, args.n_predictions):

        model = FullyConnected(
            layer_size, args.dropout_prob, args.lr, args.weight_decay
        )

        if args.init_states_dir is None:
            init_model_state = copy.deepcopy(model.state_dict())

            if not os.path.exists(args.results_dir):
                os.makedirs(args.results_dir)

            with open(
                f"{args.results_dir}/init_model_state_prediction_{i_prediction}.pkl",
                "wb",
            ) as f:
                pickle.dump(init_model_state, f)
        else:
            with open(
                f"{args.init_states_dir}/init_model_state_prediction_{i_prediction}.pkl",
                "rb",
            ) as f:
                init_model_state = pickle.load(f)

        loop.set_model_and_init_state(model, init_model_state)

        my_dm.train.unset_labels()

        if args.start_idx_path is not None:
            my_dm.train.label_by_idx(my_dm.start_idx)
        else:
            my_dm.train.label_randomly(
                init_train_size, random_state=random_state[i_prediction]
            )

        print(f"Doing prediction {i_prediction+1}")

        if args.start_iteration is None:
            i_iteration = 0
        else:
            i_iteration = args.start_iteration

        while True:

            print(f"Doing AL iteration {i_iteration+1}")

            logger = pl.loggers.TensorBoardLogger(
                f"{args.results_dir}",
                name=f"pred{i_prediction}-AL{i_iteration}",
            )

            if my_dm.val_dataloader() is not None:
                early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
                    monitor="val_avg_pr",
                    min_delta=0.00,
                    patience=10,
                    verbose=False,
                    mode="max",
                )
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                    monitor="val_avg_pr",
                    filename="AL-{epoch:02d}",
                    save_top_k=1,
                    mode="max",
                )

                trainer = pl.Trainer(
                    max_epochs=args.epochs,
                    gpus=1,
                    logger=logger,
                    auto_select_gpus=True,
                    callbacks=[
                        early_stop_callback,
                        checkpoint_callback,
                        ResetCallback(init_model_state),
                    ],
                )
            else:
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                    filename="AL-{epoch:02d}",
                )
                trainer = pl.Trainer(
                    max_epochs=args.epochs,
                    gpus=1,
                    logger=logger,
                    auto_select_gpus=True,
                    num_sanity_val_steps=0,
                    callbacks=[
                        ResetCallback(init_model_state),
                        checkpoint_callback
                    ],
                )

            trainer.fit(model, my_dm)

            trainer.test()

            should_continue = loop.step(logger)

            logger.experiment.close()

            # Remove checkpoints of intermediate runs
            ckpt_dir = f"{args.results_dir}/checkpoints/"
            if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
                shutil.rmtree(ckpt_dir)

            print(f"Number of active threads: {threading.active_count()}")

            if not should_continue:
                print("AL stopping criterion is fulfilled. Stopping...")
                break

            i_iteration += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train neural network for binary synthesis prediction"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="How many epochs to train. Default: 50"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate. Default: 1e-3"
    )

    parser.add_argument(
        "--weight_decay",
        "-wd",
        type=float,
        default=0.01,
        help="Weight decay for optimzer AdamW. Default: 0.01",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="NN_results",
        help="Where to results and checkpoints. Default 'NN_results'",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size. Default: 8"
    )

    parser.add_argument(
        "--n_predictions",
        type=int,
        default=5,
        help="Number of times to retrain network and predict. Default = 5",
    )
    parser.add_argument(
        "--input_path_train",
        type=str,
        required=True,
        help="Path for input train data",
    )
    parser.add_argument(
        "--target_path_train",
        type=str,
        required=True,
        help="Path for target train data",
    )
    parser.add_argument(
        "--input_path_test",
        type=str,
        required=True,
        help="Path for input test data",
    )
    parser.add_argument(
        "--target_path_test",
        type=str,
        required=True,
        help="Path for target test data",
    )

    parser.add_argument(
        "--yields_path",
        type=str,
        required=True,
        help="Path for yields of training data",
    )

    parser.add_argument(
        "--start_idx_path",
        type=str,
        required=True,
        help="Path to list with indices for starting points",
    )

    parser.add_argument(
        "--init_states_dir",
        type=str,
        required=False,
        help="Director of initial model states",
    )

    parser.add_argument(
        "--query_size",
        type=int,
        default=1,
        help="Points to query in each iteration of Active Learning. Default = 1",
    )

    parser.add_argument(
        "--labelled",
        type=int,
        default=1500,
        help="Maximum number of points to label before quitting AL loop. Default = 1500",
    )

    parser.add_argument(
        "--initial_size",
        type=int,
        help="Size of the initial training set. Default = query size",
    )

    parser.add_argument(
        "--start_iteration",
        type=int,
        help="Iteration to start active learning from (important for logging). Starts at 0. Default = 0",
    )

    parser.add_argument(
        "--start_prediction",
        type=int,
        help="Prediction to start active learning from (important for logging). Starts at 0. Default = 0",
    )

    parser.add_argument(
        "--query_strategy",
        required=True,
        choices=[
            "margin",
            "random",
        ],
        help="Query strategy to use",
    )

    parser.add_argument(
        "--layer_size",
        nargs="+",
        required=True,
        help="Size of the hidden layers",
        type=int,
    )

    parser.add_argument(
        "--random_state",
        nargs="+",
        type=int,
        help="Random state. Otherwise None is used.",
    )

    parser.add_argument(
        "--dropout_prob",
        type=float,
        help="Dropout probability. Default = 0.5",
        default=0.5,
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Fraction of validation set. Default = 0.0",
        default=0.0,
    )

    args = parser.parse_args()

    main()