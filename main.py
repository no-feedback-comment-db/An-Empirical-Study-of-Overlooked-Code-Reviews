"""
    Python script for pull request comments classification.

    You can train and validate the models mentioned in the paper
    by specifying arguments such as `model`.

    Here is an example of reproducing the `gatting` model in the paper.

    ```shell
        python3 main.py --task "gatting_10fold" \
                        --data_root "./data" \
                        --result_root "./reproducing_results"\
                        --model "gatting" \
                        --global_seed 413 \
                        --train_epochs 3 \
                        --learning_rate 1e-5 \
                        --over_write
    ```

    You can also export the running logs by redirecting the output to log file.

    ```shell
        python3 main.py --task "gatting_10fold" \
                        --data_root "./data" \
                        --result_root "./reproducing_results"\
                        --model "gatting" \
                        --global_seed 413 \
                        --train_epochs 3 \
                        --learning_rate 1e-5 \
                        --over_write > gatting_10fold.log 2>&1 &
    ```
"""
import os
import pandas
import torch
import random
import warnings
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from datasets import load_dataset
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorWithPadding,
)
from multimodal_transformers.model import AutoModelWithTabular, TabularConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from transformers import logging

    logging.set_verbosity_error()
except ImportError:
    pass


@dataclass
class Config:
    # Model selection:
    # You can choose one of `comment only`, `concat`, `concat+mlp`, `ml+concat` or `gatting`.
    model: str

    # Task name (the name for result directory).
    task: str = f"task-{datetime.now().strftime('%m-%d(%H:%M:%S)')}"

    # Dataset root.
    data_root: str = "./data"

    # Dataset file name.
    data_file: str = "dataset.csv"

    # Result root.
    result_root: str = "./results"

    # You can choose any BERT model available on Hugging Face.
    bert_model: str = "bert-base-uncased"

    # Training/testing batch size.
    batch_size: int = 32

    # Whether to use cuda to accelerate training and testing.
    use_cuda: bool = torch.cuda.is_available()

    # Global seed for reproducing.
    global_seed: int = 42

    # Global learning rate.
    learning_rate: float = 1e-5

    # Total epochs for training.
    train_epochs: int = 5

    # If `over_write` is set to `True`,
    # then the script will overwrite the contents of the `task` directory.
    over_write: bool = False


class PRClassifier(nn.Module):
    """
    Model used to classify PR comments into
    two categories: those with feedback and those without feedback.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        bert_config = AutoConfig.from_pretrained(config.bert_model)
        combine_methods = {
            "comment only": "text_only",
            "concat": "concat",
            "concat+mlp": "mlp_on_concatenated_cat_and_numerical_feats_then_concat",
            "mlp+concat": "individual_mlps_on_cat_and_numerical_feats_then_concat",
            "gatting": "gating_on_cat_and_num_feats_then_sum",
        }
        tabular_config = TabularConfig(
            combine_feat_method=combine_methods[config.model],
            num_labels=2,
            numerical_feat_dim=3,
            cat_feat_dim=3,
        )
        bert_config.tabular_config = tabular_config
        self.model = AutoModelWithTabular.from_pretrained(
            config.bert_model, config=bert_config
        )

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        labels,
        numerical_features=None,
        category_features=None,
        **kwargs,
    ):
        if self.config.model == "comment only":
            loss, _, layer_out = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
                numerical_feats=None,
                cat_feats=None,
            )
        else:
            loss, _, layer_out = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
                numerical_feats=numerical_features,
                cat_feats=category_features,
            )
        return loss, layer_out[1]


class PRDataset(Dataset):
    def __init__(self, data_file: str, conifg: Config) -> None:
        super().__init__()
        self.config = conifg
        self.dataset = load_dataset("csv", data_files=data_file)["train"]
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.preprocess()

    def preprocess(self):
        def tokenize_func(example):
            """Tokenize text features"""
            return self.tokenizer(example["comment"], padding=True, truncation=True)

        def convert_label(example):
            """Convert str labels to numerical labels"""
            return {"numerical_label": int(example["label"])}

        def extract_features(example):
            return {
                "numerical_features": [
                    example["repeat words"],
                    example["author reply interval(project)"],
                    example["author reply interval(PR)"],
                ],
                "category_features": [
                    example["comment after merge(close)?"],
                    example["same with last comment?"],
                    example["declare reviewer's comment?"],
                ],
            }

        self.dataset = self.dataset.map(tokenize_func)
        self.dataset = self.dataset.map(convert_label)
        self.dataset = self.dataset.map(extract_features)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {
            "input_ids": self.dataset[index]["input_ids"],
            "token_type_ids": [0] * len(self.dataset[index]["input_ids"]),
            "attention_mask": self.dataset[index]["attention_mask"],
            "labels": self.dataset[index]["numerical_label"],
            "numerical_features": self.dataset[index]["numerical_features"],
            "category_features": self.dataset[index]["category_features"],
        }


def prepare_environment(config: Config):
    """
    prepare runtime environment for training and testing.

    Args:
        config (Config): the global config.
    """
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # check data root available
    if not os.path.exists(config.data_root):
        raise RuntimeError(f"data root {config.data_root} not exist")

    # check data file available
    config.data_file = os.path.join(config.data_root, config.data_file)
    if not os.path.exists(config.data_file):
        raise RuntimeError(f"data file {config.data_file} not exist")

    # check model available
    config.model = config.model.lower()
    assert config.model in (
        "comment only",
        "concat",
        "concat+mlp",
        "mlp+concat",
        "gatting",
    ), f"unsported model {config.model}"

    # make directory for results saving
    if not os.path.exists(config.result_root):
        os.makedirs(config.result_root)
        warnings.warn(
            f"result root {config.result_root} not exist, automatically make one"
        )

    # make directory for task saving
    config.task_root = os.path.join(config.result_root, config.task)
    if os.path.exists(config.task_root) and not config.over_write:
        raise RuntimeError(
            f"task {config.task_root} already exist, if you want to over write it, set over_write to True"
        )
    else:
        os.makedirs(config.task_root, exist_ok=True)

    # setup global random seed for reproducing
    random.seed(config.global_seed)
    torch.manual_seed(config.global_seed)
    np.random.seed(config.global_seed)
    if config.use_cuda:
        torch.cuda.manual_seed_all(config.global_seed)

    # determine training/testing device
    if config.use_cuda and torch.cuda.is_available():
        config.device = torch.device("cuda")
    else:
        config.device = "cpu"


def to_device(batch: dict, device):
    """
    Move a batch to specified device (cuda / cpu).

    Args:
        batch (dict[str, torch.Tensor]): the batch to be moved.
        device (torch.divice): the target divice to move.
    """
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    else:
        return batch.to(device=device)


def train(
    model: nn.Module, train_loader: DataLoader, config: Config, fold_num: int = None
):
    """
    Train model.

    Args:
        model (nn.Module): the model to be trained.
        train_loader (DataLoader): the dataloader of training data.
        config (Config): traning config.
        fold_num (Optional[int]): use to generage a better log. Defaults to `None`.
    """
    # prepare for training
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    model.to(device=config.device)

    # start training
    for epoch in range(1, config.train_epochs + 1):
        model.train()
        optimizer.zero_grad()

        tqdm_desc = (
            f"[fold {fold_num} epoch {epoch}]"
            if fold_num is not None
            else f"[epoch {epoch}]"
        )
        for batch in tqdm(train_loader, total=len(train_loader), desc=tqdm_desc):
            batch = to_device(batch, config.device)
            loss, _ = model(**batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if fold_num is not None:
            print(f"[fold {fold_num} epoch {epoch}]: loss = {loss:.5f}")
        else:
            print(f"[epoch {epoch}]: loss = {loss:.5f}")

    torch.save(
        {"state_dict": model.state_dict()},
        os.path.join(config.task_root, "saved_model.pt"),
    )
    print(
        f"traning finished, model saved to {os.path.join(config.task_root, 'saved_model.pt')}"
    )


def test(
    model: nn.Module, test_loader: DataLoader, config: Config, fold_num: int = None
) -> dict:
    """
    Test model.

    Args:
        model (nn.Module): the model to test.
        test_loader (DataLoader): test data loader.
        config (Config): test arguments.

    Return:
        dict[str, float]: test metrics dict.
    """
    model.eval()
    model.to(device=config.device)
    pred_labels, true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = to_device(batch, config.device)
            _, out = model(**batch)
            preds = (
                torch.nn.functional.softmax(out, dim=1)
                .argmax(dim=1)
                .cpu()
                .detach()
                .numpy()
            )
            pred_labels.append(preds)
            true_labels.append(batch["labels"].cpu().detach().numpy())

    pred_labels = [x for xx in pred_labels for x in xx]
    true_labels = [x for xx in true_labels for x in xx]

    test_metrics = {
        "accuracy": accuracy_score(y_true=true_labels, y_pred=pred_labels),
        "pos_precision": precision_score(
            y_true=true_labels, y_pred=pred_labels, pos_label=1
        ),
        "pos_recall": recall_score(y_true=true_labels, y_pred=pred_labels, pos_label=1),
        "pos_f1": f1_score(y_true=true_labels, y_pred=pred_labels, pos_label=1),
        "neg_precision": precision_score(
            y_true=true_labels, y_pred=pred_labels, pos_label=0
        ),
        "neg_recall": recall_score(y_true=true_labels, y_pred=pred_labels, pos_label=0),
        "neg_f1": f1_score(y_true=true_labels, y_pred=pred_labels, pos_label=0),
    }

    if fold_num is not None:
        print(f"[fold {fold_num}]: test finished!")
    else:
        print("test finished!")
    for metric_name, metric_value in test_metrics.items():
        if fold_num is not None:
            print(f"[fold {fold_num}]: {metric_name} = {round(metric_value*100, 2)}%")
        else:
            print(f"{metric_name} = {round(metric_value*100, 2)}%")
    return test_metrics


def get_dataloader(config: Config, train_mode: bool = False, test_mode: bool = False):
    """
    Create `torch.utils.data.Dataset` and generage corresponding `torch.utils.data.DataLoader`
    for training and testing.


    Args:
        config (Config): the global config.
        train_mode (bool): generate dataloader for traning.
                            There must be only one `True` value between `train_mode` and `test_mode`.
        test_mode (bool): generate dataloader for testing.
                            There must be only one `True` value between `train_mode` and `test_mode`.

    Return:
        torch.utils.data.DataLoader: the desired dataloader.
    """
    assert (
        train_mode or test_mode
    ), "load_dataset: `train_mode` and `test_mode` should not bot be `False`"
    assert not (
        train_mode and test_mode
    ), "load_dataset: `train_mode` and `test_mode` should not bot be `True`"

    if train_mode:
        dataset = PRDataset(os.path.join(config.data_root, "train.csv"), config)
    else:
        dataset = PRDataset(os.path.join(config.data_root, "test.csv"), config)

    def collate_function(batch):
        bert_features = defaultdict(list)
        other_features = defaultdict(list)

        for item in batch:
            for key, value in item.items():
                if key in ("input_ids", "token_type_ids", "attention_mask", "labels"):
                    bert_features[key].append(value)
                else:
                    other_features[key].append(value)

        bert_features = dataset.data_collator(bert_features)

        numerical_features = torch.tensor(
            other_features["numerical_features"], dtype=torch.float32
        )
        category_features = torch.tensor(
            other_features["category_features"], dtype=torch.float32
        )
        bert_features["numerical_features"] = numerical_features
        bert_features["category_features"] = category_features
        return bert_features

    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True if train_mode else False,
        num_workers=3,
        collate_fn=collate_function,
    )


def run_stratified_10fold_cross_validation(config: Config):
    """
    Train and validate the specified model using the stratified 10-fold cross-validation algorithm
    based on the configuration information in the `config`, and generate the average performance metrics.

    Args:
        config (Config): the global config.

    Return:
        dict[str, float]: the average performance metrics dict.
    """
    total_dataset = pandas.read_csv(config.data_file)
    total_dataset_x = [[i] for i in range(len(total_dataset))]
    total_dataset_y = [int(row["label"]) for _, row in total_dataset.iterrows()]

    skf = StratifiedKFold(n_splits=10)
    cross_validation_metrics = []
    for fold_index, (train_index, test_index) in enumerate(
        skf.split(total_dataset_x, total_dataset_y)
    ):
        train_dataset = total_dataset.iloc[train_index]
        train_dataset.to_csv(os.path.join(config.data_root, "train.csv"), index=False)

        test_dataset = total_dataset.iloc[test_index]
        test_dataset.to_csv(os.path.join(config.data_root, "test.csv"), index=False)

        if os.path.exists(os.path.join(config.task_root, "saved_model.pt")):
            os.remove(os.path.join(config.task_root, "saved_model.pt"))

        print(f"start training for fold {fold_index + 1}")

        # training
        model = PRClassifier(config)
        train(
            model=model,
            train_loader=get_dataloader(config, train_mode=True),
            config=config,
            fold_num=fold_index + 1,
        )

        # testing
        model = PRClassifier(config)
        state_dict = torch.load(os.path.join(config.task_root, "saved_model.pt"))[
            "state_dict"
        ]
        model.load_state_dict(state_dict)
        metrics = test(
            model=model,
            test_loader=get_dataloader(config, test_mode=True),
            config=config,
            fold_num=fold_index + 1,
        )
        cross_validation_metrics.append(metrics)
        print()

    def average_metrics(metrics: list):
        average_metrics = defaultdict(list)
        for metric in metrics:
            for k, v in metric.items():
                average_metrics[k].append(v)
        for key in average_metrics:
            average_metrics[key] = np.mean(average_metrics[key])
        return average_metrics

    print("stratified 10 fold finished!")
    return average_metrics(cross_validation_metrics)


if __name__ == "__main__":
    # parse arguments and prepare environments
    parser = HfArgumentParser(Config)
    config = parser.parse_args_into_dataclasses()[0]
    prepare_environment(config)

    # running stratified 10-fold cross-validation
    results = run_stratified_10fold_cross_validation(config)

    # export average performance metrics dict
    try:
        from prettytable import PrettyTable

        table = PrettyTable(
            [
                "Model",
                "Accuracy",
                "Positive Precision",
                "Positive Recall",
                "Positive F1-Score",
                "Negative Precision",
                "Negative Recall",
                "Negative F1-Score",
            ]
        )
        table.add_row(
            [
                config.model,
                f"{round(results['accuracy'] * 100, 2)}%",
                f"{round(results['pos_precision'] * 100, 2)}%",
                f"{round(results['pos_recall'] * 100, 2)}%",
                f"{round(results['pos_f1'] * 100, 2)}%",
                f"{round(results['neg_precision'] * 100, 2)}%",
                f"{round(results['neg_recall'] * 100, 2)}%",
                f"{round(results['neg_f1'] * 100, 2)}%",
            ]
        )
        print(table)
    except ImportError:
        for metric_name, metric_value in results.items():
            print(f"average {metric_name} = {round(metric_value * 100, 2)}%")
