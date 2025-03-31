import numpy as np
import pandas as pd
from dgl.dataloading import GraphDataLoader
import pytorch_lightning as pl

from Dataset.GraphDataset import GymPoseDataset

from Model.GraphModel import ALIGNN
from Model.Config import BaseConfig, BaseConfigMLP, SmallConfig, SmallConfigMLP

# class weight finder
from sklearn.utils.class_weight import compute_class_weight

# tqdm 
from tqdm import tqdm   
import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import json

import argparse

from pytorch_lightning.loggers import WandbLogger

datasets_list_path = "data/datasets.json"
with open(datasets_list_path, "r") as f:
    datasets_list = json.load(f)

datasets_list = [d for d in datasets_list if d["type"] == "3d"]

model_configs = [
    {
        "name": "BaseConfig",
        "config": BaseConfig
    },
    {
        "name": "BaseConfigMLP",
        "config": BaseConfigMLP
    },
    {
        "name": "SmallConfig",
        "config": SmallConfig
    },
    {
        "name": "SmallConfigMLP",
        "config": SmallConfigMLP
    }
]

def build_datasets(dataset_lists, data_path, preprocessed_path):
    for dataset in dataset_lists:
        # read_csv
        ann_df = pd.read_csv(os.path.join(data_path, dataset["annotations_file"]))
        points = np.load(os.path.join(data_path, dataset["points_file"]))

        train_df = ann_df[ann_df["split"] == "train"]
        val_df = ann_df[ann_df["split"] == "val"]
        test_df = ann_df[ann_df["split"] == "test"]

        ws = compute_class_weight(class_weight="balanced", classes=np.unique(train_df["label"]), y=train_df["label"])
        ws = torch.tensor(ws, dtype=torch.float32)

        train_ws = (train_df['label'].value_counts() / len(train_df)).values
        train_ws = torch.tensor(train_ws, dtype=torch.float32)
        train_ws = 1 - train_ws

        # ONLY TO TEST FUNCTIONALITY
        # cut the dfs to 40 samples
        # train_df = train_df.head(40)
        # val_df = val_df.head(40)
        # test_df = test_df.head(40)

        train_dataset = GymPoseDataset(
            train_df, points, dataset["name"] + "_train", preprocessed_path
        )
        val_dataset = GymPoseDataset(
            val_df, points, dataset["name"] + "_val", preprocessed_path
        )
        test_dataset = GymPoseDataset(
            test_df, points, dataset["name"] + "_test", preprocessed_path
        )

        yield train_dataset, val_dataset, test_dataset, train_ws


def run(
    model_configs, 
    datasets_list: list, 
    tries: int,  
    batch_size: int = 32,
    num_workers: int = 16,
    accelerator: str = "cuda",
    save_results_on_runtime: bool = True
):
    total_runs = len(model_configs) * len(datasets_list) * tries
    print(f"Total runs: {total_runs}")
    print(f"Number of model_configs: {len(model_configs)}")
    print(f"Number of datasets: {len(datasets_list)}")
    print(f"Tries: {tries}")
    input("Press enter to continue...")

    all_results = []

    for model_config in model_configs:
        for dataset in datasets_list:
            for try_i in range(tries):
                for train, val, test, ws in build_datasets([dataset], "data", "preprocessed_dataset"):
                    # set up the dataloaders
                    train_loader = GraphDataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=num_workers > 0)
                    val_loader = GraphDataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=num_workers > 0)
                    test_loader = GraphDataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)

                    # set up the model
                    model = ALIGNN(
                        config=model_config["config"], weights=ws
                    )

                    print("model_config: ", model_config)
                    
                    checkpoint_callback = ModelCheckpoint(
                        monitor='val_f1',
                        filename=f'{model_config["name"]}_{dataset["name"]}_{try_i}',
                        save_top_k=3,
                        mode='max'
                    )

                    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='step')

                    # set up the trainer
                    trainer = pl.Trainer(
                        max_epochs=model_config["config"].epochs,
                        accelerator=accelerator,
                        callbacks=[checkpoint_callback, lr_logger]
                    )

                    # train and evaluate the model
                    trainer.fit(model, train_loader, val_loader)

                    best_model_paths = checkpoint_callback.best_k_models

                    for k, best_model_path in enumerate(best_model_paths):
                        k_model = ALIGNN.load_from_checkpoint(best_model_path, config=model_config["config"], weights=ws)
                        k_model.eval()
                        results = trainer.test(model=k_model, dataloaders=test_loader)

                        print("results: ", results)

                        all_results.append({
                            "model": model_config["name"],
                            "dataset": dataset["name"],
                            "try": try_i,
                            "k_best": k,
                            "path": best_model_path,
                            "results": results
                        })

                        if save_results_on_runtime:
                            # append the results to the file
                            with open("results.json", "w") as f:
                                json.dump(all_results, f)

    return all_results

# add args for tries batch_size, num_workers and accelerator
parser = argparse.ArgumentParser()

parser.add_argument("--tries", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=30)
parser.add_argument("--accelerator", type=str, default="cuda")
parser.add_argument("--save_results_on_runtime", type=bool, default=True)

results = run(
    model_configs, 
    datasets_list, 
    parser.parse_args().tries, 
    parser.parse_args().batch_size, 
    parser.parse_args().num_workers,
    parser.parse_args().accelerator,
    parser.parse_args().save_results_on_runtime
)

print("results: ", results)
# save the results
with open("results.json", "w") as f:
    json.dump(results, f)