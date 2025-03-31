import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pickle as pkl
import base64

def get_test_inferences(model, test_loader, device = 'cuda'):
    """
    Get inferences from the model
    """
    model.eval()
    model = model.to(device)
    inferences = []

    for batch in test_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            inferences.append(outputs)

    inferences = torch.cat(inferences, dim=0)

    return inferences

def get_pl_test(model, test_loader):
    trainer = Trainer()
    trainer.test(model, test_loader)
    return model.test_results

def train_model(model, configs, epochs, train_loader, val_loader, test_loader, no_trials, callbacks = None, device = 'cuda'):
    """
    Args:
    - model: PyTorch model factory
    - configs: dictionary containing model-name and model-parameters
    - train_loader: PyTorch DataLoader
    - val_loader: PyTorch DataLoader
    - test_loader: PyTorch DataLoader

    Returns:
    - results: dataframes containing training, validation and test results
    """

    results = {}

    for config in configs.keys():
        model_results = []
        for trial in range(no_trials):
            model = model(configs[config])
            trainer = Trainer(max_epochs=epochs, accelerator=device, callbacks=callbacks)
            trainer.fit(model, train_loader, val_loader)
            model_results.append(get_pl_test(model, test_loader))

        results[config] = model_results

    return results

def formated_results(results):
    """
    Args:
    - results: array of dictionaries containing model results
    """
    formatted_results = {}
    for model in results.keys():
        test_results = results[model]
        test_results = pd.DataFrame(test_results)
        formatted_results[model] = test_results.mean(axis=0).round(2).astype(str) + '+/-' + test_results.std(axis=0).round(2).astype(str)

    return formatted_results



