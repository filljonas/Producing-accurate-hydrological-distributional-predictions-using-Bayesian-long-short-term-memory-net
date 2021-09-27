"""
Trains and evaluates bayesian and non-bayesian neural networks

@author: Jonas Fill

Based on:
Title: Long Short-Term Memory (LSTM) for rainfall-runoff modelling
Author: Kratzert, Frederick
Date: 2020
Availability: https://github.com/kratzert/pangeo_lstm_example/blob/master/LSTM_for_rainfall_runoff_modelling.ipynb
"""

import torch
import numpy as np


def train_epoch_bbb(DEVICE, model, optimizer, loader, loss_func, num_samples):
    """Train model for a single epoch.

    :param model: A torch.nn.Module implementing the LSTM model
    :param optimizer: One of PyTorchs optimizer classes.
    :param loader: A PyTorch DataLoader, providing the trainings
        data in mini batches.
    :param loss_func: The loss function to minimize.
    :param epoch: The current epoch (int) used for the progress bar
    """
    # Set model to train mode (important for dropout) -> this is optional for BBB as no dropout is used
    model.train()
    # Keep track of current mini-batch
    minibatch_counter = 0
    # Number of mini-batches in the DataLoader
    num_batches = len(loader)
    # Global losses over all mini-batches
    global_loss = 0
    global_likelihood_cost = 0
    global_complexity_cost = 0
    # Keep track of total number of samples
    total_samples = 0
    # Request mini-batch of data from the loader
    for xs, ys in loader:
        minibatch_counter += 1
        # Delete previously stored gradients from the model
        optimizer.zero_grad()
        # Push data to GPU (if available)
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        complexity_cost_weight = (2 ** (num_batches - minibatch_counter)) / ((2 ** num_batches) - 1)
        # Get model predictions
        _, loss, likelihood_cost, complexity_cost = model.sample_elbo(inputs=xs,
                                                                      labels=ys,
                                                                      criterion=loss_func,
                                                                      #  In https://www.nitarshan.com/bayes-by-backprop/ the authors take 2 samples
                                                                      sample_nbr=num_samples,
                                                                      complexity_cost_weight=complexity_cost_weight)
        # Calculate gradients
        loss.backward()
        # Update the weights
        optimizer.step()
        # Update global losses
        global_loss += loss.item()
        global_likelihood_cost += likelihood_cost.item()
        global_complexity_cost += complexity_cost.item()
        total_samples += len(xs)
    return global_loss / total_samples, global_likelihood_cost / total_samples, global_complexity_cost / total_samples


def eval_model_bbb(DEVICE, model, loader, loss_func, num_training_samples, batch_size, sample_nbr=30):
    """Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.

    :return: Two torch Tensors, containing the observations and
        model predictions
    """
    model.eval()
    obs = []
    preds = []
    # Global losses over all mini-batches
    global_loss = 0
    global_likelihood_cost = 0
    global_complexity_cost = 0
    # Number of mini-batches in the DataLoader
    num_batches = len(loader)
    # Keep track of total number of samples
    num_samples = 0
    # inference mode -> no backpropagation
    with torch.no_grad():
        minibatch_counter = 0
        # request mini-batch of data from the loader
        for xs, ys in loader:
            minibatch_counter += 1
            # push data to GPU (if available)
            xs, ys = xs.to(DEVICE), ys.to(DEVICE)
            complexity_cost_weight = (2 ** (num_batches - minibatch_counter)) / ((2 ** num_batches) - 1)
            # Get model predictions and loss
            y_hat, loss, likelihood_cost, complexity_cost = model.sample_elbo(inputs=xs,
                                                                              labels=ys,
                                                                              criterion=loss_func,
                                                                              sample_nbr=sample_nbr,
                                                                              complexity_cost_weight=complexity_cost_weight)
            obs.append(ys.cpu())
            preds.append(y_hat)
            # Update global losses
            global_loss += loss.item()
            global_likelihood_cost += likelihood_cost.item()
            global_complexity_cost += complexity_cost.item()
            num_samples += len(xs)
    return torch.cat(obs), np.concatenate(tuple(x for x in preds), axis=1), \
           global_loss / num_samples, global_likelihood_cost / num_samples, global_complexity_cost / num_samples

