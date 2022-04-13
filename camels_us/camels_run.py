"""
Implements the Bayesian neural network adapted for the CAMELS-US-dataset

@author: Jonas Fill
"""

from camels_dataset import *
from models.bbb_model_camels import *
from train_eval import *
from neuralhydrology.metrics import calculate_all_metrics

from pathlib import Path

import time
import numpy as np
import xarray as xr
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Globals
CAMELS_ROOT = Path('../CAMELS/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # This line checks if GPU is available
# Other model-independent parameters for the run
PLOT_FREQUENCY = 1  # How often should we plot? (20 for instance means every 20 epochs)
SAVE_MODEL = False  # Should we save the model?
ONLY_EVAL = False  # Only evaluate the model, training was already done
# The root for these paths is the main subdirectory of this project
SAVE_PATH = 'path_to_folder_where_models_should_be_saved'
LOAD_PATH = 'path_to_pre_trained_model_to_load_for_evaluation'

# Define confidence interval for our predictions
ci_multiplier = 2

# Returns list of 241 catchments that are relevant for our evaluation
catchments = get_relevant_catchments()


class Hyperparameters():
    def __init__(self, hidden_size=50, num_layers=1, sequence_length=92, batch_size=256, optimizer='adam',
                 catchment='c_01013500', n_epochs=50, technique='bbb', learning_rate=0.004,
                 yearly_periodicity=True, io_behavior='many-to-one', discharge_as_input=False,
                 prior_sigma_1=10, posterior_rho_init=-2.5, prior_pi=1, num_samples=5):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.catchment = catchment
        self.n_epochs = n_epochs
        self.technique = technique
        self.learning_rate = learning_rate
        self.yearly_periodicity = yearly_periodicity
        self.io_behavior = io_behavior
        self.discharge_as_input = discharge_as_input
        self.prior_sigma_1 = prior_sigma_1
        self.posterior_rho_init = posterior_rho_init
        self.prior_pi = prior_pi
        self.num_samples = num_samples


"""
Define the hyperparameters for training here.
The catchment the model should be trained on is defined in the format 'c_<catchment_number>'.
The default setting performs training for 50 epochs with the hyperparameters proposed in the papers for the
'many-to-one'-case.
"""
hyperparameters_default = Hyperparameters(hidden_size=50, num_layers=1, sequence_length=92, batch_size=256,
                                          optimizer='adam', catchment='c_01013500', n_epochs=50,
                                          technique='bbb', learning_rate=0.004, yearly_periodicity=True,
                                          io_behavior='many-to-one', discharge_as_input=False, prior_sigma_1=10,
                                          posterior_rho_init=-2.5, prior_pi=1, num_samples=5)


def model_pipeline(config=hyperparameters_default):
    """
    Import and process data
    """
    # How many days in the future do we predict? (in case we use a many-to-many-architecture)
    if config.io_behavior.startswith('many-to-many'):
        days_future = int(config.io_behavior[config.io_behavior.rfind('-') + 1:])
    else:
        days_future = 1
    # Training data
    start_date = pd.to_datetime("1980-01-01", format="%Y-%m-%d")
    end_date = pd.to_datetime("1995-09-30", format="%Y-%m-%d")
    ds_train = CamelsTXT(config.catchment[2:], seq_length=config.sequence_length + days_future - 1, period="train",
                         dates=[start_date, end_date], yearly_periodicity=config.yearly_periodicity,
                         io_behavior=config.io_behavior, discharge_as_input=config.discharge_as_input)
    tr_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)

    # Means/stds of the training period for normalization
    means = ds_train.get_means()
    stds = ds_train.get_stds()

    # Test data. We use the feature means/stds of the training period for normalization
    start_date = pd.to_datetime("1995-10-01", format="%Y-%m-%d")
    end_date = pd.to_datetime("2010-09-30", format="%Y-%m-%d")
    ds_test = CamelsTXT(config.catchment[2:], seq_length=config.sequence_length + days_future - 1, period="eval",
                        dates=[start_date, end_date],
                        means=means, stds=stds, yearly_periodicity=config.yearly_periodicity,
                        io_behavior=config.io_behavior, discharge_as_input=config.discharge_as_input)
    test_loader = DataLoader(ds_test, batch_size=2048, shuffle=False)

    # Define model
    model = BayesByBackpropModel(config.hidden_size, num_layers=config.num_layers,
                                 yearly_periodicity=config.yearly_periodicity,
                                 io_behavior=config.io_behavior, discharge_as_input=config.discharge_as_input,
                                 prior_sigma_1=config.prior_sigma_1, posterior_rho_init=config.posterior_rho_init,
                                 prior_pi=config.prior_pi).to(DEVICE)
    # If only evaluation is done, load model parameters from a pre-trained model
    if ONLY_EVAL:
        model.load_state_dict(torch.load(f'../{LOAD_PATH}',
                                         map_location=torch.device('cpu')))
    # Choose optimizer
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.learning_rate)
    # Define data cost
    loss_func = nn.MSELoss()

    """
    Training
    """
    for i in range(config.n_epochs):
        # Training step
        if not ONLY_EVAL:
            loss, likelihood_cost, complexity_cost = train_epoch_bbb(DEVICE, model, optimizer, tr_loader, loss_func,
                                                                     config.num_samples)
            sys.stdout.write(f'Epoch {i + 1}: ')
            # Write current loss to console
            sys.stdout.write(f'Loss: {loss:.4f}'
                             f'  Likelihood cost: {likelihood_cost:.4f}'
                             f'  Complexity cost: {complexity_cost:.4f}   \n')
        # Perform evaluation if needed in current epoch
        if (i + 1) % PLOT_FREQUENCY == 0 or i == 0:
            """
            Evaluation
            """
            eval(config, model, test_loader, loss_func, ds_train, ds_test, days_future, i)
        # If only an evaluation is done, abort after the first step
        if ONLY_EVAL:
            break
        else:
            sys.stdout.write('\n-------------------------------------------------------------------------------------\n')
    return model


def eval(config, model, test_loader, loss_func, ds_train, ds_test, days_future, i):
    """
    Evaluation of accuracy
    """
    obs, preds, test_loss, test_likelihood_cost, test_complexity_cost = eval_model_bbb(DEVICE, model,
                                                                                       test_loader,
                                                                                       loss_func,
                                                                                       ds_test.num_samples,
                                                                                       2048)
    # Use correct scaling for predictions
    preds = ds_train.local_rescale(preds, variable='output')
    # Flatten and reshape arrays in case single-shot or many-to-many was used
    if config.io_behavior.startswith('single-shot') or config.io_behavior.startswith('many-to-many'):
        preds = preds.flatten().reshape(preds.shape[0], preds.shape[1] * days_future)
        obs = obs.flatten().reshape(-1, 1)
    # Calculate means and standard predictions of predictions
    preds_means = preds.mean(axis=0).reshape(-1, 1)
    preds_std = preds.std(axis=0).reshape(-1, 1)
    obs = obs.numpy()
    # Compute various evaluation metrics
    obs_xr = xr.DataArray(data=obs.flatten())
    preds_means_xr = xr.DataArray(data=preds_means.flatten())
    eval_metrics = calculate_all_metrics(obs_xr, preds_means_xr)
    nse = eval_metrics['NSE']
    mse = eval_metrics['MSE']
    rmse = eval_metrics['RMSE']
    kge = eval_metrics['KGE']
    alpha_nse = eval_metrics['Alpha-NSE']
    beta_nse = eval_metrics['Beta-NSE']
    pearson_r = eval_metrics['Pearson-r']
    fhv = eval_metrics['FHV']
    fms = eval_metrics['FMS']
    flv = eval_metrics['FLV']
    # Write evaluation metrics to console
    sys.stdout.write(f'Test loss: {test_loss}   Test likelihood cost: {test_likelihood_cost}'
                     f'    Test complexity cost: {test_complexity_cost}    Test NSE: {nse:.2f}\n'
                     f'Test MSE: {mse:.2f}      Test RMSE: {rmse:.2f}       Test KGE: {kge:.2f}     '
                     f'Test Alpha NSE: {alpha_nse:.2f}      Test Beta NSE: {beta_nse:.2f}       '
                     f'Test Pearson R: {pearson_r:.2f}      Test FHV: {fhv:.2f}     Test FMS: {fms:.2f}     '
                     f'Test FLV: {flv:.2f}\n')
    """
    Evaluation of uncertainty
    """
    ##
    # Compute prediction interval
    ##
    # Calculate lower and upper bound for confidence interval
    lower_bound = preds_means - (preds_std * ci_multiplier)
    upper_bound = preds_means + (preds_std * ci_multiplier)
    under_upper = upper_bound > obs
    over_lower = lower_bound < obs
    total = (under_upper == over_lower)
    sys.stdout.write(f'{np.mean(total) * 100:.2f} % of the obsereved values are in our confidence interval\n')
    #
    # Compute measures for resolution as well as gaussian log likelihood
    #
    preds_std_bessel = preds.std(axis=0, ddof=1).flatten()
    # Compute absolute resolution index
    tmp = 1 / preds_std_bessel
    absolute_resolution_index = tmp.mean()
    # Compute relative resolution index
    tmp = preds_means.flatten() / preds_std_bessel
    relative_resolution_index = tmp.mean()
    # Compute gaussian log likelihood
    tmp = ((preds_means.flatten() - obs.flatten()) ** 2) / (preds_std_bessel ** 2)
    gaussian_log_likelihood = tmp.mean()
    sys.stdout.write(f'Absolute resolution index: {absolute_resolution_index}\n'
                     f'Relative resolution index: {relative_resolution_index}\n'
                     f'Gaussian log likelihood: {gaussian_log_likelihood}\n')
    preds_flattened = np.reshape(preds, (-1, 1))
    #
    # Compute KL divergence
    #
    # Create histograms
    all_values = np.concatenate((preds_flattened, obs))
    min_value = all_values.min()
    max_value = all_values.max()
    diff = max_value - min_value
    bin_size = diff / 400
    edges = np.arange(min_value, max_value + math.exp(-6), bin_size)[:401]
    hist_pred, bin_edges_pred = np.histogram(preds_flattened.flatten(), edges, density=True)
    hist_obs, bin_edges_obs = np.histogram(obs.flatten(), edges, density=True)
    hist_pred_norm = hist_pred * np.diff(bin_edges_pred)
    hist_obs_norm = hist_obs * np.diff(bin_edges_obs)
    # Convex combinations smoothing
    # Based on:
    # https://mathoverflow.net/questions/72668/how-to-compute-kl-divergence-when-pmf-contains-0s
    alpha = 0.25
    hist_pred_norm_cc = alpha * (1 / 400) + (1 - alpha) * hist_pred_norm
    hist_obs_norm_cc = alpha * (1 / 400) + (1 - alpha) * hist_obs_norm
    kl_pq = kl_divergence(hist_pred_norm_cc, hist_obs_norm_cc)
    sys.stdout.write('KL(P || Q): %.3f bits\n' % kl_pq)
    # Calculate (Q || P)
    kl_qp = kl_divergence(hist_obs_norm_cc, hist_pred_norm_cc)
    sys.stdout.write('KL(Q || P): %.3f bits\n' % kl_qp)
    """
    Create plot and save model
    """
    # Create data range
    date_range, _, _ = create_date_range(ds_test)
    # Show plot
    plot_bayesian(preds_means, upper_bound, lower_bound, obs, date_range)
    # Save model if desired
    if SAVE_MODEL:
        torch.save(model.state_dict(), f'../{SAVE_PATH}/{i + 1}.pt')


if __name__ == '__main__':
    start = time.time()
    model = model_pipeline()
    end = time.time()
    print(f'Total runtime of the script: {end - start}s')
