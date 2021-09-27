"""
Bayes-by-Backprop-model model adapted for CAMELS-US

@author: Jonas Fill
"""

import math
import torch
import torch.nn as nn

from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator


@variational_estimator
class BayesByBackpropModel(nn.Module):
    """
    Implements the many-to-one Bayesian LSTM
    """
    def __init__(self, hidden_size, num_layers=1, yearly_periodicity=False, io_behavior='many-to-one', discharge_as_input=False, prior_sigma_1=0.25, posterior_rho_init=-6, prior_pi=1):
        super(BayesByBackpropModel, self).__init__()
        self.num_layers = num_layers
        self.yearly_periodicity = yearly_periodicity
        self.io_behavior = io_behavior
        self.discharge_as_input = discharge_as_input
        self.input_size = 5
        self.prior_sigma_1 = prior_sigma_1
        self.posterior_rho_init = posterior_rho_init
        self.prior_pi = prior_pi
        # When we consider yearly periodicity, there are 2 additional input measures (sin, cos) to consider
        if self.yearly_periodicity:
            self.input_size += 1
        if self.discharge_as_input:
            self.input_size += 1
        # Number of days in the future the model predicts in the future. Depends on what's specified in the io_behavior
        if self.io_behavior.startswith('many-to-many'):
            self.days_future = int(self.io_behavior[self.io_behavior.rfind('-') + 1:])
        else:
            self.days_future = None
        self.lstm_1 = BayesianLSTM(self.input_size, hidden_size, prior_sigma_1=self.prior_sigma_1,
                                   posterior_rho_init=self.posterior_rho_init, prior_pi=self.prior_pi)
        self.lstm_2 = BayesianLSTM(hidden_size, hidden_size, prior_sigma_1=self.prior_sigma_1,
                                   posterior_rho_init=self.posterior_rho_init, prior_pi=self.prior_pi)
        # Define additional LSTM-layer in case we use a many-to-many-architecture and discharge as input
        # for the prediction phase we can't use discharge as input, so one input unit less!
        self.lstm_1_pred = BayesianLSTM(self.input_size - 1, hidden_size, prior_sigma_1=self.prior_sigma_1,
                                   posterior_rho_init=self.posterior_rho_init, prior_pi=self.prior_pi)
        if self.io_behavior == 'many-to-one':
            self.linear = nn.Linear(hidden_size, 1)
        elif self.io_behavior.startswith('single-shot'):
            self.num_shots = int(self.io_behavior[self.io_behavior.rfind('-') + 1:])
            self.linear = nn.Linear(hidden_size, self.num_shots)
        elif self.io_behavior.startswith('many-to-many'):
            self.linear = nn.Linear(hidden_size, 1)
        else:
            raise Exception('No valid I/O-behavior for the LSTM!')

    def forward(self, x):
        """
        Forward evaluations involves different steps depending on the architecture
        """
        if self.io_behavior.startswith('many-to-many') and not self.discharge_as_input:
            x_, _ = self.lstm_1(x)
            if self.num_layers == 2:
                x_, _ = self.lstm_2(x_)
            output = torch.zeros(x.shape[0], self.days_future)
            counter = -self.days_future
            while counter <= -1:
                x_p = x_[:, counter, :]
                x_p = self.linear(x_p)
                output[:, counter + self.days_future] = x_p.flatten()
                counter += 1
            return output
        elif self.io_behavior.startswith('many-to-many') and self.discharge_as_input:
            x_warmup = x[:, :x.shape[1] - self.days_future, :]
            # Discharge is not "seen" during the prediction phase
            x_pred = x[:, x.shape[1] - self.days_future:, [0, 1, 2, 3, 4, 6]]
            x_, (h_t_1, c_t_1) = self.lstm_1(x_warmup)
            if self.num_layers == 2:
                _, (h_t_2, c_t_2) = self.lstm_2(x_)
            x_, _ = self.lstm_1_pred(x_pred, (h_t_1, c_t_1))
            if self.num_layers == 2:
                x_, _ = self.lstm_2(x_, (h_t_2, c_t_2))
            x_ = self.linear(x_)
            x_ = x_.flatten(start_dim=1)
            return x_
        elif self.discharge_as_input:
            x_warmup = x[:, :x.shape[1] - 1, :]
            # Discharge is not "seen" during the prediction phase
            x_pred = x[:, x.shape[1] - 1:, [0, 1, 2, 3, 4, 6]]
            x_, (h_t_1, c_t_1) = self.lstm_1(x_warmup)
            if self.num_layers == 2:
                _, (h_t_2, c_t_2) = self.lstm_2(x_)
            x_, _ = self.lstm_1_pred(x_pred, (h_t_1, c_t_1))
            if self.num_layers == 2:
                x_, _ = self.lstm_2(x_, (h_t_2, c_t_2))
            x_ = self.linear(x_)
            x_ = x_.flatten(start_dim=1)
            return x_
        else:
            x_, _ = self.lstm_1(x)
            if self.num_layers == 2:
                x_, _ = self.lstm_2(x_)
            x_ = x_[:, -1, :]
            x_ = self.linear(x_)
            return x_