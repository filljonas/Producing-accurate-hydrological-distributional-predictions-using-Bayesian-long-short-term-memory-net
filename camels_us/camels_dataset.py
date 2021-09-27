""" Based on:
Title: Long Short-Term Memory (LSTM) for rainfall-runoff modelling
Author: Kratzert, Frederick
Date: 2020
Availability: https://github.com/kratzert/pangeo_lstm_example/blob/master/LSTM_for_rainfall_runoff_modelling.ipynb
"""

import datetime

from torch.utils.data import Dataset
from util import *
from typing import List
import torch


class CamelsTXT(Dataset):
    """Torch Dataset for basic use of data from the CAMELS data set.

    This data set provides meteorological observations and discharge of a given
    basin from the CAMELS data set.
    """

    def __init__(self, basin: str, seq_length: int=365,period: str=None,
                 dates: List=None, means: pd.Series=None, stds: pd.Series=None,
                 yearly_periodicity=False, io_behavior='many-to-one', discharge_as_input=False):
        """Initialize Dataset containing the data of a single basin.

        :param basin: 8-digit code of basin as string.
        :param seq_length: (optional) Length of the time window of
            meteorological input provided for one time step of prediction.
        :param period: (optional) One of ['train', 'eval']. None loads the
            entire time series.
        :param dates: (optional) List of pd.DateTimes of the start and end date
            of the discharge period that is used.
        :param means: (optional) Means of input and output features derived from
            the training period. Has to be provided for 'eval' period. Can be
            retrieved if calling .get_means() on the data set.
        :param stds: (optional) Stds of input and output features derived from
            the training period. Has to be provided for 'eval' period. Can be
            retrieved if calling .get_stds() on the data set.
        :param seasonal_periodicity: (optional) Adds seasonal periodicity data
            in the form of sine and cosine
        """
        self.basin = basin
        self.seq_length = seq_length
        self.period = period
        self.dates = dates
        self.means = means
        self.stds = stds
        self.yearly_periodicity = yearly_periodicity
        self.io_behavior = io_behavior
        self.discharge_as_input = discharge_as_input

        # load data into memory
        self.x, self.y = self._load_data()

        # store number of samples as class attribute
        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def _load_data(self):
        """Load input and output data from text files."""
        df, area = load_forcing(self.basin)
        df['QObs(mm/d)'] = load_discharge(self.basin, area)

        if self.dates is not None:
            # If meteorological observations exist before start date use these as well. Similar to hydrological warmup period.
            if self.dates[0] - pd.DateOffset(days=self.seq_length) > df.index[0]:
                start_date = self.dates[0] - pd.DateOffset(days=self.seq_length)
            else:
                start_date = self.dates[0]
            df = df[start_date:self.dates[1]]

        # If training period store means and stds
        if self.period == 'train':
            self.means = df.mean()
            self.stds = df.std()

        # Extract input and output features from DataFrame
        x = np.array([df['prcp(mm/day)'].values,
                      df['srad(W/m2)'].values,
                      df['tmax(C)'].values,
                      df['tmin(C)'].values,
                      df['vp(Pa)'].values]).T
        y = np.array([df['QObs(mm/d)'].values]).T

        if self.discharge_as_input:
            x_p = np.array([df['QObs(mm/d)'].values]).T
            x = np.concatenate((x, x_p), axis=1)

        # Normalize data
        x = self._local_normalization(x, variable='inputs')

        # Add sine and cosine signals for yearly periodicity
        if self.yearly_periodicity:
            timestamp = df.index.map(datetime.datetime.timestamp)
            day = 24 * 60 * 60
            year = (365.2425) * day

            df['Year sin'] = np.sin(timestamp * (2 * np.pi / year))

            # Build array with periodicity information
            x_p = np.array([df['Year sin'].values]).T
            x = np.concatenate((x, x_p), axis=1)

        # Reshape data for LSTM training
        seq_single_shot = int(self.io_behavior[self.io_behavior.rfind('-') + 1:]) if self.io_behavior.startswith('single-shot') or self.io_behavior.startswith('many-to-many') else None
        x, y = reshape_data(x, y, self.seq_length, self.io_behavior, self.discharge_as_input, seq_single_shot=seq_single_shot)

        if self.period == "train":
            # Delete all samples, where discharge is NaN
            if np.sum(np.isnan(y)) > 0 or np.sum(np.isnan(x)) > 0:
                print(f"Deleted some records because of NaNs {self.basin}")
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(x)), axis=0)
                x = np.delete(x, np.argwhere(np.isnan(x)), axis=0)

            # Deletes all records, where no discharge was measured (-999)
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)

            # normalize discharge
            y = self._local_normalization(y, variable='output')

        # convert arrays to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        # get_discharge_series(x, y, 5)
        return x, y

    def _local_normalization(self, feature: np.ndarray, variable: str) -> \
            np.ndarray:
        """Normalize input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            means = np.array([self.means['prcp(mm/day)'],
                              self.means['srad(W/m2)'],
                              self.means['tmax(C)'],
                              self.means['tmin(C)'],
                              self.means['vp(Pa)']])
            if self.discharge_as_input:
                means_p = np.array([self.means['QObs(mm/d)']])
                means = np.concatenate((means, means_p), axis=0)
            stds = np.array([self.stds['prcp(mm/day)'],
                             self.stds['srad(W/m2)'],
                             self.stds['tmax(C)'],
                             self.stds['tmin(C)'],
                             self.stds['vp(Pa)']])
            if self.discharge_as_input:
                stds_p = np.array([self.stds['QObs(mm/d)']])
                stds = np.concatenate((stds, stds_p), axis=0)
            feature = (feature - means) / stds
        elif variable == 'output':
            feature = ((feature - self.means["QObs(mm/d)"]) /
                       self.stds["QObs(mm/d)"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def local_rescale(self, feature: np.ndarray, variable: str) -> \
            np.ndarray:
        """Rescale input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            means = np.array([self.means['prcp(mm/day)'],
                              self.means['srad(W/m2)'],
                              self.means['tmax(C)'],
                              self.means['tmin(C)'],
                              self.means['vp(Pa)']])
            if self.discharge_as_input:
                means_p = np.array([self.means['QObs(mm/d)']])
                means = np.concatenate((means, means_p), axis=0)
            stds = np.array([self.stds['prcp(mm/day)'],
                             self.stds['srad(W/m2)'],
                             self.stds['tmax(C)'],
                             self.stds['tmin(C)'],
                             self.stds['vp(Pa)']])
            if self.discharge_as_input:
                stds_p = np.array([self.stds['QObs(mm/d)']])
                stds = np.concatenate((stds, stds_p), axis=0)
            feature = feature * stds + means
        elif variable == 'output':
            feature = (feature * self.stds["QObs(mm/d)"] +
                       self.means["QObs(mm/d)"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds
