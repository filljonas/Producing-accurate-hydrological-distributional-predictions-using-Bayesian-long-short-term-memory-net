"""
Various utility functions for Bayesian neural networks

@author: Jonas Fill
"""

import math
from typing import Tuple
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from pathlib import Path
import sys

CAMELS_ROOT = Path('../CAMELS/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2')

catchment_list = ['01', '03', '11', '17']

"""
Title: Long Short-Term Memory (LSTM) for rainfall-runoff modelling
Author: Kratzert, Frederick
Date: 2020
Availability: https://github.com/kratzert/pangeo_lstm_example/blob/master/LSTM_for_rainfall_runoff_modelling.ipynb
"""
def load_forcing(basin: str) -> Tuple[pd.DataFrame, int]:
    """Load the meteorological forcing data of a specific basin.

    :param basin: 8-digit code of basin as string.

    :return: pd.DataFrame containing the meteorological forcing data and the
        area of the basin as integer.
    """
    # root directory of meteorological forcings
    forcing_path = CAMELS_ROOT / 'basin_mean_forcing' / 'daymet'

    # get path of forcing file
    files = list(forcing_path.glob("**/*_forcing_leap.txt"))
    files = [f for f in files if basin == f.name[:8]]
    if len(files) == 0:
        raise RuntimeError(f'No forcing file file found for Basin {basin}')
    else:
        file_path = files[0]

    # read-in data and convert date to datetime index
    with file_path.open('r') as fp:
        df = pd.read_csv(fp, sep='\s+', header=3)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/"
             + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # load area from header
    with file_path.open('r') as fp:
        content = fp.readlines()
        area = int(content[2])

    return df, area


"""
Title: Long Short-Term Memory (LSTM) for rainfall-runoff modelling
Author: Kratzert, Frederick
Date: 2020
Availability: https://github.com/kratzert/pangeo_lstm_example/blob/master/LSTM_for_rainfall_runoff_modelling.ipynb
"""
def load_discharge(basin: str, area: int) ->  pd.Series:
    """Load the discharge time series for a specific basin.

    :param basin: 8-digit code of basin as string.
    :param area: int, area of the catchment in square meters

    :return: A pd.Series containng the catchment normalized discharge.
    """
    # root directory of the streamflow data
    discharge_path = CAMELS_ROOT / 'usgs_streamflow'

    # get path of streamflow file file
    files = list(discharge_path.glob("**/*_streamflow_qc.txt"))
    files = [f for f in files if basin in f.name]
    if len(files) == 0:
        raise RuntimeError(f'No discharge file found for Basin {basin}')
    else:
        file_path = files[0]

    # read-in data and convert date to datetime index
    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    with file_path.open('r') as fp:
        df = pd.read_csv(fp, sep='\s+', header=None, names=col_names)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/"
             + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # normalize discharge from cubic feed per second to mm per day
    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10 ** 6)

    return df.QObs


""" Based on:
Title: Long Short-Term Memory (LSTM) for rainfall-runoff modelling
Author: Kratzert, Frederick
Date: 2020
Availability: https://github.com/kratzert/pangeo_lstm_example/blob/master/LSTM_for_rainfall_runoff_modelling.ipynb
"""
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int, io_behavior: str, discharge_as_input: bool, seq_single_shot=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction
    :param seq_single_shot: Output length in case single shot is enabled

    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape

    if io_behavior == 'many-to-one':
        x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
        y_new = np.zeros((num_samples - seq_length + 1, 1))
        for i in range(0, x_new.shape[0]):
            x_new[i, :, :num_features] = x[i:i + seq_length, :]
            y_new[i, :] = y[i + seq_length - 1, 0]
        return x_new, y_new
    # Support for single shot
    elif io_behavior.startswith('single-shot'):
        x_new = np.zeros(((num_samples - seq_length + 1) // seq_single_shot, seq_length, num_features))
        y_new = np.zeros(((num_samples - seq_length + 1) // seq_single_shot, seq_single_shot))
        # Current index in old arrays
        i = 0
        for index_new in range(x_new.shape[0]):
            x_new[index_new, :, :num_features] = x[i:i + seq_length, :]
            y_new[index_new, :] = y[i + seq_length - 1:i + seq_length + seq_single_shot - 1, 0]
            i += seq_single_shot
        return x_new, y_new
    elif io_behavior.startswith('many-to-many'):
        x_new = np.zeros(((num_samples - seq_length + 1) // seq_single_shot, seq_length, num_features))
        y_new = np.zeros(((num_samples - seq_length + 1) // seq_single_shot, seq_single_shot))
        # Current index in old arrays
        i = 0
        for index_new in range(x_new.shape[0]):
            x_new[index_new, :, :num_features] = x[i:i + seq_length, :]
            y_new[index_new, :] = y[i + seq_length - seq_single_shot:i + seq_length, 0]
            i += seq_single_shot
        return x_new, y_new
    else:
        raise Exception('No valid I/O-behavior for the LSTM!')


def get_relevant_catchments():
    """
    Get list of all relevant catchments to use
    :return:
    """
    files = []
    # root directory of the streamflow data
    discharge_path = CAMELS_ROOT / 'usgs_streamflow'
    for catchment in catchment_list:
        current_directory = discharge_path / catchment
        tmp_files = list(current_directory.glob("**/*_streamflow_qc.txt"))
        files.extend([f.name[:8] for f in tmp_files])
    return files


def plot_bayesian(preds_means, upper_bound, lower_bound, obs, date_range):
    """
    Plots hydrograph with predictions, prediction inverval and observations
    """
    fig = go.Figure([
        go.Scatter(
            name='Prediction interval',
            x=date_range,
            y=(upper_bound.flatten() + lower_bound.flatten()) / 2,
            mode='lines',
            marker=dict(color='rgba(68, 68, 68, 0.3)'),
            line=dict(width=1),
            showlegend=True
        ),
        go.Scatter(
            name='Prediction',
            x=date_range,
            y=preds_means.flatten(),
            mode='lines',
            line=dict(color='orange')
        ),
        go.Scatter(
            name='Upper bound',
            x=date_range,
            y=upper_bound.flatten(),
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower bound',
            x=date_range,
            y=lower_bound.flatten(),
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        ),
        go.Scatter(
            name='Observation',
            x=date_range,
            y=obs.flatten(),
            mode='lines',
            line=dict(color='blue'),
        ),
    ])
    fig.update_layout(
        yaxis_title='Discharge measure (mm/d)',
        xaxis_title='Date',
        hovermode='x',
        font_family='Arial',
        font_size=25
    )
    fig.show()


def kl_divergence(p, q):
    """
    # Calculate KL divergence between distributions p and q
    """
    return sum(p[i] * math.log2(p[i]/q[i]) for i in range(len(p)))


def get_discharge_series(x, y, discharge_index):
    """
    Given tensors of input data and labels, create some sequences containing input and ground-truth discharge
    :param x: input
    :param y: labels
    :param discharge_index: which input scalar is the discharge?
    :return:
    """
    limit = 20
    for i in range(limit):
        sys.stdout.write(f'Inputs: ')
        for j in range(len(x[i])):
            sys.stdout.write(f'{x[i][j][discharge_index]}   ')
        sys.stdout.write(f'Label: {y[i][0]}\n')


def create_date_range(ds_test):
    """
    Create date ranges for plots
    """
    start_date = ds_test.dates[0]
    end_date = ds_test.dates[1]
    date_range = pd.date_range(start_date, end_date)
    return date_range, start_date, end_date