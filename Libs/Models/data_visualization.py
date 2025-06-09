import polars as pl
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

# Custom Libs
from Libs.Profile.stocks import Stock
from Libs.Models.data_manipulation import convert_date_to_numeric, convert_numeric_to_date

# Colors
cmap_custom = plt.colormaps['tab20']

# Fonts
# fontname = 'SourceCodePro-Light.ttf'
# fontname = 'times.ttf'
# fontname = 'CascadiaCode.ttf'

# for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
#     if fontname in font:
#         fontname=font

# custom_font = fm.FontProperties(fname=fontname)
# mpl.rcParams['font.family'] = custom_font.get_name()

# Font sizes
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['figure.titlesize'] = 16

# Custom Libs
from Libs.Management.bank_api import get_history_movements, get_history_balance
from Libs.Models.function_utils import nice_range

def get_uniformed_x(x_vals : list):
    """
    Returns the sorted list of all X values encountered (1D)
    """
    uniformed_x = set(x_vals[0])
    for x in x_vals[1:]:
        uniformed_x.update(set(x))
    return sorted(list(uniformed_x))


def plot_history_movements(movement_df=None, time_threshold=None, figsize=(12, 6)):
    """
    Plots the movements history.
    """

    if movement_df is None: # In case of regularized dataset
        movements_df = get_history_movements()

    mov_cols = movements_df.columns

    if time_threshold:
        movements_df = movements_df.filter(pl.col(movements_df.columns[0])>dt.datetime.strptime(time_threshold, '%Y-%m-%d'))

    plt.figure(figsize=figsize)
    x = movements_df[mov_cols[0]]
    y = movements_df[mov_cols[-1]]

    y_min, y_max = y.min(), y.max()
    plt.yticks(nice_range(y_min, y_max, 20))
    plt.plot(x, y, marker='o', linestyle='--', linewidth=1, color='black', label='Payment', alpha=0.7)

    # Fill areas above 0 (green)
    plt.fill_between(x, y, 0, where=(y > 0), interpolate=True, color='limegreen', alpha=0.3)

    # Fill areas below 0 (red)
    plt.fill_between(x, y, 0, where=(y < 0), interpolate=True, color='firebrick', alpha=0.3)

    plt.axhline(0, color='black', linewidth=0.5, linestyle='--') # Optional reference line

    plt.title(f"Movements History ({movements_df[mov_cols[0]].min().strftime('%Y/%m/%d')} - {movements_df[mov_cols[0]].max().strftime('%Y/%m/%d')})")
    plt.xlabel('Date')
    plt.ylabel('€')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(figsize)
    return fig


def plot_history_balance(balance_df=None, balance_col=None, time_threshold=None, figsize=(12, 6)):
    """
    Plots the movements history.
    """

    if balance_df is None: # In case of regularized dataset
        balance_df = get_history_balance()

    mov_cols = balance_df.columns

    if time_threshold:
        balance_df = balance_df.filter(pl.col(balance_df.columns[0])>dt.datetime.strptime(time_threshold, '%Y-%m-%d'))

    x = balance_df[mov_cols[0]]
    y = None

    if balance_col is None:
        y = balance_df[mov_cols[-1]]
    else:
        y = balance_df[balance_col]
    
    y_min, y_max = y.min(), y.max()
    y_min = min(y_min, 0)
    plt.yticks(nice_range(y_min, y_max, 30))
    plt.plot(x, y, marker='o', linestyle='--', linewidth=1, color='black', label='Balance', alpha=0.7)

    # Fill areas above 0 (green)
    plt.fill_between(x, y, 0, where=(y > 0), interpolate=True, color='limegreen', alpha=0.3)

    # Fill areas below 0 (red)
    plt.fill_between(x, y, 0, where=(y < 0), interpolate=True, color='firebrick', alpha=0.3)

    plt.axhline(0, color='black', linewidth=0.5, linestyle='--') # Optional reference line

    plt.title(f"Balance History ({balance_df[mov_cols[0]].min().strftime('%Y/%m/%d')} - {balance_df[mov_cols[0]].max().strftime('%Y/%m/%d')})")
    plt.xlabel('Date')
    plt.ylabel('€')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.legend()
    
    fig = plt.gcf()
    fig.set_size_inches(figsize)

    return fig


def plot_generic_polars(dataset : list, x_col : str, y_col : str, params={}):
    """
    Plot a generic polars dataframe.
    NOTE: if you want to plot over balance or movements, x_cols should probably be a datetime [mu].
    """

    x = dataset[x_col]
    y = dataset[y_col]

    marker='o' if params.get('marker') is None else params.get('marker')
    linestyle='--' if params.get('linestyle') is None else params.get('linestyle')
    linewidth=1 if params.get('linewidth') is None else params.get('linewidth')
    color='black' if params.get('color') is None else params.get('color')
    alpha=0.7 if params.get('alpha') is None else params.get('alpha')

    plt.plot(x, y, marker=marker, linestyle=linestyle, 
             linewidth=linewidth, color=color, label=params.get('label'), alpha=alpha)

    if 'title' in params:
        plt.title(params['title'])

    plt.xlabel(params.get('xlabel'))
    plt.ylabel(params.get('ylabel'))
    plt.xticks(rotation=45 if params.get('rotation') is None else params.get('rotation'))
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.legend()

    fig = plt.gcf()
    if 'figsize' in params:
        fig.set_size_inches(params.get('figsize'))

    return fig


def plot_generic_timeseries(dates : list, vals : list, params={}):
    """
    Plot a generic time series graph.
    """

    numerical_dates = convert_date_to_numeric(dates)[:,0]

    marker='o' if params.get('marker') is None else params.get('marker')
    linestyle='--' if params.get('linestyle') is None else params.get('linestyle')
    linewidth=1 if params.get('linewidth') is None else params.get('linewidth')
    color='black' if params.get('color') is None else params.get('color')
    alpha=0.7 if params.get('alpha') is None else params.get('alpha')
    plt.plot(numerical_dates, vals, marker=marker, linestyle=linestyle, 
             linewidth=linewidth, color=color, label=params.get('label'), alpha=alpha)

    if 'title' in params:
        plt.title(params['title'])

    plt.xlabel(params.get('xlabel'))
    plt.ylabel(params.get('ylabel'))
    plt.xticks(numerical_dates, dates, rotation=45 if params.get('rotation') is None else params.get('rotation'))
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.legend()

    fig = plt.gcf()
    if 'figsize' in params:
        fig.set_size_inches(params.get('figsize'))
    return fig