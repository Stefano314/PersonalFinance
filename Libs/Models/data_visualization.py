import polars as pl
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

# Colors
cse_color = 'royalblue'
secondary_color = 'darkred'
cmap_custom = plt.colormaps['tab20']

# Fonts
# fontname = 'SourceCodePro-Light.ttf'
# fontname = 'times.ttf'
fontname = 'CascadiaCode.ttf'

for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    if fontname in font:
        fontname=font

custom_font = fm.FontProperties(fname=fontname)
mpl.rcParams['font.family'] = custom_font.get_name()

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

def plot_history_movements(time_threshold=None):
    """
    Plots the movements history.
    """

    movements_df = get_history_movements()
    mov_cols = movements_df.columns

    if time_threshold:
        movements_df = movements_df.filter(pl.col(movements_df.columns[0])>dt.datetime.strptime(time_threshold, '%Y-%m-%d'))

    plt.figure(figsize=(12, 6))
    x = movements_df[mov_cols[0]]
    y = movements_df[mov_cols[-1]]

    y_min, y_max = y.min(), y.max()
    plt.yticks(nice_range(y_min, y_max, 20))  # Increase 'num' for more ticks
    plt.plot(x, y, marker='o', linestyle='--', linewidth=1, color='black', label='payment', alpha=0.7)

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
    plt.show()


def plot_history_balance(time_threshold=None):
    """
    Plots the movements history.
    """

    balance_df = get_history_balance()
    mov_cols = balance_df.columns

    if time_threshold:
        balance_df = balance_df.filter(pl.col(balance_df.columns[0])>dt.datetime.strptime(time_threshold, '%Y-%m-%d'))

    plt.figure(figsize=(12, 6))
    x = balance_df[mov_cols[0]]
    y = balance_df[mov_cols[-1]]

    y_min, y_max = y.min(), y.max()
    y_min = min(y_min, 0)
    plt.yticks(nice_range(y_min, y_max, 30))  # Increase 'num' for more ticks
    plt.plot(x, y, marker='o', linestyle='--', linewidth=1, color='black', label='payment', alpha=0.7)

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
    plt.show()
    