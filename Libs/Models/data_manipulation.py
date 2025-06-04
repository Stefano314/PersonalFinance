import polars as pl
import numpy as np
import datetime as dt


def get_regularized_dataset(dataset : pl.DataFrame, reg_period='1d') -> pl.DataFrame:
    """
    Groups the dataset by regular date intervals. It creates a dataset of imports.
    """
    dataset_cols = dataset.columns
    regularized_dataset = dataset.group_by_dynamic(dataset_cols[0], every=reg_period).agg([pl.col(dataset_cols[-1]).sum().alias("total_value")])
    return regularized_dataset


def get_moving_average(dataset : pl.DataFrame, window_size=2, regularize=False) -> pl.DataFrame:
    """
    Moving average evaluation, producing a column called 'moving_average'.
    """
    if regularize:
        print("[INFO] Regularizing dataset (1d) before calculating moving average.")
        dataset = get_regularized_dataset(dataset)

    dataset = dataset.with_columns(
        moving_average=pl.col(dataset.columns[-1]).rolling_mean(center=True, window_size=window_size),
    )
    return dataset