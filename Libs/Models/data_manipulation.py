import polars as pl
import numpy as np
import datetime as dt
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import polars as pl

def string_to_numeric_timestamp(date_str, fmt="%Y-%m-%d"):
    date = dt.datetime.strptime(date_str, fmt)
    return int(date.timestamp() * 1_000_000) # Polars wants us, maybe it can be changed


def numeric_timestamp_to_string(timestamp, fmt="%Y-%m-%d"):
    date = dt.datetime.fromtimestamp(timestamp / 1_000_000)
    return date.strftime(fmt)


def get_date_range(start_date : str, end_date : str, interval='1mo'):
    return [d.strftime("%Y-%m-%d") for d in pl.date_range(start=pl.lit(start_date), end=pl.lit(end_date), interval=interval, eager=True).to_list()]


def get_regularized_dataset(dataset : pl.DataFrame, reg_period='1d', x_cols=None, y_cols=None) -> pl.DataFrame:
    """
    Groups the dataset by regular date intervals. It summs the import content in the grouped reg_period.
    --> dates equally spaced, the imports are the sum in that periods
    """
    if x_cols is None or y_cols is None:
        dataset_cols = dataset.columns
    else:
        dataset_cols=[x_cols,y_cols]

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


def convert_date_to_numeric(dataset : pl.DataFrame, date_cols : list=None) -> pl.DataFrame:
    """
    Take a polars datetime [us] column and convert it to an integer in the form YYYY-MM-DD.
    Can also take a list of dates
    """

    # For convenience, if we pass the list of string dates we get back the conversion    
    if isinstance(dataset, list):
        return np.array([string_to_numeric_timestamp(dat) for dat in dataset]).reshape(-1,1)

    for col in date_cols:
        # dataset = dataset.with_columns(pl.col(col).cast(pl.Utf8).str.slice(0,10).str.replace_all('-', '').cast(pl.Int32))
        dataset = dataset.with_columns(pl.col(col).cast(pl.Int64))

    return dataset


def convert_numeric_to_date(dataset : pl.DataFrame, date_cols : list=None) -> pl.DataFrame:
    """
    Take a polars integer column in the form YYYYMMDD and convert it to a datetime [us] column.
    """

    # For convenience, if we pass the list of numbers we get back the conversion    
    if isinstance(dataset, list):
        return [numeric_timestamp_to_string(number) for number in dataset]

    for col in date_cols:
        # dataset = dataset.with_columns(
        #     pl.col(col).cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").cast(pl.Datetime("us"))
        # )
        dataset = dataset.with_columns(
            pl.col(col).cast(pl.Datetime("us"))
        )
    return dataset


# =============== Data Imputation ===============
def simple_imputation(df, numerical_cols : list, mode='mean', pre_trained = None):
    """
    Simple imputation to fill numerical null values
    """
    if pre_trained is not None: # Perform the imputation

        # Store imputation values
        imputation_values = {}

        if mode.lower() == 'mean':
            df_imputation = df.with_columns(
                pl.col(numerical_cols).fill_null(pl.col(numerical_cols).mean())
            )
            imputation_values = {col : df.select(pl.col(col)).mean().to_numpy()[0,0] for col in numerical_cols}

        elif mode.lower() == 'median':
            df_imputation = df.with_columns(
                pl.col(numerical_cols).fill_null(pl.col(numerical_cols).median())
            )
            imputation_values = {col : df.select(pl.col(col)).median().to_numpy()[0,0] for col in numerical_cols}

        return df_imputation, imputation_values

    else: # Use the given one
        for col, val in pre_trained.items():
            df_imputation = df.with_columns(pl.col(col).fill_null(val))

        return df_imputation, pre_trained


def simple_linear_imputation(dataset: pl.DataFrame, x_col: str, y_col: str) -> pl.DataFrame:
    """
    Impute missing values in 'y_col' using a linear regression on 'x_col'.
    """

    from sklearn.linear_model import LinearRegression

    # Filter rows where y_col is not null to train model
    train_dataset = dataset.filter(~pl.col(y_col).is_null())

    # Train linear regression model
    X_train = train_dataset.select(x_col).to_numpy() # Column vector format
    y_train = train_dataset[y_col].to_numpy() # Row vector format

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Impute missing y_col values
    X_all = dataset.select(x_col).to_numpy() # Column vector format
    y_imputed = np.where(
        dataset[y_col].is_null(),
        model.predict(X_all),
        dataset[y_col].to_numpy()
    )

    return dataset.with_columns([
        pl.Series(name=y_col, values=y_imputed)
    ]) 


