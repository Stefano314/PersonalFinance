import polars as pl
from sklearn.linear_model import LinearRegression

# Custom Libs
from Libs.global_vars import balance_file_schema, movements_file_schema
from Libs.Models.data_manipulation import convert_date_to_numeric, simple_linear_imputation


def linear_fit(dataset : pl.DataFrame, cols_to_fit = None):
    """
    cols_to_fit = {'time_col' : [time_col], 'numerical_cols' : [c1, c2, ...]}
    """

    # If no columns are given, the convention is to take the first column as time, and the last as numerical
    numerical_cols = None
    time_columns = None
    if cols_to_fit is None:
        numerical_cols = [dataset.columns[-1]]
        time_columns = [dataset.columns[0]]
    else:
        numerical_cols = cols_to_fit['time_col']
        time_columns = cols_to_fit['numerical_cols']

    # Convert date to numeric
    dataset = convert_date_to_numeric(dataset, time_columns)

    # Linear imputation for each variable if needed
    if dataset.select(numerical_cols).null_count().item()>0:
        for time_col in time_columns:
            for num_col in numerical_cols:
                dataset = simple_linear_imputation(dataset, x_col=time_col, y_col=num_col)

    time_vals = dataset.select(time_columns).to_numpy() # Column vector
    vals_to_fit = dataset[numerical_cols].to_numpy() # Row vectors

    # Linear fit
    linear_model = LinearRegression().fit(time_vals, vals_to_fit)

    # Add fit to dataset
    dataset = dataset.with_columns(linear_fit=linear_model.predict(time_vals).T[0])

    return dataset, {'model' : linear_model, 'x_cols' : time_columns}

    

