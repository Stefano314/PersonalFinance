import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

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
        print('[FUNINFO] Performing linear imputation ..')
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

def get_model_prediction(X : np.ndarray, model : dict, prediction_column=None):
    """
    X : X matrix to predict (if single, it should be a vector column, Ax=pred).
    model is like {'model' : linear_model, 'x_cols' : time_columns}
    """

    if prediction_column is None:
        prediction_column='model_prediction'

    prediction = model['model'].predict(X)
    A = pl.DataFrame()

    for ind, column_vector in enumerate(X.T):
        A = A.with_columns(pl.Series(model['x_cols'][ind], column_vector))

    A = A.with_columns(pl.Series(prediction_column, prediction.T[0]))
    return A
    

def goal_reach_prediction(model : dict, start_date : str, goal : float):
    """
    Given a model that has .predict() (sklearn mainly), get the date that reaches the specified goal.
    goal_reach_prediction(linear_model, '2025-07-01', 40_000)
    """
    from scipy.optimize import fsolve
    from Libs.Models.data_manipulation import convert_numeric_to_date

    def func(x):
        return model['model'].predict([x])[0,0] - goal

    x_solution, = fsolve(func, convert_date_to_numeric([start_date])[0,0])
    if abs(model['model'].predict([[x_solution]])[0,0]-goal) < 1e-6:
        return convert_numeric_to_date([int(x_solution)])
    else:
        return None


def fit_generic_polynomial_statsmodel(dataset : pl.DataFrame, x_col : str, y_col : str, deg=1):
    """
    """
    x = dataset.select(x_col).to_numpy()
    y = dataset[y_col].to_numpy()

    x_poly = np.ones((x.shape[0], deg + 1))
    for i in range(1, deg + 1):
        x_poly[:, i] = x[:, 0] ** i

    model = sm.OLS(y, x_poly).fit(method = "qr")

    dataset = dataset.with_columns(model_fit=model.predict(x_poly).T[0])

    return dataset, {'model' : model, 'x_col' : x_col}


def model_predict_statsmodel(X : np.ndarray, model : dict, deg=1, pred_col=None):
    """
    """
    if X.ndim == 1:
        X = X.reshape((-1,1))
    if pred_col is None:
        pred_col='model_prediction'

    X_poly = np.ones((X.shape[0], deg + 1))
    for i in range(1, deg + 1):
        X_poly[:, i] = X[:, 0] ** i
    
    prediction = model['model'].get_prediction(X_poly).summary_frame()
    A = pl.DataFrame()

    for ind, column_vector in enumerate(X.T):
        A = A.with_columns(pl.Series(model['x_col'], column_vector))
    A = A.with_columns(pl.Series(pred_col, prediction['mean'].values.T))
    return A, prediction