# **Algorithms**
List of algorithms and models for forecasting

### Example
```python
from Libs.Models.data_visualization import plot_history_movements, plot_history_balance, overlay_generic_graph
from Libs.Management.bank_api import get_history_balance
from Libs.Models.data_manipulation import get_regularized_dataset, get_moving_average

# Get Balance, regularize it and take the moving average
balance_df = get_history_balance()
df = get_regularized_dataset(balance_df, reg_period='1d')
df_ma = get_moving_average(df, window_size=15)

# Perform date to int conversion so that we can perform data imputation on the missing vals from moving average
from Libs.Models.data_manipulation import simple_linear_imputation, convert_date_to_numeric, convert_numeric_to_date
df_ma_imputed = convert_date_to_numeric(df_ma, ['date'])
df_ma_imputed = simple_linear_imputation(df_ma_imputed, 'date', 'moving_average')

# Perform a linear fit on the imputed dataset and go back to date format
from Libs.Models.Predictions.algorithms import linear_fit
from Libs.Models.Predictions.algorithms import get_model_prediction

df_ma_imputed, linear_model = linear_fit(dataset=df_ma_imputed, cols_to_fit=None)
df_ma_imputed = convert_numeric_to_date(df_ma_imputed, ['date'])

# Define some new dates for forecasting and perform the prediction with the model created
X = convert_date_to_numeric(['2025-06-01', '2025-07-01', '2025-08-01'])
pred_df = get_model_prediction(X, linear_model)
pred_df = convert_numeric_to_date(pred_df, date_cols=['date'])

# Show the results
plot_history_balance(balance_df=df, additional_plots=[{'plot_function' : overlay_generic_graph,
                                                       'plot_parameters':{'dataset':df_ma, 'col_to_plot':'moving_average',
                                                                          'color':'darkred', 'linestyle':'--','label':'Moving Average'}}])

plot_history_balance(balance_df=df_ma, balance_col='total_value', additional_plots=[
    {'plot_function':overlay_generic_graph, 'plot_parameters':{'dataset':df_ma_imputed, 'col_to_plot':'moving_average', 'label':'moving average (linear imputation)'}},
    {'plot_function':overlay_generic_graph, 'plot_parameters':{'dataset':pred_df, 'col_to_plot':'model_prediction', 'label':'linear prediction', 'marker':'o', 'linestyle':'--'}}
    ]
)
```