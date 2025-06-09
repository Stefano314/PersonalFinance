# **Data Visualization**

#### Generic Plot Procedure with Fit
```python
from Libs.Models.data_visualization import plot_history_balance, plot_generic_polars
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
plot_history_balance(balance_df=df)
plot_generic_polars(dataset=df_ma, x_col='date', y_col='moving_average', params={'figsize':(10,5), 'label':'moving average', 'marker':'', 'linewidth':3, 'color':'darkblue'})
plot_generic_polars(dataset=df_ma_imputed, x_col='date', y_col='moving_average', params={'figsize':(10,5), 'label':'imputed data', 'marker':'', 'linewidth':1, 'color':'red'})
fig = plot_generic_polars(dataset=pred_df, x_col='date', y_col='model_prediction', params={'figsize':(10,5), 'label':'prediction', 'color':'darkred'})
```

#### Time Series Example
```python
from Libs.Models.data_visualization import get_uniformed_x, plot_generic_timeseries
import numpy as np

range1 = ['2025-09-01', '2025-10-02', '2025-11-3','2025-11-04','2025-11-05']
vals1 = np.random.rand(len(range1))*10+np.linspace(0,10, len(range1))

range2 = ['2025-10-03', '2025-10-11', '2025-11-02', '2025-11-04']
vals2 = np.random.rand(len(range2))+np.linspace(0,10, len(range2))

plot_generic_timeseries(range1, vals1, params={'label':'vals1', 'color':'black'})
plot_generic_timeseries(range2, vals2, params={'label':'vals2', 'color':'red'})

uniformed_x = get_uniformed_x([range1,range2])
fig = plot_generic_timeseries(uniformed_x, [None]*len(uniformed_x), params={'xlabel':'dates', 'ylabel':'EUR', 'title':'Plot Simulation'})
fig.set_size_inches((10, 6))
fig.show()
```