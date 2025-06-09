import datetime as dt
import polars as pl
import numpy as np
import os

# Graph
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
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


import requests
from bs4 import BeautifulSoup

# Custom Libs
from Libs.global_vars import STOCKS_HISTORY_PATH

class Stock:

    def __init__(self, stock_name : str=None, stock_amount : float=0, stock_price : float=0,
                 stock_prior_risk:float=0, stock_capital_gain_tax=0.26, last_stock_exchange_commission=5, stock_to_load : str = None):
        """
        """

        self.stock_name = stock_name
        self.stock_amount = stock_amount # Number of stocks bought
        self.stock_original_price = stock_price # Single stock price
        self.stock_current_price = stock_price # Current price is the same as the purchase one
        self.stock_prior_risk = stock_prior_risk # Bayesian prior. Depending on what you define as risk.

        # self.stock_mantainment_fee = 0.002 # Statal fee, not considered
        self.stock_capital_gain_tax = stock_capital_gain_tax
        self.last_stock_exchange_commission = last_stock_exchange_commission # If 0 simply it wasnt an exchange, just a convention

        self.stock_total_price = self.stock_amount*self.stock_original_price # Total amount of stocks bought

        # Total net value one could get from selling the stock
        self.stock_net_value = self.stock_total_price*(1-self.stock_capital_gain_tax) - self.last_stock_exchange_commission

        self.stock_last_date = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stock_history = {'date' : [self.stock_last_date], 'stock name' : [self.stock_name], 'stock price' : [self.stock_original_price],
                            'stock amount' : [self.stock_amount], 'stock total price' : [self.stock_total_price], 'stock prior risk' : [self.stock_prior_risk],
                            'stock capital gain tax' : [self.stock_capital_gain_tax], 'stock exchange commission' : [self.last_stock_exchange_commission],
                            'stock net value' : [self.stock_net_value]}
        
        if stock_to_load is not None:
            self.__load_stock(stock_to_load)
            self.stock_last_date = self.stock_history['date'][-1]
            self.stock_name = self.stock_history['stock name'][-1]
            self.stock_original_price = self.stock_history['stock price'][0]
            self.stock_current_price = self.stock_history['stock price'][-1]
            self.stock_amount = self.stock_history['stock amount'][-1]
            self.stock_prior_risk = self.stock_history['stock prior risk'][-1]
            self.stock_total_price = self.stock_history['stock total price'][-1]
            self.stock_capital_gain_tax = self.stock_history['stock capital gain tax'][-1]
            self.last_stock_exchange_commission = self.stock_history['stock exchange commission'][-1]
            self.stock_net_value = self.stock_history['stock net value'][-1]



    def __load_stock(self, stock_to_load : str):
        self.stock_history = pl.read_csv(STOCKS_HISTORY_PATH+stock_to_load, separator=';').to_dict(as_series=False)


    def __update_stock_name(self, new_name : str):
        """
        Update stock name.
        """
        self.stock_name = new_name if new_name is not None else self.stock_name


    def __update_stock_price(self, new_price : float):
        """
        When stock changes priece.
        """
        self.stock_current_price = new_price if new_price is not None else self.stock_current_price


    def __update_stock_amount(self, new_amount : float):
        """
        When purchasing/selling stocks.
        """
        self.stock_amount = new_amount if new_amount is not None else  self.stock_amount
    
    
    def __update_stock_date(self):
        """
        Update last stock check date.
        """
        self.stock_last_date = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    def __update_stock_capital_gain_tax(self, capital_gain_tax : float):
        """
        """
        self.stock_capital_gain_tax = capital_gain_tax if capital_gain_tax is not None else self.stock_capital_gain_tax


    def __update_stock_net_value(self):
        """
        Update the net value of the stock when sell
        """
        # if last_stock_exchange_commission is 0 it means that no exchanges have been done
        self.stock_net_value = self.stock_total_price*(1-self.stock_capital_gain_tax) - self.last_stock_exchange_commission


    def __update_stock_history(self):
        """
        Add the current photo of the stock to history
        """

        self.stock_history['date'].append(self.stock_last_date)
        self.stock_history['stock name'].append(self.stock_name)
        self.stock_history['stock price'].append(self.stock_current_price)
        self.stock_history['stock amount'].append(self.stock_amount)
        self.stock_history['stock prior risk'].append(self.stock_prior_risk)
        self.stock_history['stock total price'].append(self.stock_total_price)

        self.stock_history['stock capital gain tax'].append(self.stock_capital_gain_tax)
        self.stock_history['stock exchange commission'].append(self.last_stock_exchange_commission)
        self.stock_history['stock net value'].append(self.stock_net_value)


    def __update_stock_prior_risk(self, new_prior_risk : float):
        """
        Evaluate the total price of the stocks (n_stocks*stock_price).
        """
        self.stock_prior_risk=new_prior_risk if new_prior_risk is not None else self.stock_prior_risk


    def __update_stock_total_price(self):
        """
        Evaluate the total price of the stocks (n_stocks*stock_price).
        """
        self.stock_total_price=self.stock_amount*self.stock_current_price


    def update_stock(self, stock_name=None, stock_amount=None, stock_price=None,
                     stock_prior_risk=None, last_stock_exchange_commission=0,
                     capital_gain_tax=None):

        self.__update_stock_name(stock_name)
        self.__update_stock_amount(stock_amount)
        self.__update_stock_price(stock_price)
        self.__update_stock_prior_risk(stock_prior_risk)

        self.__update_stock_total_price()

        self.__update_stock_capital_gain_tax(capital_gain_tax)
        self.last_stock_exchange_commission=last_stock_exchange_commission
        self.__update_stock_net_value()

        self.__update_stock_date()
        self.__update_stock_history()


    def save_history(self, filename : str = None):
        """
        Save history as a CSV.
        """
        filename = self.stock_name+'_history.csv' if filename is None else filename

        current_history_df = pl.DataFrame(self.stock_history)

        if os.path.exists(STOCKS_HISTORY_PATH+filename):
            old_history_df = pl.read_csv(STOCKS_HISTORY_PATH+filename, separator=';')
            current_history_df = pl.concat([old_history_df, current_history_df])
        
        current_history_df.unique().sort('date').write_csv(STOCKS_HISTORY_PATH+filename, separator=';')


    def __str__(self):
        response=''
        for k,v in self.stock_history.items():
            response+=f'- {k} (last 5) : {v[-5:]}\n'

        return response


def get_stock_current_price(index : str, cert : str=None):
    """
    Get price by parsing the result from Google Finance.
    """

    response = requests.get(
        f'https://www.google.com/finance/quote/{index}:INDEXSP?hl=it&window=1Y',
        verify=cert,
        timeout=10
    )

    soup = BeautifulSoup(response.text, 'html.parser')
    return float(soup.find('div', class_='YMlKec fxKbKc').text.replace('.','').replace(',','.'))


def project_stock_price(stock : Stock, expected_year_gain : float, end_date : str,
                        start_date=None, dividends_months=None, return_datetime = False,
                        noise=False):
    """
    We need an expected year percentage gain. The end_date must be in the form 'YYYY-MM-DD'.
    dividends_months is None for no accumulation, otherwise is how many months pass until the cap is increased (ex. dividends_months=4)
    Returns the x and y that made the graph.

    a2 = a1(1+r_m*t)
    a3 = a2(1+r_m*t)
    --> aN = a1(1+r_m*t)^N
    N : Number of times we had dividends (payments)
    r_m : monthly interest --> expected_r_in_n_months/n_months_considered !we assume to be a constant gain, so you take and you get r% always!
    t : n_months required for dividend (payment)

    ex.
    test = portfolio.get_stock('S&P500')
    test.update_stock(stock_prior_risk=0.02) # Increase risk to appreciate

    # Without risk, they are more or less the same in terms of gain, but with noise things can change
    _,a = project_stock_price(stock=test, expected_year_gain=1, return_datetime=True, noise=True,
                            start_date='2025-01-01', end_date='2026-01-01', dividends_months=None)
    r,b = project_stock_price(stock=test, expected_year_gain=0.5, return_datetime=True, noise=True,
                            start_date='2025-01-01', end_date='2026-01-01', dividends_months=2)
    plot(r,a) plot(r,b)
    """
    from Libs.Models.data_manipulation import convert_date_to_numeric, convert_numeric_to_date, get_date_range

    if start_date is None:
        start_date = dt.datetime.now().strftime("%Y-%m-%d")
    
    date_range = get_date_range(start_date=start_date, end_date=end_date)
    single_step_percentage = expected_year_gain/(len(date_range)-1) # Current date doesnt matter
    # numeric_date_range = convert_date_to_numeric(date_range)
    current_stock_val = stock.stock_current_price


    if dividends_months is None:
        if noise: # If noise, make random fluctuations according to the stock risk
            current_stock_val = np.random.normal(current_stock_val, current_stock_val*stock.stock_prior_risk, len(date_range)) # noisy value
            print(current_stock_val)
        else:
            current_stock_val = [current_stock_val]*len(date_range) # constant value
        earnings_vector = [np.round(current_stock_val[i]*(1+single_step_percentage*i),2) for i in range(len(date_range))]

        if return_datetime:
            date_range = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in date_range]

        return date_range, earnings_vector
    else:
        dividends_eval_indexes = get_date_range(start_date=start_date, end_date=end_date, interval=f"{dividends_months}mo")
        dividends_eval_indexes = [i for i, date in enumerate(date_range) if date in dividends_eval_indexes if i!=0] # Get the position indexes

        earnings_vector = [current_stock_val]
        noise_val = 0 # no noise at time 0
        for i in range(len(date_range)-1):
            
            if noise and i!=0: # If noise, make random fluctuations according to the stock risk
                noise_val = earnings_vector[-1]-np.random.normal(earnings_vector[-1], earnings_vector[-1]*stock.stock_prior_risk)
            prev_value = earnings_vector[-1] + noise_val # get last capital value

            if i in dividends_eval_indexes:
                # Add dividend before applying interest
                prev_value += prev_value * single_step_percentage

            # Apply compound interest for this period
            new_value = np.round(prev_value * (1 + single_step_percentage), 2)
            earnings_vector.append(new_value)

        if return_datetime:
            date_range = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in date_range]

        return date_range, earnings_vector


def plot_stock_value(stock : Stock, time_range=None, figsize=(12, 6)):
    """
    Plots Stock value over time.
    Returns the x and y that made the graph (x=numerical dates)
    """
    from Libs.Models.data_manipulation import string_to_numeric_timestamp

    stock_values = np.array(stock.stock_history['stock price'])
    stock_dates = np.array(stock.stock_history['date'])
    numerical_stock_dates = [string_to_numeric_timestamp(d, fmt='%Y-%m-%d %H:%M:%S') for d in stock_dates]

    val_derivative = np.insert(np.diff(stock_values),0,0)
    val_derivative = val_derivative/val_derivative.max()
    colors = np.where(val_derivative < 0, 'red', 'green')

    if time_range:
        indexes_to_keep = [ind for ind,d in stock_dates if d>=time_range[0] and d<=time_range[1]]
        stock_values = stock_values[indexes_to_keep]
        stock_dates = stock_dates[indexes_to_keep]

    plt.figure(figsize=figsize)
    ax = plt.gca()
    plt.plot(numerical_stock_dates, stock_values, marker='o', linestyle='--', linewidth=1, color='black', label='Stock Value', alpha=0.7)

    # Derivatives up or down
    for stock_d, stock_v, deriv, color in zip(numerical_stock_dates, stock_values, val_derivative, colors):
        ax.quiver(
                stock_d, stock_v, 0, deriv, headaxislength=3.5, alpha=0.4,
                color=color, width=0.003, scale=1, angles='xy',
                scale_units='xy', headwidth=3, headlength=4
            )

    plt.axhline(np.mean(stock_values), color='darkred', linewidth=0.5, linestyle='--', label='mean') # Mean value reference
    plt.xticks(numerical_stock_dates, stock_dates)
    # plt.ylim([stock_values.min()*0.97, stock_values.max()*1.03])
    plt.title(f"Stock Value ({dt.datetime.strptime(stock_dates[0], '%Y-%m-%d %H:%M:%S').strftime('%Y/%m/%d')} - {dt.datetime.strptime(stock_dates[-1], '%Y-%m-%d %H:%M:%S').strftime('%Y/%m/%d')})")
    plt.xlabel('Date')
    plt.ylabel('€')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.5)
    plt.tight_layout()

    plt.legend()        
    plt.show()
    return numerical_stock_dates, stock_values