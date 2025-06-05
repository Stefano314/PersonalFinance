import datetime as dt
import polars as pl
import os

# Custom Libs
from Libs.global_vars import STOCKS_HISTORY_PATH
class Stock:

    def __init__(self, stock_name : str=None, stock_amount : float=0, stock_price : float=0, stock_prior_risk:float=0, stock_to_load : str = None):
        """
        """

        self.stock_name = stock_name
        self.stock_amount = stock_amount # Number of stocks bought
        self.stock_original_price = stock_price # Single stock price
        self.stock_current_price = stock_price # Current price is the same as the purchase one
        self.stock_prior_risk = stock_prior_risk

        self.stock_total_price = self.stock_amount*self.stock_original_price # Total amount of stocks bought
        self.stock_last_date = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stock_history = {'date' : [self.stock_last_date], 'stock name' : [self.stock_name], 'stock price' : [self.stock_original_price],
                            'stock amount' : [self.stock_amount], 'stock total price' : [self.stock_total_price], 'stock prior risk' : [self.stock_prior_risk]}
        
        if stock_to_load is not None:
            self.__load_stock(stock_to_load)
            self.stock_last_date = self.stock_history['date'][-1]
            self.stock_name = self.stock_history['stock name'][-1]
            self.stock_original_price = self.stock_history['stock price'][0]
            self.stock_current_price = self.stock_history['stock price'][-1]
            self.stock_amount = self.stock_history['stock amount'][-1]
            self.stock_prior_risk = self.stock_history['stock prior risk'][-1]
            self.stock_total_price = self.stock_history['stock total price'][-1]


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


    def update_stock(self, stock_name=None, stock_amount=None, stock_price=None, stock_prior_risk=None):

        self.__update_stock_name(stock_name)
        self.__update_stock_amount(stock_amount)
        self.__update_stock_price(stock_price)
        self.__update_stock_prior_risk(stock_prior_risk)
        self.__update_stock_date()

        self.__update_stock_total_price()
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
        return f"- Stock Name : {self.stock_name}\n- Stock Price : {self.stock_original_price}\n- Stock Amount : {self.stock_amount}\n- Stock Total Price : {self.stock_total_price}\n- Stock Prior Risk : {self.stock_prior_risk}\n- Stock Last Date Check : {self.stock_last_date}"