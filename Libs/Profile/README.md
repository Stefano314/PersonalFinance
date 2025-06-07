# **Stocks**
Stocks are objects that allow some functions, mainly to track the history of the purchased stock. These can also embed the concept of bonds.
A stock can be created from scratches or loaded using the constructor (```Stock(stock_to_load='test.csv')```).

Any stock can be saved (without creating duplicates) using ```save_history(self, filename : str = None)```.

The risks are given as a prior probability.

### Example
```python
from Libs.Profile.stocks import Stock
from Libs.Profile.stocks import get_stock_current_price

test_stock = Stock(stock_name='SP600', stock_amount=100, stock_price=get_stock_current_price('SP600'), stock_prior_risk=0.001)
print(test_stock)
test_stock.update_stock(stock_prior_risk=0.002)
print(test_stock)
test_stock.save_history()

test_stock = Stock(stock_to_load='SP600.csv')
```

# **Portfolio**
A portfolio is a package of stocks. We can add or remove stocks from it. We dont save the portfolio, since we simply have to load the single stocks in it.
(```portfolio = FinancialPortfolio(name='bigpack', stocks=[Stock('S&P500', 100, 50.2), Stock(stock_to_load='test.csv')])```)

We can get the single ammounts of all the stocks with the relative risks.

### Example
```python
from Libs.Profile.portfolio import FinancialPortfolio
from Libs.Profile.stocks import Stock
import os
from Libs.global_vars import STOCKS_HISTORY_PATH

portfolio = FinancialPortfolio('verybigpack')

# Load every file
for file_stock in [f for f in os.listdir(STOCKS_HISTORY_PATH) if f.endswith('.csv')]:
    portfolio.add_stock(Stock(stock_to_load=file_stock))
print(portfolio)
portfolio.get_ammounts_with_risks()
```