from Libs.Profile.stocks import Stock

class FinancialPortfolio:
    """
    Represents a financial subject with a name and an optional description.
    """

    def __init__(self, name: str, stocks : list=None):
        self.name = name
        self.stocks = {} if stocks is None else {stock.stock_name : stock for stock in stocks}


    def add_stock(self, stock : Stock):
        """
        """
        if stock.stock_name in self.stocks:
            raise ValueError(f"[ERROR] Stock with name '{stock.stock_name}' already exists.")
        self.stocks[stock.stock_name] = stock


    def remove_stock(self, stock_name: str):
        """
        Removes a stock from the financial subject by its name.
        """
        if not stock_name in self.stocks:
            raise ValueError(f"[ERROR] Can't find any '{stock_name}' in stocks ({list(self.stocks.keys())}).")
        self.stocks.pop(stock_name)

    
    def get_ammounts_with_risks(self):
        """
        {name : [ammount, risk], ... }.
        """
        return {name : [stock.stock_total_price, stock.stock_prior_risk] for name, stock in self.stocks.items()}


    def get_stock(self, stock_name):
        """
        Get Stock object from dict.
        """
        return self.stocks[stock_name]


    def __str__(self):
        print_text = f"# FinancialPortfolio: '{self.name}'\n\n"
        for stock_name, stock in self.stocks.items():
            print_text+=f"=== {stock_name} ===\n"
            print_text+=stock.__str__()
            print_text+='\n\n'
        print_text += "====================="

        return print_text