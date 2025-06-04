import polars as pl

BALANCE_HISTORY_FILE = 'data/history/balance.csv'
MOVEMENTS_HISTORY_FILE = 'data/history/movements.csv'

movements_file_schema = {'data' : pl.Datetime, 'operazione' : pl.Utf8, 'dettagli' : pl.Utf8,
                         'conto o carta' : pl.Utf8, 'contabilizzazione' : pl.Utf8,
                         'categoria' : pl.Utf8, 'valuta' : pl.Utf8, 'importo' : pl.Float64}

balance_file_schema = {'date' : pl.Datetime, 'balance' : pl.Float64}