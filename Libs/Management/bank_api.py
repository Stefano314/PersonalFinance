import os
import polars as pl
import pandas as pd # For excel reading
import datetime as dt

from Libs.global_vars import BALANCE_HISTORY_FILE, MOVEMENTS_HISTORY_FILE, balance_file_schema, movements_file_schema


def get_current_balance(balance_file_path : str = BALANCE_HISTORY_FILE) -> float:
    """
    Returns the last recorded balance from the balance history file
    """
    
    history = pl.read_csv(balance_file_path, schema=balance_file_schema, separator=';').sort(by=list(balance_file_schema.keys())[0]) # lol
    return history[-1].select("balance").to_series().item()


def get_history_balance(balance_file_path : str = BALANCE_HISTORY_FILE) -> pl.DataFrame:
    """
    Returns the whole balance history
    """
    history = pl.read_csv(balance_file_path, schema=balance_file_schema, separator=';').sort(by=list(balance_file_schema.keys())[0]) # lol
    return history


def get_history_movements(movements_file_path : str = MOVEMENTS_HISTORY_FILE) -> pl.DataFrame:
    """
    Returns the whole movements history
    """
    history = pl.read_csv(movements_file_path, schema=movements_file_schema, separator=';').sort(by=list(movements_file_schema.keys())[0]) # lol
    return history


def update_balance_file(new_balance : float, balance_file_path : str = BALANCE_HISTORY_FILE) -> None:
    """
    Updates the balance history file with a the specified new balance at the current datetime.
    """
    
    with open(balance_file_path, "a") as f:
        if os.stat(balance_file_path).st_size == 0:
            f.write("date;balance\n")
        f.write(f"""\n{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")};{new_balance}""")


def update_movements_file(new_orig_mov_file : str, movements_file_path : str = MOVEMENTS_HISTORY_FILE):
    """
    NOTE: This function is made for a very specific bank file format, using an excel. It's quite easy to adapt tho.
    
    Updates the movements history file with a new movements file. The new movements will be appended to the existing file,
    and it will not produce duplicates.
    """
    original_mov_file = pd.read_excel(new_orig_mov_file, header=13) # 13 since the bank file sucks and starts at row 14
    original_mov_file = original_mov_file[[c for c in original_mov_file.columns if 'unnamed' not in c.lower()]]
    original_mov_file.columns = [c.lower() for c in original_mov_file.columns]

    # Append to the history movements file the new content
    with open(movements_file_path, 'a') as f:
        if os.stat(movements_file_path).st_size == 0:
            f.write(';'.join(original_mov_file.columns) + '\n')
        for _, row in original_mov_file.iloc[1:].iterrows():
            f.write(';'.join(str(x) for x in row) + '\n')

    original_mov_file = pl.from_pandas(original_mov_file, schema_overrides=movements_file_schema)
    original_mov_file.unique().sort(by=original_mov_file.columns[0]).write_csv(movements_file_path, separator=';')


def reconstruct_balance(current_balance=None) -> pl.DataFrame:
    """
    Reconstruct the balance from movements and the current balance.
    ==> Last balance will be the current one, the rest will be created using the cumulative sum of the movements,
    starting by the difference as initial balance.

    NOTE: Use this only if you dont have a balance history, otherwise just use the `update_balance_from_movements` function.
    """

    movements_df = get_history_movements()
    try:
        current_balance = get_current_balance()
    except:
        if current_balance is None:
            raise ValueError("[ERROR] No current balance provided and unable to retrieve it from the balance history file.")
        else: pass

    mov_cols = movements_df.columns

    # Group the movements
    grouped_movements = movements_df.group_by(mov_cols[0]).agg(pl.sum(mov_cols[-1]).alias('daily_balance')).sort(mov_cols[0], descending=False)
    grouped_movements = grouped_movements.select(pl.col(mov_cols[0]), pl.cum_sum('daily_balance').alias('cum_sum'))

    # Reconstruct the balance
    final_balance_cum_sum = grouped_movements['cum_sum'].tail(1).to_numpy()[0]
    starting_balance = current_balance - final_balance_cum_sum

    reconstructed_balance = pl.DataFrame({'date': grouped_movements[movements_df.columns[0]],
                                          'balance' : (grouped_movements['cum_sum']+starting_balance).round(2)},
                                          schema=balance_file_schema)
    
    reconstructed_balance.write_csv(BALANCE_HISTORY_FILE, separator=';')


def update_balance_from_movements() -> pl.DataFrame:
    """
    Updates the balance history file considering the movements that are newer than the last balance entry.
    This function will calculate the cumulative sum of the movements and update the balance history file accordingly, grouping by date.
    """

    movements_df = get_history_movements() # Ordered by date
    balance_df = get_history_balance() # Ordered by date

    max_mov_date = movements_df[movements_df.columns[0]].max()
    max_bal_date = balance_df[balance_df.columns[0]].max()
    if max_mov_date <= max_bal_date:
        print("[INFO] Movements are already up to date with the balance history.")
        return 0

    # Get last balance value to be updated with the remaining movements cumulative
    last_balance = balance_df[balance_df.columns[-1]].tail(1).to_numpy()[0]
    remaining_movements = movements_df.filter(pl.col(movements_df.columns[0]) > max_bal_date)

    mov_cols = movements_df.columns
    grouped_movements = remaining_movements.group_by(mov_cols[0]).agg(pl.sum(mov_cols[-1]).alias('daily_balance')).sort(mov_cols[0], descending=False)
    grouped_movements = grouped_movements.select(pl.col(mov_cols[0]), pl.cum_sum('daily_balance').alias('cum_sum'))

    new_balance = pl.DataFrame({'date': grouped_movements[movements_df.columns[0]],
                                'balance' : (grouped_movements['cum_sum']+last_balance).round(2)},
                                schema=balance_file_schema)
    
    with open(BALANCE_HISTORY_FILE, mode="a", encoding='utf-8') as f:
        new_balance.write_csv(f, include_header=False, separator=';')
