import pandas as pd


# Log out the entire DataFrame without truncation
def print_dataframe(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.to_string(index=False))
