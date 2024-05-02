import pandas as pd


# Log out the entire DataFrame without truncation
def print_dataframe_fully(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.to_string(index=False))


def check_dataframe_equality(df1, df2):
    """
    Check if two Pandas DataFrames are equal.

    Parameters:
    df1 (DataFrame): The first DataFrame to be compared.
    df2 (DataFrame): The second DataFrame to be compared.

    Returns:
    bool: True if the DataFrames are equal, False otherwise.
    """
    if not df1.index.equals(df2.index):
        print("Indices are different")
        print("First DataFrame indices:", df1.index)
        print("Second DataFrame indices:", df2.index)
        return False

    if not df1.columns.equals(df2.columns):
        print("Columns are different")
        print("First DataFrame columns:", df1.columns)
        print("Second DataFrame columns:", df2.columns)
        return False

    # Check if data values are different
    if not df1.equals(df2):
        print("Data values are different")

        # Check if data types are different for each column
        for col in df1.columns:
            if df1[col].dtype != df2[col].dtype:
                print(f"Data type of column '{col}' is different")
                print(f"Data type in df1: {df1[col].dtype}")
                print(f"Data type in df2: {df2[col].dtype}")

        return False

    print("DataFrames are equal")


def reset_dataframe_index(dataframe):
    """
    Reset the index of a Pandas DataFrame.

    Parameters:
    dataframe (DataFrame): The DataFrame whose index will be reset.

    Returns:
    DataFrame: The DataFrame with the index reset.
    """
    return dataframe.reset_index(drop=True)


def get_num_unique_ids(dataframe):
    num_unique_ids = dataframe['ID'].nunique()
    return num_unique_ids


def check_dataframe(dataframe):
    if dataframe is None:
        print("DataFrame is None.")
        return False

    if dataframe.empty:
        print("DataFrame is empty.")
        return False

    if len(dataframe) == 0:
        print("DataFrame has no rows.")
        return False

    return True
