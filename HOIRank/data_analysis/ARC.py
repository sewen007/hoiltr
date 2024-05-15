import pandas as pd


def MARC(df1, df2):
    """
    Given 2 dataframes that contain similar doc_ids, this function will return the Maximum Rank Change value
     and Average Rank change for that list.
    :param df1: Dataframe 1
    :param df2: Dataframe 2
    :return: Maximum Rank Change value and Average Rank change
    """
    df1['index_old'] = df1.index
    df2['index_new'] = df2.index

    # Merge the dataframes to get the new positions of items
    merged_df = pd.merge(df1, df2, on='doc_id', suffixes=('_old', '_new'))

    # Calculate the difference in index positions
    merged_df['position_diff'] = merged_df['index_old'] - merged_df['index_new']

    # print(merged_df)
    ARC = (abs(merged_df['position_diff'])).mean()
    formatted_ARC = "{:.2f}".format(ARC)
    # Assign positive or negative sign based on the movement
    # merged_df['movement_sign'] = merged_df['position_diff'].apply(lambda x: '+' if x < 0 else '-')
    MRC = max(merged_df['position_diff'])

    return MRC, formatted_ARC, merged_df
