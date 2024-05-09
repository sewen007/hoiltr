import pandas as pd

def avg_pos_diff_skew(skew1, skew2, group):
    """
    Given the filepaths to different skews, this function will look at the skew of each position in both skews and return
    the average positional difference when comparing the two skews. A value of greater than one indicates that skew
    or the representation of that group increased and a values of less than one indicates that the skew or the representation
    of that group went down. The closer to 0 means the less of an impact that was made on the skews.
    :param skew1: Filepath to skew BEFORE ALTERATIONS, THIS IS THE CONTROL
    :param skew2: Filepath to skew AFTER ALTERATIONS, THIS IS THE BETTER/WORSE SKEW
    :param group: (str) group that you want the average positional difference for, "0" or "1"
    :return:
    """

    df1 = pd.read_csv(skew1)
    df2 = pd.read_csv(skew2)

    if len(df1.index) != len(df2.index):
        print("The two skews have different sizes of rows, exiting.")
        return

    skews_group = "Group " + group
    skews_before = df1[skews_group]
    skews_after = df2[skews_group]

    count = 0

    for i in range(0, len(df1.index)):
        count += float(skews_after[i]) - float(skews_before[i])

    return count/len(df1.index)


def avg_pos_diff_ndcg(r1, r2):
    """
    Given the filepaths to different rankings, this function will look at the ndcg of each position in both rankings and return
    the average positional difference when comparing the two ndcg's. A value of greater than one indicates that ndcg increased 
    or the utility of that ranking increased and a values of less than one indicates that the ndcg or the utility
    of that ranking went down. The closer to 0 means the less of an impact that was made on the skews.
    :param skew1: Filepath to skew BEFORE ALTERATIONS, THIS IS THE CONTROL
    :param skew2: Filepath to skew AFTER ALTERATIONS, THIS IS THE BETTER/WORSE SKEW
    :param group: (str) group that you want the average positional difference for, "0" or "1"
    :return:
    """

    df1 = pd.read_csv(r1)
    df2 = pd.read_csv(r2)

    if len(df1.index) != len(df2.index):
        print("The two skews have different sizes of rows, exiting.")
        return

    ndcg_before = df1["NDCG"]
    ndcg_after = df2["NDCG"]

    count = 0

    for i in range(0, len(df1.index)):
        count += float(ndcg_after[i]) - float(ndcg_before[i])

    return count/len(df1.index)


