"""
Ref: Geyik et al. 2019
https://arxiv.org/pdf/1905.01989.pdf
"""

from .ValidInputs import *


def skew(ranking_ids, group_ids, group, pos):

    """
    Calculates the skew fairness metric for a group in a position of a ranking
    :param ranking_ids: numpy array of positive integers → ranking of items represented by corresponding ID numbers
    :param pos: positive integer → 1-indexed position at which skew is calculated
    :param group: positive integer → demographic group for which skew is calculated
    :param group_ids: numpy array of positive integers → demographic group for each corresponding item in ranking
    :return: float value → skew

    """

    # check that the arguments passed in are valid
    check_Skew(ranking_ids, group_ids, group, pos)

    # get the first k items
    first_k = group_ids[0:pos]

    # return the skew value
    return prob(group, first_k) / prob(group, group_ids)


if __name__ == "__main__":
    # Sample Testing Code
    listA = np.array([1, 2, 3, 4])
    ID_A = np.array([1, 1, 0, 1])
    group_ID = 1
    k = 3

    print('skew = ', skew(listA, ID_A, group_ID, 3))
