"""
Ref: Geyik et al. 2019
https://arxiv.org/pdf/1905.01989.pdf
"""

import numpy as np
from .ValidInputs import *


def NDKL(ranking_ids, group_ids):
    """
    Calculates the NDKL fairness metric for a ranking
    :param ranking_ids: numpy array of positive integers → ranking of items represented by corresponding ID numbers
    :param group_ids: numpy array of positive integers → demographic group for each corresponding item in ranking
    :return: float value → NDKL
    """

    # Check to make sure given data is clean and usable
    check_NDKL(ranking_ids, group_ids)

    # Stores the number of groups and the length of the list
    num_groups, list_length = np.max(group_ids), len(group_ids)

    # Stores the distributions of the groups based on the full list
    dr = distributions(group_ids, num_groups)

    # Define Z as an array of Z scores, which is the exposure function
    Z = Z_Vector(list_length)

    # Return the results fo 1 over the sum of Z scores times the kl_divergences of the sublists multiplied by their
    # respective Z score
    return (1 / np.sum(Z)) * np.sum(
        [Z[i] * kl_divergence(distributions(group_ids[0: i + 1], num_groups), dr) for i in range(0, list_length)])
