"""
Ref: Singh and Joachims 2018
https://arxiv.org/abs/1802.07281
"""
import numpy as np
from .ValidInputs import *

def avg_exp(ranking_ids, group_ids):

    """
    Calculates the average exposure fairness metric for each demographic group of a ranking
    :param ranking_ids: numpy array of positive integers → ranking of items represented by corresponding ID numbers
    :param group_ids: numpy array of positive integers → demographic group for each corresponding item in ranking
    :return: avg_exp: numpy array of float values → average exposure for each group in the ranking
    """

    # Check to make sure given data is clean and usable
    check_AvgExp(ranking_ids, group_ids)

    # Calculate the exposure for each ranking
    exposure = 1 / np.log2(np.array(range(0, len(group_ids))) + 2)
    print("Exposure: ", exposure)
    # For each group, get the mean of the exposure value for its members
    return np.array([np.mean(exposure[group_ids == i]) for i in range(0, np.max(group_ids) + 1)])


def dp_exp(average_exposures: np.array) -> float:
    """
    This is a function to calculate the average exposure of subgroups represented by the group_ids array
    :param average_exposures: numpy array of float values → average exposures for each corresponding group
    :return: float value → disparate exposure of the ranking
    """

    # Check to make sure given data is clean and usable
    check_DpExp(average_exposures)

    # Return the value of the minimal exposure over the maximal exposure
    return np.min(average_exposures) / np.max(average_exposures)


def exposure_ratio(protected_exposure, unprotected_exposure):

    return protected_exposure/unprotected_exposure

