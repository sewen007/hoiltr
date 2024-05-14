"""
Ref: Järvelin and Kekäläinen 2002
https://dl.acm.org/doi/10.1145/582415.582418
"""

import numpy as np
from .ValidInputs import *


def NDCG(ranking_ids: np.array, scores: np.array, pos: int) -> float:

    """
    Calculates the NDCG utility metric for a ranking
    :param ranking_ids: numpy array of positive integers → ranking of items represented by corresponding ID numbers
    :param scores: numpy array of float values → utility scores for each corresponding item in ranking
    :param pos: positive integer → 1-indexed position above which NDCG is calculated
    :return: float value → NDCG
    """

    # Check to make sure given data is clean and usable
    check_NDCG(ranking_ids, scores, pos)

    # Get the scores only to the given position for calculation
    scores = scores[0:pos]

    # Define Z as an array of Z scores, which is the exposure function
    Z = Z_Vector(len(scores))

    # Return the result of 1 over the sum of Z scores times the scores multiplied by their respective Z score
    return (1 / np.sum(Z)) * np.sum(np.multiply(scores, Z))
