import numpy as np


def check_ranking_and_group(rankings_ids, group_ids):
    arrays = [rankings_ids, group_ids]

    check_all_nparrays(arrays)
    check_all_integers(arrays)

    check_increase_values(group_ids)
    check_all_distinct([rankings_ids])


def check_increase_values(group_ids):
    if set(group_ids) != set(range(0, np.max(group_ids) + 1)):
        return
    # raise Exception("group_ids does not conaint integers [0," + str(np.max(group_ids)) + "]")


def check_all_distinct(arrays):
    for i in arrays:
        if len(i) != len(np.unique(i)):
            raise Exception("Array does not content all unique values")


def check_all_nparrays(arrays):
    common_length = len(arrays[0])
    for i in arrays:
        if not isinstance(i, np.ndarray):
            raise Exception("Arrays must be the np.array data type")
        elif not np.issubdtype(i.dtype, np.number):
            raise Exception("Array is not numeric")
        elif len(i) != common_length:
            raise Exception("Array length does not match")


def check_all_integers(arrays):
    for i in arrays:
        if np.any(i % i.astype(int) != 0):
            raise Exception("Arrays must only contain of integers data type")


def check_position(pos, length):
    if pos <= 0 or pos > length:
        raise Exception("Pos is not within the bounds of the arrays")


def check_steps(steps, length):
    check_all_nparrays([steps])
    if np.max(steps) > length:
        raise Exception("Too many steps")
    for i in range(0, len(steps) - 1):
        if steps[i] > steps[i + 1]:
            raise Exception("Steps are not increasing")


# Check Functions for Each Metric

def check_Skew(ranking_ids, group_ids, group, pos):
    check_ranking_and_group(ranking_ids, group_ids)
    check_position(pos, len(ranking_ids))

    if group not in group_ids:
        return
        # raise Exception("Group does not appear in the group_ids array")


def check_NDKL(ranking_ids, group_ids):
    check_ranking_and_group(ranking_ids, group_ids)


def check_rKL(ranking_ids, group_ids, steps):
    check_ranking_and_group(ranking_ids, group_ids)
    check_steps(steps, len(group_ids))
    if np.any(np.unique(group_ids) != np.array([0, 1])):
        return
        # raise Exception("Groups are not binary")


def check_AvgExp(ranking_ids, group_ids):
    check_ranking_and_group(ranking_ids, group_ids)


def check_DpExp(average_exposures):
    check_all_nparrays([average_exposures])

    for i in average_exposures:
        if i < 0 or i > 1:
            raise Exception("Values in average_exposures are not all between 0 and 1")


def check_NDCG(ranking_ids, scores, pos):
    arrays = [ranking_ids, scores]

    check_all_nparrays(arrays)
    check_all_distinct([ranking_ids])

    check_position(pos, len(ranking_ids))


def check_Kendall_Tau(ranking_ids_1, ranking_ids_2):
    arrays = [ranking_ids_1, ranking_ids_2]

    check_all_nparrays(arrays)
    check_all_distinct(arrays)


# Shared Defined Functions

OFFSET = 1E-6


def prob(a, b):
    return ((b == a).sum()) / len(b)


def Z_Vector(k):
    return 1 / np.log2(np.array(range(0, k)) + 2)


def Z(k):
    return np.sum(Z_Vector(k))


def distributions(passed_ranked_list: np.array, num_groups: int) -> np.array:
    # Returns an array with each group id's probability based on the passed ranked_list
    return np.array([prob(i, passed_ranked_list) for i in range(0, num_groups + 1)])


def kl_divergence(p: np.array, q: np.array) -> float:
    # Gives p, q an offset in case either of them are 0
    p += OFFSET
    q += OFFSET
    # Returns the kl divergence of p and q
    return np.sum(p * np.log(p / q))
