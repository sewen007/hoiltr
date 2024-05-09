from scipy.stats import kendalltau


def kT(X, Y):
    """
    calculate kendall tau correlation coefficient between two rankings
    :param X: rank 1
    :param Y: rank 2
    :return: kendall tau correlation coefficient
    """
    corr, p_value = kendalltau(X, Y, variant='c')
    return corr
