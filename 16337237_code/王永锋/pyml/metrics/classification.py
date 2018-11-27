import numpy as np
from pyml.logger import logger


def precision_score(y_true, y_pred):
    """Compute the precision

    适用于分类问题，直接计算预测出来的Y的正确率

    Parameters
    --------------

    y_true : 1d array-like
        Ground truth (correct) target values.

    y_pred : 1d array-like
        Estimated targets as returned by a classifier.

    Return
    -------
        accuracy_rate : double
    """
    y_true = y_true.reshape(-1).astype(int)
    y_pred = y_pred.reshape(-1).astype(int)
    assert(len(y_pred) == len(y_true))
    # assert(len(y_pred.shape) == 1)
    # assert(len(y_true.shape) == 1)
    total_num = len(y_pred)
    success_num = 0
    for i in range(0, total_num):
        if (y_true[i] == y_pred[i]):
            success_num += 1
    logger.debug('y_true : {}'.format(y_true))
    logger.debug('y_pred : {}'.format(y_pred))
    result = float(success_num)/total_num
    logger.debug('result : {}'.format(result))
    return result

