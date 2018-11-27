import numpy as np

def pearson_correlation(y_pred,y_true):
    """ calculation the pearson correlation coefficient

    如果输入的是二维矩阵，那么就计算多个feature的平均相关系数。

    Parameters
    -------------

    y_pred : 1d array-like the vector of predicted y
        or 2d array-like shape(n_samples, n_features)

    y_true : 1d array-like the vector of ground true y
        or 2d array-like shape(n_samples, n_features)

    Returns
    -------------
    
    cor : double

    """

    # solve 1d array-like y
    if (len(y_pred.shape) == len(y_true.shape) and len(y_true.shape) ==1 ):
        result = np.corrcoef(y_pred, y_true)
        return result[0][1]
    # solve 2d array-like y

    # column : features
    # row : sample
    # 计算矩阵的转置
    pre_Y_T = y_pred.T
    validation_Y_T = y_true.T
    # 得到矩阵后计算相关系数
    # 分别计算6个情绪的相关系数
    value_d = y_pred.shape[1] # 数据有多少维度？
    corrcoef_s = np.zeros((1,value_d))
    for i in range(0,value_d):
        result = np.corrcoef(pre_Y_T[i],validation_Y_T[i])
        # print('i: pre:\n')
        # print(pre_Y_T[i])
        # print('i: validation:\n')
        # print(validation_Y_T[i])
        # print(result)
        corrcoef_s[0][i] = result[0][1]
    return np.average(corrcoef_s)