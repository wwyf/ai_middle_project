import unittest
import numpy as np

from pyml.neighbors.regression import KNeighborsRegressor

class test_classification(unittest.TestCase):

    def test_KNeighborsClassifier_no_parameters(self):
        mini_train_X = np.array([
            [1,2,3,4,5,6,7,8],
            [2,3,4,5,6,7,8,9],
            [3,4,5,6,7,8,9,10],
            [4,5,6,7,8,9,10,11],
            [5,6,7,8,9,10,11,12],
            [6,7,8,9,10,11,12,13],
            [7,8,9,10,11,12,13,14]
        ])
        mini_train_Y = np.array([
            [1,2,3],
            [2,3,4],
            [3,4,5],
            [4,5,6],
            [5,6,7],
            [6,7,8],
            [7,8,9]
        ])
        mini_test_X = np.array([
            [2,3,4,5,6,7.5,8,9],
            [4,5,6,7.5,8,9,10,11]
        ])
        mini_standard_out_Y = np.array([[2.041796, 3.041796, 4.041796],
       [4.041796, 5.041796, 6.041796]])
        k_clf = KNeighborsRegressor(k=2)
        k_clf.fit(mini_train_X, mini_train_Y)
        Y_pred = k_clf.predict(mini_test_X)
        # 默认，要7位有效数字都要相同,改成了6位
        np.testing.assert_almost_equal(mini_standard_out_Y, Y_pred, decimal=6)

if __name__ == '__main__':
    unittest.main()

