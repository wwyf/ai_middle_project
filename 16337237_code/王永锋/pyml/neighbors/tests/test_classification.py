import unittest
import numpy as np

from pyml.neighbors.classification import KNeighborsClassifier

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
            1,2,3,4,5,6,7
        ])
        mini_test_X = np.array([
            [2,3,4,5,6,7.5,8,9],
            [4,5,6,7.5,8,9,10,11]
        ])
        mini_standard_out_Y = np.array([
            2,4
        ])
        k_clf = KNeighborsClassifier()
        k_clf.fit(mini_train_X, mini_train_Y)
        Y_pred = k_clf.predict(mini_test_X)
        # 默认，要7位有效数字都要相同
        np.testing.assert_equal(mini_standard_out_Y, Y_pred)
    def test_KNeighborsClassifier_no_parameters_k3_multi_neighbors(self):
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
            1,2,3,4,3,6,7
        ])
        mini_test_X = np.array([
            [2,3,4,5,6,7.5,8,9],
            [4,5,6,7.5,8,9,10,11]
        ])
        mini_standard_out_Y = np.array([
            2,3
        ])
        k_clf = KNeighborsClassifier(k=3)
        k_clf.fit(mini_train_X, mini_train_Y)
        Y_pred = k_clf.predict(mini_test_X)
        # 默认，要7位有效数字都要相同
        np.testing.assert_equal(mini_standard_out_Y, Y_pred)

if __name__ == '__main__':
    unittest.main()

