import unittest
import numpy as np

from pyml.metrics.regression import pearson_correlation

class test_regression(unittest.TestCase):

    def test_pearson_correlation_single_feature(self):
        y_pred = np.array([1,2,3,4,5,6,3,1])
        y_true = np.array([1,2,3,4,5,6,4,1])
        # 默认，要7位有效数字都要相同
        self.assertAlmostEqual(pearson_correlation(y_true,y_pred),0.98122119)

    def test_pearson_correlation_multi_feature(self):
        y_pred = np.array([[1,2,3,4,5,6,3,1],[1,2,3,4,5,6,4,1]]).T
        y_true = np.array([[1,2,3,4,5,6,4,1],[1,2,3,4,5,6,3,1]]).T
        # 默认，要7位有效数字都要相同
        self.assertAlmostEqual(pearson_correlation(y_true,y_pred),0.98122119)


if __name__ == '__main__':
    unittest.main()
