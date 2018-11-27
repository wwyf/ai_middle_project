import unittest
import numpy as np

from pyml.metrics.classification import precision_score

class test_classification(unittest.TestCase):

    def test_precision_score(self):
        y_pred = np.array([1,2,3,4,5,6,3,1])
        y_true = np.array([1,2,3,4,5,6,4,1])
        # 默认，要7位有效数字都要相同
        self.assertAlmostEqual(precision_score(y_true,y_pred),7/8)

if __name__ == '__main__':
    unittest.main()

