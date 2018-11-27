import unittest
import numpy as np

from pyml.tree.classification import entropy
from pyml.tree.classification import condition_entropy
from pyml.tree.classification import information_gain
from pyml.tree.classification import gini
from pyml.tree.classification import condition_gini
# from pyml.tree.classification import get_best_split_point
# from pyml.tree.classification import divide_dataset
from pyml.tree.classification import DecisionTreeClassifier

class test_classification(unittest.TestCase):

    def test_entropy(self):
        test_data = np.array(['a','a','a','sdfsd','sdfsd','we'])
        test_result = entropy(test_data)
        self.assertAlmostEqual(test_result, 1.4591479)
    def test_condition_entropy(self):
        left = np.array(['y','y','m','o','o','o','m','y','y','o','y','m','m','o'])
        right = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])
        test_result = condition_entropy(left, right)
        self.assertAlmostEqual(test_result, 0.6935361)
    def test_information_gain(self):
        left = np.array(['y','y','m','o','o','o','m','y','y','o','y','m','m','o'])
        right = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])
        test_result = information_gain(left, right)
        self.assertAlmostEqual(test_result, 0.24674981)
    def test_gini(self):
        test_data = np.array([1,1,1,1,1,1,1,1,1,0,0,0,0,0])
        test_result = gini(test_data)
        self.assertAlmostEqual(test_result, 0.459183673)
    def test_condition_gini(self):
        left = np.array(['y','y','m','o','o','o','m','y','y','o','y','m','m','o'])
        right = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])
        test_result = condition_gini(left, right)
        self.assertAlmostEqual(test_result, 0.34285714)
    def test_DecisionTreeClassifier_id3(self):
        X = np.array([
            ['y','y','m','o','o','o','m','y','y','o','y','m','m','o'], 
            # age 0.246
            [0,0,0,0,1,1,1,0,1,1,1,0,1,0],
            # is student? 0.151
            ['h','h','h','m','l','l','l','m','l','m','m','m','h','m'] 
            # salary 0.029
        ])
        X = X.T
        Y = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])
        dt_clf = DecisionTreeClassifier()
        dt_clf.fit(X,Y, feature_names=['age', 'is_student?', 'salary'])
        y_pred = dt_clf.predict(X)
        test_y_pred = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(y_pred, test_y_pred)
    def test_DecisionTreeClassifier_c45(self):
        X = np.array([
            ['y','y','m','o','o','o','m','y','y','o','y','m','m','o'], 
            # age 0.246
            [0,0,0,0,1,1,1,0,1,1,1,0,1,0],
            # is student? 0.151
            ['h','h','h','m','l','l','l','m','l','m','m','m','h','m'] 
            # salary 0.029
        ])
        X = X.T
        Y = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])
        dt_clf = DecisionTreeClassifier(method='c4.5')
        dt_clf.fit(X,Y, feature_names=['age', 'is_student?', 'salary'])
        y_pred = dt_clf.predict(X)
        test_y_pred = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(y_pred, test_y_pred)


if __name__ == '__main__':
    unittest.main()

