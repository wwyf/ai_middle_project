import unittest
import numpy as np

from pyml.linear_model.classification import sigmoid
from pyml.linear_model.classification import LogisticClassifier


class test_classification(unittest.TestCase):

    def test_sigmoid(self):
        result = sigmoid(np.array([0,2]))
        true_result = np.array([0.5, 0.88079708])
        np.testing.assert_almost_equal(result, true_result)
    def test_propagate(self):
        w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
        test_dw = np.array([[0.99993216],[1.99980262]])
        test_db = 0.49993523062470574
        test_cost = 6.000064773192205
        lc = LogisticClassifier()
        grads, cost = lc.propagate(w, b, X, Y)
        np.testing.assert_array_almost_equal(grads['dw'], test_dw)
        np.testing.assert_array_almost_equal(grads['db'], test_db)
        np.testing.assert_array_almost_equal(cost, test_cost)
    def test_optimier(self):
        w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
        std_w = np.array([[0.1124579 ],[0.23106775]])
        std_b = np.array(1.5593049248448891)
        std_dw = np.array([[0.90158428],[1.76250842]])
        std_db = np.array(0.4304620716786828)
        std_cost = [6.000064773192205]
        lc = LogisticClassifier(learning_rate = 0.009)
        params, grads, costs = lc.optimize(w, b, X, Y, num_iterations= 100)
        np.testing.assert_array_almost_equal(params['w'], std_w)
        np.testing.assert_array_almost_equal(params['b'], std_b)
        np.testing.assert_array_almost_equal(grads['dw'], std_dw)
        np.testing.assert_array_almost_equal(grads['db'], std_db)
        np.testing.assert_array_almost_equal(costs, std_cost)
    def test_pred(self):
        w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
        lc = LogisticClassifier()
        lc.parameters['w'] = w
        lc.parameters['b'] = b
        y_pred = lc.predict(X.T)
        std_y_pred = np.array([1,1])
        np.testing.assert_array_almost_equal(y_pred, std_y_pred)

if __name__ == '__main__':
    unittest.main()

