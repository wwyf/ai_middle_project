import unittest
import numpy as np
from math import sqrt

from pyml.metrics.pairwise import euclidean_distance
from pyml.metrics.pairwise import absolute_distance
from pyml.metrics.pairwise import l_p_distance
from pyml.metrics.pairwise import cosine_similarity
from pyml.metrics.pairwise import cosine_distance

class test_pairwise(unittest.TestCase):

    def test_cosine_similarity(self):
        v1 = np.array([[0,1]])
        v2 = np.array([[1,0]])
        # 默认，要7位有效数字都要相同
        self.assertAlmostEqual(cosine_similarity(v1,v2),0)
        v1 = np.array([[1,1]])
        v2 = np.array([[1,1]])
        # 默认，要7位有效数字都要相同
        self.assertAlmostEqual(cosine_similarity(v1,v2),1)
        v1 = np.array([[1,0]])
        v2 = np.array([[sqrt(2)/2 ,sqrt(2)/2]])
        # 默认，要7位有效数字都要相同
        self.assertAlmostEqual(cosine_similarity(v1,v2), sqrt(2)/2)

    def test_cosine_distance(self):
        v1 = np.array([[1,1]])
        v2 = np.array([[1,0]])
        # 默认，要7位有效数字都要相同
        self.assertAlmostEqual(cosine_distance(v1,v2),0.29289321)

    def test_l_p_distance(self):
        v1 = np.array([[2,4]])
        v2 = np.array([[1,6]])
        self.assertAlmostEqual(l_p_distance(v1,v2),3)

    def test_euclidean_distance(self):
        v1 = np.array([[1,1]])
        v2 = np.array([[0,0]])
        self.assertAlmostEqual(euclidean_distance(v1,v2),1.41421356)
    
    def test_absolute_distance(self):
        v1 = np.array([[2,4]])
        v2 = np.array([[1,6]])
        self.assertAlmostEqual(absolute_distance(v1, v2), 3)
        
if __name__ == '__main__':
    unittest.main()

