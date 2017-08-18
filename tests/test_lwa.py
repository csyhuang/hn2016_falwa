import unittest
import numpy as np
from math import pi
from hn2016_falwa.api import lwa
from hn2016_falwa import oopinterface

class Test_hn2016_falwa(unittest.TestCase):

    def test1_lwa(self):
        test_vort = (np.ones((5,5))*np.array([1,2,3,4,5])).swapaxes(0,1)
        test_q_part = np.array([1,2,3,4,5])
        self.assertTrue(np.array_equal(lwa(5,5,test_vort,test_q_part,np.ones(5)),np.zeros((5,5))))

    def test2_lwa(self):
        test_vort = (np.ones((5,5))*np.array([1,2,3,4,5])).swapaxes(0,1)
        equal_matrix_b = (np.ones((5,5))*np.array([6.,1.,0.,3.,0.])).swapaxes(0,1)
        test_q_part = np.array([5,4,3,2,1])
        self.assertTrue(np.array_equal(lwa(5,5,test_vort,test_q_part,np.ones(5)),equal_matrix_b))

    def test1_lwa_oop(self):
        test_vort = (np.ones((5,5))*np.array([1,2,3,4,5])).swapaxes(0,1)
        test_q_part = np.array([1,2,3,4,5])
        self.assertTrue(np.array_equal(oopinterface.lwa_shared(5,5,test_vort,test_q_part,np.ones(5)),np.zeros((5,5))))

    def test2_lwa_oop(self):
        test_vort = (np.ones((5,5))*np.array([1,2,3,4,5])).swapaxes(0,1)
        equal_matrix_b = (np.ones((5,5))*np.array([6.,1.,0.,3.,0.])).swapaxes(0,1)
        test_q_part = np.array([5,4,3,2,1])
        self.assertTrue(np.array_equal(oopinterface.lwa_shared(5,5,test_vort,test_q_part,np.ones(5)),equal_matrix_b))

if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(Test_hn2016_falwa)
    unittest.TextTestRunner(verbosity=2).run(suite)
