import unittest
import test_lwa

def my_module_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_lwa)
    return suite