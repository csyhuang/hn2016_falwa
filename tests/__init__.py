import unittest
#import test_lwa
import os
__path__=[os.path.dirname(os.path.abspath(__file__))]
from . import test_lwa


def my_module_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_lwa)
    return suite
