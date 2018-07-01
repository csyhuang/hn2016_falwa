import unittest
import os
__path__=[os.path.dirname(os.path.abspath(__file__))]
from . import test_basis


def my_module_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_basis)
    return suite
