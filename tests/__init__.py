import unittest
import os
__path__=[os.path.dirname(os.path.abspath(__file__))]
from . import test_basis, test_oopinterface


def my_module_suite():
    return unittest.TestSuite([unittest.TestLoader().loadTestsFromModule(module)
                               for module in [test_basis, test_oopinterface]])
