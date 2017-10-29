import os
import sys
import unittest

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.common import *

class TestLibCommon(unittest.TestCase):
    def test_eprint1(self):
        eprint('Hello')
        self.assertEquals(sys.stderr.getvalue(), 'Hello\n')

    def test_eprint2(self):
        eprint(123)
        self.assertEquals(sys.stderr.getvalue(), '123\n')

if __name__ == "__main__":
    unittest.main(buffer=True)
