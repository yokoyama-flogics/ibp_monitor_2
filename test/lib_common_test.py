import sys
import unittest

sys.path.append('..')
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
