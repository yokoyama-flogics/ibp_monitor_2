import os
import sys
import unittest

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sigrecd_mig1 import *

class TestSigRec(unittest.TestCase):
    def test_nextday_datestr(self):
        self.assertEqual(nextday_datestr('20171030'), '20171031')
        self.assertEqual(nextday_datestr('20171031'), '20171101')
        self.assertEqual(nextday_datestr('20171231'), '20180101')
        self.assertEqual(nextday_datestr('20180228'), '20180301')
        self.assertEqual(nextday_datestr('20200228'), '20200229')

if __name__ == "__main__":
    unittest.main(buffer=True)
