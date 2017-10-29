"""
Test all items
"""

import unittest

unittest.TextTestRunner(buffer=True, verbosity=2).run(unittest.TestLoader().discover(start_dir='.', pattern='*_test.py'))
