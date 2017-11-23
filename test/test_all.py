"""
Test all items
"""

import os
import unittest

# Change directory to unit tests exist
os.chdir(os.path.join(os.path.dirname(__file__), '.'))

unittest.TextTestRunner(buffer=True, verbosity=2).run(unittest.TestLoader().discover(start_dir='.', pattern='*_test.py'))
